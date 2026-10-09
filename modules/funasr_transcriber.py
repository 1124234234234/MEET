"""
实时语音转写（Socket.IO 流式）。

流程：浏览器每 100ms 发送一块 16bit PCM 音频 → 服务端按静音断句累积 →
调用本地 FunASR 识别（含标点恢复）→ 推送分句结果 → 停止时保存音频、
执行完整分析（说话人分离 / 摘要 / 合规）并写入数据库。

相比旧实现修正的问题：
  1. 旧版本用 `total_start_time` 只在「识别出文字」时才推进，静音或识别
     失败时会漏加时长，后续分句时间戳整体前移、与音频错位；
  2. 旧版本用 `len(all_audio_data) * 0.1` 估算时长，块长实际不固定（浏览器
     定时器抖动），会议时长经常算错；
  3. 旧版本丢弃了前端传入的会议标题，回放列表里全是「实时转写会议 」；
  4. 旧版本用正则给文本硬补标点（在某些字后面无脑加「？」），文本会被改坏，
     现在直接用 FunASR 的标点恢复模型；
  5. 断线时转写器不回收，长时间运行会持续占用内存。
"""
import base64
import os
import threading
import uuid

import numpy as np

from modules.audio_io import DEFAULT_SR, save_wav
from modules.meeting_store import save_meeting

transcribers = {}

# 缓冲区达到该长度即强制识别一次，避免一直等不到静音而不出结果
MAX_BUFFER_SECONDS = 8.0
# 静音判定阈值（16bit PCM 归一化后的 RMS）
SILENCE_RMS = 0.012
# 连续静音超过该时长即认为一句话说完
SILENCE_SECONDS = 0.35
# 最短识别片段，过短不作为独立分句
MIN_CHUNK_SECONDS = 0.4


class FunASRRealtimeTranscriber:
    """实时转写会话状态：音频缓冲、分句历史、音频落盘。"""

    def __init__(self, language='zh', knowledge_items=None, audio_file_path=None,
                 score_weights=None, meeting_title=None, hotwords=None,
                 sample_rate=DEFAULT_SR):
        self.language = language
        self.knowledge_items = knowledge_items or []
        self.score_weights = score_weights
        self.meeting_title = meeting_title or '实时会议'
        self.hotwords = hotwords

        self.buffer = []
        self.audio_file_path = audio_file_path
        self.all_audio_data = []
        # 采集端实际采样率：浏览器可能给到 44100/48000，若一律按 16000 解释，
        # 静音断句阈值、录音时长、分句时间戳都会按错误比例计算。
        # 这里按真实采样率累积，识别时由引擎内部重采样到 16k。
        self.sr = int(sample_rate or DEFAULT_SR)

        self.silence_samples = 0
        self.is_speaking = False
        self.transcription_history = []
        # 实时比对过程中累计的合规告警（风险词命中 / 要点覆盖）
        self.compliance_history = []
        # 已提交识别的时间偏移，按「实际音频长度」推进，避免时间戳漂移
        self.committed_seconds = 0.0

    # ---------- 音频接收 ----------

    def add_audio_chunk(self, audio_data):
        """接收一块音频；达到断句条件时返回识别结果，否则返回 None。"""
        try:
            if isinstance(audio_data, bytes):
                if not audio_data:
                    return None
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                audio_np = np.asarray(audio_data, dtype=np.float32)

            if audio_np.size == 0:
                return None

            self.buffer.append(audio_np)
            self.all_audio_data.append(audio_np.copy())

            rms = float(np.sqrt(np.mean(audio_np ** 2)))
            if rms > SILENCE_RMS:
                self.is_speaking = True
                self.silence_samples = 0
            else:
                self.silence_samples += len(audio_np)

            buffered_samples = sum(len(chunk) for chunk in self.buffer)
            should_process = False
            if self.is_speaking and self.silence_samples >= SILENCE_SECONDS * self.sr:
                should_process = True
            elif buffered_samples >= MAX_BUFFER_SECONDS * self.sr:
                should_process = True

            if not should_process:
                return None

            self.is_speaking = False
            self.silence_samples = 0
            return self._process_buffer()
        except Exception as exc:
            print(f'[实时转写] 音频块处理错误: {exc}')
            return None

    # ---------- 识别 ----------

    def _process_buffer(self):
        """识别当前缓冲区内容，推进时间偏移并累计分句历史。"""
        if not self.buffer:
            return None

        audio = np.concatenate(self.buffer)
        self.buffer = []
        chunk_seconds = len(audio) / self.sr

        if chunk_seconds < MIN_CHUNK_SECONDS:
            # 片段过短：计入时长但不产出分句，避免碎片化与时间戳错位
            self.committed_seconds += chunk_seconds
            return None

        segments = self._transcribe(audio)
        # 关键修正：无论是否识别出文字，音频时长都要推进
        self.committed_seconds += chunk_seconds

        if not segments:
            return None

        for segment in segments:
            self.transcription_history.append(segment)

        text = ''.join(segment['text'] for segment in segments)

        # 实时合规比对：边转写边与知识库对照，命中风险词或覆盖到必传要点
        # 立刻回传，便于会议进行中当场提醒；时间戳即该句的起止时间，
        # 会后可按时间点回溯核查。
        compliance_alerts = self._check_live_compliance(segments)

        return {
            'text': text,
            'segments': [{
                'text': segment['text'],
                'start': segment['start_time'],
                'end': segment['end_time'],
                'confidence': segment['confidence'],
            } for segment in segments],
            'compliance': compliance_alerts,
            'is_final': False,
        }

    def _check_live_compliance(self, segments):
        """对本批分句做实时合规检查，返回告警列表（无知识库时返回空）。"""
        if not self.knowledge_items:
            return None

        try:
            from modules.compliance_checker import realtime_compliance_check

            alerts = []
            for segment in segments:
                result = realtime_compliance_check(
                    segment['text'],
                    self.knowledge_items,
                    segment['start_time'],
                    segment['end_time'],
                )
                if result.get('has_risk') or result.get('covered_points'):
                    alerts.append(result)
                    self.compliance_history.append(result)
            return alerts or None
        except Exception as exc:
            print(f'[实时转写] 实时合规检查失败: {exc}')
            return None

    def _transcribe(self, audio):
        """
        调用本地识别引擎（带标点恢复与热词）。

        交给引擎自己的 VAD 切分缓冲音频（而不是把整块当一句），
        这样每句都有独立的起止时间，边界更贴合真实语句；
        与文件级分析走同一条分段逻辑，识别质量也保持一致。
        返回的绝对时间 = 缓冲区内相对时间 + 已提交时长。
        """
        try:
            from modules import asr_engine

            engine = asr_engine.get_engine('auto')
            result = engine.transcribe_array(
                audio,
                sr=self.sr,
                language=self.language,
                hotwords=self.hotwords,
            )
        except Exception as exc:
            print(f'[实时转写] 识别失败: {exc}')
            return []

        offset = self.committed_seconds
        segments = []
        for item in result.get('segments', []):
            text = (item.get('text') or '').strip()
            if not text:
                continue
            segments.append({
                'speaker': 'SPEAKER_00',
                'text': text,
                'start_time': round(offset + item['start'], 2),
                'end_time': round(offset + item['end'], 2),
                'confidence': float(item.get('confidence', 1.0) or 1.0),
            })
        return segments

    # ---------- 收尾 ----------

    def duration_seconds(self):
        """按实际采样点计算录音时长（不使用块数估算）。"""
        total_samples = sum(len(chunk) for chunk in self.all_audio_data)
        return total_samples / self.sr

    def save_audio_file(self):
        """把本次会话收到的全部音频写成一个 wav 文件。"""
        if not self.all_audio_data or not self.audio_file_path:
            return None
        try:
            full_audio = np.concatenate(self.all_audio_data)
            save_wav(self.audio_file_path, full_audio, self.sr)
            print(f'[实时转写] 音频已保存: {self.audio_file_path}')
            return self.audio_file_path
        except Exception as exc:
            print(f'[实时转写] 保存音频失败: {exc}')
            return None

    def get_final_result(self):
        """冲刷剩余缓冲并返回最终转写结果。"""
        tail = self._process_buffer() if self.buffer else None

        transcriptions = [dict(seg) for seg in self.transcription_history]
        full_text = ''.join(seg['text'] for seg in transcriptions)
        if tail:
            full_text += tail['text']

        return {
            'text': full_text.strip(),
            'transcriptions': transcriptions,
            'segments': [
                {'text': s['text'], 'start': s['start_time'],
                 'end': s['end_time'], 'confidence': s['confidence']}
                for s in transcriptions
            ],
            'compliance_history': list(self.compliance_history),
            'is_final': True,
        }


def register_socketio_events(socketio):
    """注册实时转写相关的 Socket.IO 事件。"""

    @socketio.on('connect')
    def handle_connect():
        print('[实时转写] 客户端已连接')

    @socketio.on('disconnect')
    def handle_disconnect():
        from flask import request

        sid = getattr(request, 'sid', None)
        print(f'[实时转写] 客户端断开: {sid}')
        # 回收会话，避免长时间运行内存持续增长
        if sid and sid in transcribers:
            del transcribers[sid]

    @socketio.on('start_transcription')
    def handle_start(data):
        from flask import request

        from models import KnowledgeBase, ScoreWeight

        data = data or {}
        sid = request.sid
        language = data.get('language', 'zh')
        enable_compliance = data.get('enable_compliance', True)
        meeting_title = data.get('meeting_title') or '实时会议'
        hotwords = data.get('hotwords')

        try:
            from flask import current_app

            upload_folder = current_app.config.get('UPLOAD_FOLDER', 'uploads')
            default_weights = current_app.config.get('SCORE_WEIGHTS')
        except Exception:
            upload_folder = 'uploads'
            default_weights = None

        os.makedirs(upload_folder, exist_ok=True)
        file_id = str(uuid.uuid4())
        audio_file_path = os.path.join(upload_folder, f'{file_id}_realtime.wav')

        knowledge_items = []
        score_weights = None
        if enable_compliance:
            knowledge_items = KnowledgeBase.query.filter_by(status='active').all()
            db_weights = ScoreWeight.query.all()
            if db_weights:
                score_weights = {w.weight_name: w.weight_value for w in db_weights}
            else:
                score_weights = default_weights

        transcribers[sid] = FunASRRealtimeTranscriber(
            language=language,
            knowledge_items=knowledge_items,
            audio_file_path=audio_file_path,
            score_weights=score_weights,
            meeting_title=meeting_title,
            hotwords=hotwords,
        )

        socketio.emit('transcription_started', {
            'status': 'started',
            'language': language,
            'compliance_enabled': enable_compliance,
            'meeting_title': meeting_title,
        }, to=sid)
        print(f'[实时转写] 会话开始 {sid}，音频: {audio_file_path}')

    @socketio.on('audio_chunk')
    def handle_audio_chunk(data):
        from flask import request

        sid = request.sid
        if sid not in transcribers:
            socketio.emit('error', {'message': '请先开始转写会话'}, to=sid)
            return

        try:
            if isinstance(data, dict) and 'audio' in data:
                audio_bytes = base64.b64decode(data['audio'])
            elif isinstance(data, str):
                audio_bytes = base64.b64decode(data)
            else:
                audio_bytes = data

            result = transcribers[sid].add_audio_chunk(audio_bytes)
            if result:
                socketio.emit('transcription_result', result, to=sid)
        except Exception as exc:
            print(f'[实时转写] 音频块错误: {exc}')
            socketio.emit('error', {'message': f'音频处理错误: {exc}'}, to=sid)

    @socketio.on('stop_transcription')
    def handle_stop():
        from flask import current_app, request

        sid = request.sid
        print(f'[实时转写] 收到停止请求 {sid}')

        if sid not in transcribers:
            socketio.emit('error', {'message': '没有活跃的转写会话'}, to=sid)
            return

        transcriber = transcribers[sid]
        try:
            audio_file_path = transcriber.save_audio_file()
        except Exception as exc:
            print(f'[实时转写] 保存音频失败: {exc}')
            audio_file_path = None

        try:
            result = transcriber.get_final_result()
        except Exception as exc:
            print(f'[实时转写] 获取最终结果失败: {exc}')
            result = {'text': '', 'transcriptions': [], 'segments': []}

        duration = transcriber.duration_seconds()
        result['audio_file'] = audio_file_path
        result['audio_duration'] = round(duration, 2)
        result['duration'] = int(duration)
        result['meeting_title'] = transcriber.meeting_title

        socketio.emit('transcription_stopped', {
            'audio_file': audio_file_path,
            'audio_duration': result['audio_duration'],
            'text': result.get('text', ''),
        }, to=sid)

        if not audio_file_path or not os.path.exists(audio_file_path):
            # 没有音频可分析（例如未采集到任何数据），直接收尾
            _persist(socketio, sid, transcriber, result, audio_file_path, app=None)
            return

        if not result.get('text', '').strip() and not result.get('transcriptions'):
            # 用户开始后直接停止、全程没有有效语音：不写库，也不留静音音频文件。
            # 补齐与正常结果一致的键，避免客户端按字段读取时抛异常。
            print('[实时转写] 会话没有有效语音内容，跳过入库')
            if audio_file_path and os.path.exists(audio_file_path):
                try:
                    os.remove(audio_file_path)
                except OSError as exc:
                    print(f'[实时转写] 清理空音频失败: {exc}')
            result.setdefault('keywords', [])
            result.setdefault('topics', [])
            result.setdefault('sentiment', {})
            result.setdefault('action_items', [])
            result.setdefault('decisions', [])
            result.setdefault('summary', '')
            result['compliance_report'] = None
            result['audio_quality'] = None
            result['meeting_id'] = None
            result['audio_file'] = None
            socketio.emit('realtime_analysis_progress',
                          {'progress': -1, 'message': '没有采集到有效语音内容'}, to=sid)
            socketio.emit('transcription_final', result, to=sid)
            transcribers.pop(sid, None)
            return

        def _analyze_async():
            from app import app as flask_app

            try:
                print(f'[实时转写] 开始分析 {audio_file_path}')
                from modules.analysis_pipeline import analyze_audio

                def progress(percent, message):
                    try:
                        socketio.emit('realtime_analysis_progress',
                                      {'progress': percent, 'message': message}, to=sid)
                    except Exception as exc:
                        print(f'推送进度失败: {exc}')

                # 本线程没有 Flask 应用上下文：知识库对象来自之前的请求会话，
                # 脱离上下文/会话后再访问其属性可能抛 DetachedInstanceError，
                # 因此整段分析（含合规比对）都在应用上下文内执行。
                with flask_app.app_context():
                    analysis = analyze_audio(
                        audio_file_path,
                        language=transcriber.language,
                        knowledge_items=transcriber.knowledge_items,
                        score_weights=transcriber.score_weights,
                        progress_callback=progress,
                        transcription_text=result.get('text', ''),
                        transcriptions=result.get('transcriptions', []),
                        enable_diarization=True,
                        hotwords=transcriber.hotwords,
                    )

                    if analysis:
                        result.update(analysis)
                        # 复用已有文本时 analyze_audio 会把说话人标注写回分句
                        if analysis.get('transcriptions'):
                            result['transcriptions'] = analysis['transcriptions']

                    meeting_id = _persist(None, sid, transcriber, result,
                                          audio_file_path, app=flask_app)

                if meeting_id:
                    result['meeting_id'] = meeting_id
                    result['id'] = meeting_id

                socketio.emit('transcription_final', result, to=sid)
                print(f'[实时转写] 已完成并保存 meeting_id={meeting_id}')

            except Exception as exc:
                print(f'[实时转写] 分析线程异常: {exc}')
                import traceback
                traceback.print_exc()
                socketio.emit('realtime_analysis_progress',
                              {'progress': -1, 'message': f'分析失败: {exc}'}, to=sid)
            finally:
                transcribers.pop(sid, None)

        threading.Thread(target=_analyze_async, daemon=True).start()
        print(f'[实时转写] 会话 {sid} 已停止，分析转入后台')


def _persist(socketio, sid, transcriber, result, audio_file_path, app=None):
    """把实时会议写入数据库（标题、时长、待办事项等一并保存）。"""
    try:
        compliance = result.get('compliance_report') or result.get('compliance')
        segments = result.get('transcriptions') or []
        if not segments and not (result.get('text') or '').strip():
            # 兜底：没有任何转写内容就不要建空会议记录
            print('[实时转写] 无有效转写内容，跳过入库')
            return None
        if not segments and result.get('text'):
            segments = [{
                'speaker': 'SPEAKER_00',
                'text': result['text'],
                'start_time': 0,
                'end_time': result.get('audio_duration', 0),
                'confidence': 1.0,
            }]

        meeting_id = save_meeting(
            title=transcriber.meeting_title,
            transcriptions=segments,
            summary=result.get('summary', ''),
            keywords=result.get('keywords', []),
            topics=result.get('topics', []),
            sentiment=result.get('sentiment', {}),
            action_items=result.get('action_items', []),
            decisions=result.get('decisions', []),
            compliance=compliance,
            audio_quality=result.get('audio_quality'),
            audio_path=audio_file_path,
            duration=int(result.get('audio_duration', 0) or 0),
            language=transcriber.language,
        )
        if meeting_id and socketio is not None:
            result['meeting_id'] = meeting_id
            result['id'] = meeting_id
            socketio.emit('transcription_final', result, to=sid)
        return meeting_id
    except Exception as exc:
        print(f'[实时转写] 保存会议失败: {exc}')
        import traceback
        traceback.print_exc()
        return None
