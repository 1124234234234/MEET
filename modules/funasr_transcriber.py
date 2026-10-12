"""
实时语音转写（流式）。

流程：浏览器每 500ms 发送一块 16bit PCM 音频 → 服务端按静音断句累积 →
调用本地 FunASR 识别（含标点恢复）→ 推送分句结果 → 停止时保存音频、
用整段录音重新识别一遍并执行完整分析（说话人分离 / 摘要 / 合规）后写库。

断句与识别策略（实测依据见 modules/asr_engine.group_speech_regions）：
  - 静音判据按近期语音电平自适应，不再用固定 RMS 阈值 —— 固定阈值在
    手机外放教学视频（背景音乐把电平抬到 0.06 以上）或音量偏低的录音里
    会持续判错，把一句话切成 0.3~1 秒的碎片；
  - 缓冲先经 VAD 确认有语音才送识别，纯静音/纯噪声不识别，
    避免识别模型吐幻觉文本（旧版会输出「分下。」「开把星。」这类内容）；
  - 缓冲交给引擎时先并成「一段连续发言」再整体解码，长上下文更准；
  - 缓冲区够长时只在「停顿处」切一段送识别，剩下的音频留到下一轮，
    绝不在句子中间切开 —— 切开处的字会被前后两个单元各识别半截。

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

# 连续静音超过该时长即认为一段话说完，立刻送识别。0.35 秒太短：一句话中间的
# 换气、语气停顿都会被当成句尾，实测把讲解切成 0.3~1 秒的碎片。
SILENCE_SECONDS = 0.6
# 静音按 20ms 帧统计，而不是按音频块（块大小 500ms，一个 0.45 秒的句间停顿
# 会被摊到两块里，每块都因含有语音而被判成「有语音」，停顿就永远检测不到）。
SILENCE_FRAME_MS = 20
# 静音能量判据的绝对下限（16bit PCM 归一化后的 RMS）
SILENCE_RMS_FLOOR = 0.01
# 静音能量判据的相对部分：低于「近期语音电平」的该比例视为静音。
# 固定阈值（旧版 0.012）在电平忽大忽小的录音里完全失效 —— 例如手机外放
# 教学视频，背景音乐把 RMS 抬到 0.06 以上，固定阈值几乎判不出静音；
# 反过来音量偏低的录音又会被判成一片静音。故改为按近期电平自适应。
SILENCE_REL_RATIO = 0.30
# 计算近期语音电平的滑动窗口长度（块数）与参考分位
LEVEL_WINDOW_CHUNKS = 40
LEVEL_PERCENTILE = 75
# 识别单元的软目标：缓冲区够长了就找一处停顿切一段送识别，剩余音频留到
# 下一轮。识别单元越长越准（见 asr_engine.group_speech_regions），但现场
# 展示不能等太久，6 秒是「够长」和「够快」的折中。
TARGET_UNIT_SECONDS = 6.0
# 识别单元的硬上限：连续说话一直不停顿时也必须切一刀，避免久不出结果。
MAX_UNIT_SECONDS = 15.0
# 在停顿处切分时留给下一轮的最短音频，避免切出很短的尾块
MIN_TAIL_SECONDS = 1.5
# 停顿判据：两个语音片段之间至少这么长才算可以用来切分的停顿
MIN_PAUSE_SECONDS = 0.25
# 连续说话时两次寻找切点的最小间隔，避免每个音频块都跑一遍 VAD
TARGET_CHECK_INTERVAL_SECONDS = 1.0
# 最短识别片段，过短不作为独立分句
MIN_CHUNK_SECONDS = 0.4
# 实时说话人标注的刷新间隔（秒）：间隔太短会频繁重算声纹特征，
# 太长则用户看不到说话人区分。1 分钟音频重算一次约 0.2~0.3s，取自适应值。
SPEAKER_REFRESH_SECONDS = 15.0
# 录到该长度之后才开始做说话人标注（太短判不准，也没有区分价值）
SPEAKER_MIN_SECONDS = 8.0


def _pick_cut_seconds(regions, duration):
    """
    在停顿处挑一个切点：尽量靠后，但要给下一轮留够至少 MIN_TAIL_SECONDS。

    只选「两段语音之间确实有停顿」的位置，绝不从一句话中间切开 ——
    切开处的字会被两个单元各识别半截，既丢字又前后串词（实测这是
    实时文本出现半截句子的主因）。
    """
    cut = None
    for index in range(len(regions) - 1):
        end = regions[index][1]
        pause = regions[index + 1][0] - end
        if pause >= MIN_PAUSE_SECONDS and end <= duration - MIN_TAIL_SECONDS:
            cut = end
    return cut


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
        # 近期块能量（用于自适应静音阈值，见 _silence_threshold）
        self.level_history = []
        # 上次在软目标处寻找切点时的缓冲长度
        self.last_target_check = 0.0
        # 说话人标注（实时增量）：已刷新到的音频长度、当前刷新出的说话人分段
        self.speaker_refresh_seconds = 0.0
        self.speaker_segments = []

    # ---------- 说话人（实时增量标注） ----------

    def _refresh_speaker_labels(self, force=False):
        """
        对「已经录到的全部音频」重跑一次说话人分离，并刷新所有已出分句的说话人。

        为什么这样做：边说边转的时候只有一小段缓冲，声纹特征太少、判不准；
        但整段录音累积起来之后，用与停止后完全相同的离线算法（1.5s 窗口池化 +
        余弦聚类 + 轮廓系数定人数）就能得到与最终结果一致的标注 —— 实测
        未知说话人时不再全部显示 SPEAKER_00，用户录到一半就能看出几个人在说话。

        代价：每次刷新要重算整段音频的声纹特征（1 分钟音频约 0.2~0.3s），
        因此按 SPEAKER_REFRESH_SECONDS 间隔触发，而不是每块都算。
        """
        if not self.transcription_history or not self.all_audio_data:
            return False
        recorded = sum(len(chunk) for chunk in self.all_audio_data) / self.sr
        if not force and recorded - self.speaker_refresh_seconds < SPEAKER_REFRESH_SECONDS:
            return False
        if not force and recorded < SPEAKER_MIN_SECONDS:
            return False

        self.speaker_refresh_seconds = recorded
        try:
            from modules import speaker_diarization

            audio = np.concatenate(self.all_audio_data)
            turns = speaker_diarization.speaker_diarization_simple(
                audio=audio, sr=self.sr, timeout=60)
            if not turns:
                return False
            self.speaker_segments = turns
            labelled = speaker_diarization.assign_speakers_to_segments(
                [dict(segment) for segment in self.transcription_history], turns)
            for target, updated in zip(self.transcription_history, labelled):
                target['speaker'] = updated.get('speaker') or target.get('speaker')
            return True
        except Exception as exc:
            print(f'[实时转写] 说话人刷新失败: {exc}')
            return False

    def _history_payload(self):
        """当前全部分句（含最新说话人标注），用于让前端整段刷新。"""
        return [{
            'text': segment['text'],
            'start': segment['start_time'],
            'end': segment['end_time'],
            'speaker': segment.get('speaker') or 'SPEAKER_00',
            'confidence': segment.get('confidence', 1.0),
        } for segment in self.transcription_history]

    # ---------- 音频接收 ----------

    def _silence_threshold(self):
        """按近期语音电平算出静音判据（绝对下限与相对比例取大者）。"""
        if not self.level_history:
            return SILENCE_RMS_FLOOR
        reference = float(np.percentile(self.level_history, LEVEL_PERCENTILE))
        return max(SILENCE_RMS_FLOOR, SILENCE_REL_RATIO * reference)

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
            chunk_rms = float(np.sqrt(np.mean(audio_np ** 2)))
            self.level_history.append(chunk_rms)
            if len(self.level_history) > LEVEL_WINDOW_CHUNKS:
                del self.level_history[:-LEVEL_WINDOW_CHUNKS]

            # 逐帧（20ms）统计静音：块粒度太粗，句间停顿会漏检
            threshold = self._silence_threshold()
            frame = max(1, int(self.sr * SILENCE_FRAME_MS / 1000))
            frame_count = len(audio_np) // frame
            if frame_count:
                frames = audio_np[:frame_count * frame].reshape(frame_count, frame)
                frame_rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
            else:
                frame_rms = np.array([chunk_rms], dtype=np.float64)
            for value in frame_rms:
                if value > threshold:
                    self.is_speaking = True
                    self.silence_samples = 0
                else:
                    self.silence_samples += frame

            buffered_seconds = sum(len(chunk) for chunk in self.buffer) / self.sr
            reason = None
            if self.is_speaking and self.silence_samples >= SILENCE_SECONDS * self.sr:
                # 一段话说完：立刻送识别
                reason = 'silence'
            elif buffered_seconds >= MAX_UNIT_SECONDS:
                # 连续说话到达硬上限：必须出结果
                reason = 'max_unit'
            elif (buffered_seconds >= TARGET_UNIT_SECONDS
                  and buffered_seconds - self.last_target_check >= TARGET_CHECK_INTERVAL_SECONDS):
                # 够长了：试着在停顿处切一段，切不了就继续攒（见 _process_buffer）
                reason = 'target'
                self.last_target_check = buffered_seconds

            if reason is None:
                return None

            return self._process_buffer(reason)
        except Exception as exc:
            print(f'[实时转写] 音频块处理错误: {exc}')
            return None

    # ---------- 识别 ----------

    def _drop_buffer(self, seconds):
        """丢弃缓冲内容（无语音或过短），时长照常推进，避免时间戳错位。"""
        self.buffer = []
        self.committed_seconds += seconds
        self.is_speaking = False
        self.silence_samples = 0
        self.last_target_check = 0.0

    def _process_buffer(self, reason='force'):
        """
        识别缓冲里已经说完的一段，并把结果落到分句历史。

        reason='silence' / 'max_unit' / 'force' 时整块送识别；
        reason='target' 时只在停顿处切一段（剩余音频留到下一轮），
        找不到合适停顿就返回 None 继续攒音频 —— 宁肯晚一点出结果，
        也不从一句话中间切开。
        """
        if not self.buffer:
            return None

        audio = np.concatenate(self.buffer)
        duration = len(audio) / self.sr

        if duration < MIN_CHUNK_SECONDS:
            self._drop_buffer(duration)
            return None

        from modules import asr_engine

        # 先确认缓冲里确实有语音：整块是静音或纯噪声时不识别，
        # 否则识别模型会吐幻觉文本（实测出现过「分下。」「开把星。」）。
        regions = asr_engine.detect_speech_regions(audio, sr=self.sr, fallback_full=False)
        if not regions:
            print(f'[实时转写] 缓冲 {duration:.1f}s 内未检出语音，跳过识别')
            self._drop_buffer(duration)
            return None

        cut = None
        if reason == 'target':
            cut = _pick_cut_seconds(regions, duration)
            if cut is None:
                # 还没到硬上限，且当前没有可用的停顿切点：等更多音频
                return None

        head = audio if cut is None else audio[:int(round(cut * self.sr))]
        tail = None if cut is None else audio[int(round(cut * self.sr)):]
        head_regions = regions if cut is None else [r for r in regions if r[0] < cut]
        head_seconds = len(head) / self.sr

        # 分句时间戳的起点在推进偏移之前取（head 从当前偏移处开始）
        offset = self.committed_seconds

        # 剩余音频留到下一轮；下一轮的分句时间戳从新的偏移继续
        self.buffer = [tail] if tail is not None and len(tail) else []
        self.committed_seconds = offset + head_seconds
        self.is_speaking = False
        self.silence_samples = 0
        self.last_target_check = 0.0

        segments = self._transcribe(head, head_regions, offset)
        if not segments:
            return None

        for segment in segments:
            self.transcription_history.append(segment)

        # 说话人标注（增量）：按间隔对已录音频重跑一次分离，刷新全部分句的说话人。
        # 这样录制过程中就能看到谁在说，而不是一律显示 SPEAKER_00。
        speakers_updated = self._refresh_speaker_labels()

        text = ''.join(segment['text'] for segment in segments)

        # 实时合规比对：边转写边与知识库对照，命中风险词或覆盖到必传要点
        # 立刻回传，便于会议进行中当场提醒；时间戳即该句的起止时间，
        # 会后可按时间点回溯核查。
        compliance_alerts = self._check_live_compliance(segments)

        return {
            'text': text,
            'speaker_updated': speakers_updated,
            'history': self._history_payload(),
            'segments': [{
                'text': segment['text'],
                'start': segment['start_time'],
                'end': segment['end_time'],
                'speaker': segment.get('speaker') or 'SPEAKER_00',
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

    def _transcribe(self, audio, regions, offset):
        """
        调用本地识别引擎（带标点恢复与热词）。

        把这段音频作为一个「连续发言」交给引擎：引擎内部会把其中的语音片段
        并成一个识别单元统一解码（长上下文），再按片段比例还原成带时间戳的
        分句。旧版让引擎对缓冲做细粒度切分、逐片解码，一句话被切成 0.3~1 秒
        的碎片，既是垃圾文本的根源，也慢好几倍。
        offset 是这段音频在整场会议中的起点（秒），返回的时间戳由此换算成绝对时间。
        """
        try:
            from modules import asr_engine

            engine = asr_engine.get_engine('auto')
            result = engine.transcribe_array(
                audio,
                sr=self.sr,
                language=self.language,
                hotwords=self.hotwords,
                regions=regions,
            )
        except Exception as exc:
            print(f'[实时转写] 识别失败: {exc}')
            return []

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
        # force：停止时必须把剩下的音频全部识别掉，哪怕还没到软目标。
        # 冲刷出的分句已经进了 transcription_history（见 _process_buffer），
        # 因此这里不能再把返回结果拼一遍 —— 旧版就是又拼了一次，
        # 导致最后一句在会议记录里出现两遍。
        if self.buffer:
            self._process_buffer('force')

        # 收尾时再刷新一次说话人：保证最终结果里的说话人与整段录音一致
        self._refresh_speaker_labels(force=True)

        transcriptions = [dict(seg) for seg in self.transcription_history]
        full_text = ''.join(seg['text'] for seg in transcriptions)

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
                #
                # 不传实时转写文本：实时文本是「边说边出」的中间结果（缓冲区
                # 小、上下文短，碎片较多），这里用整段录音按文件链路重转写
                # 一遍，会议记录、摘要、合规检查全部基于这份更准的文本；
                # 实时文本只负责现场显示。重转写失败时自动退回实时文本。
                with flask_app.app_context():
                    analysis = analyze_audio(
                        audio_file_path,
                        language=transcriber.language,
                        knowledge_items=transcriber.knowledge_items,
                        score_weights=transcriber.score_weights,
                        progress_callback=progress,
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
