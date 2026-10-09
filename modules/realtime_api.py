"""
实时转写的 HTTP 接口（供内置 Web 页面使用）。

为什么不用 Socket.IO：
    页面里的实时转写原先依赖 socket.io 的浏览器客户端，而那个脚本只能从
    CDN 加载。本项目定位是离线可用，离线环境下脚本加载失败会让整个实时功能
    永久卡在 await 上（点了没反应）。socket.io 客户端在项目里也没有本地副本。

    因此内置页面改用「HTTP 分块上传 + 轮询」：浏览器采集 PCM 后按块 POST，
    服务端复用同一个实时转写会话对象；停止后用轮询获取分析进度与结果。
    只用浏览器自带的 fetch / getUserMedia / AudioContext，零外部依赖。

    Socket.IO 的那套事件仍然保留（见 modules/funasr_transcriber.py），
    供已有第三方客户端继续使用；两条通道共用同一个转写会话实现。

会话状态放在进程内存里，适合单机部署（与本项目的部署形态一致）。
"""
import base64
import os
import threading
import time
import uuid

from modules.audio_io import DEFAULT_SR
from modules.funasr_transcriber import FunASRRealtimeTranscriber

# 会话与结果表
_sessions = {}
_results = {}
_lock = threading.Lock()

# 结果保留时长（秒），过期后清理，避免长时间运行内存增长
_RESULT_TTL = 3600


def _prune():
    """清理过期结果。"""
    now = time.time()
    for sid in [k for k, v in _results.items() if now - v.get('updated_at', 0) > _RESULT_TTL]:
        _results.pop(sid, None)


def _decode_audio(payload):
    """把 base64（可带 data: 前缀）解码为 bytes。"""
    if not isinstance(payload, str) or not payload:
        raise ValueError('audio 必须是非空 base64 字符串')
    data = payload.split(',', 1)[-1] if payload.startswith('data:') else payload
    return base64.b64decode(data)


def start_session(*, upload_folder, language='zh', sample_rate=DEFAULT_SR,
                  meeting_title=None, hotwords=None, enable_compliance=True,
                  knowledge_items=None, score_weights=None):
    """创建实时转写会话，返回会话信息。"""
    _prune()

    os.makedirs(upload_folder, exist_ok=True)
    session_id = str(uuid.uuid4())
    audio_path = os.path.join(upload_folder, f'{session_id}_realtime.wav')

    # 热词增强：知识库关键词 + 配置默认行业词汇 + 调用方额外指定
    from modules import asr_engine

    hotwords = asr_engine.build_hotwords(knowledge_items, extra=hotwords)

    transcriber = FunASRRealtimeTranscriber(
        language=language,
        knowledge_items=knowledge_items or [],
        audio_file_path=audio_path,
        score_weights=score_weights,
        meeting_title=meeting_title or '实时会议',
        hotwords=hotwords,
        sample_rate=sample_rate,
    )

    with _lock:
        _sessions[session_id] = transcriber
        _results[session_id] = {
            'status': 'recording',
            'progress': 0,
            'message': '录音中...',
            'meeting_id': None,
            'result': None,
            'updated_at': time.time(),
        }

    print(f'[实时转写] HTTP 会话开始 {session_id}，采样率 {sample_rate}，音频 {audio_path}')
    return {
        'session_id': session_id,
        'language': language,
        'sample_rate': transcriber.sr,
        'meeting_title': transcriber.meeting_title,
        'compliance_enabled': bool(transcriber.knowledge_items),
    }


def push_chunk(session_id, audio_payload):
    """接收一块音频并返回识别结果（可能为 None 表示暂无输出）。"""
    transcriber = _sessions.get(session_id)
    if transcriber is None:
        raise KeyError(session_id)

    audio_bytes = _decode_audio(audio_payload)
    return transcriber.add_audio_chunk(audio_bytes)


def stop_session(session_id, *, app, socketio=None, on_done=None):
    """
    结束会话：保存音频并转入后台完整分析。

    app 用于在新线程内建立应用上下文（数据库访问需要）。
    返回 (状态字典, meeting_id占位None)。
    """
    transcriber = _sessions.pop(session_id, None)
    if transcriber is None:
        raise KeyError(session_id)

    audio_file_path = transcriber.save_audio_file()
    final = transcriber.get_final_result()
    duration = transcriber.duration_seconds()

    info = {
        'audio_file': audio_file_path,
        'audio_duration': round(duration, 2),
        'text': final.get('text', ''),
        'transcriptions': final.get('transcriptions', []),
        'meeting_title': transcriber.meeting_title,
    }

    has_content = bool((info['text'] or '').strip()) or bool(info['transcriptions'])
    if not has_content:
        # 没有有效语音：不写库、不留静音音频
        print('[实时转写] HTTP 会话没有有效语音内容，跳过入库')
        if audio_file_path and os.path.exists(audio_file_path):
            try:
                os.remove(audio_file_path)
            except OSError as exc:
                print(f'[实时转写] 清理空音频失败: {exc}')
        with _lock:
            _results[session_id] = {
                'status': 'failed',
                'progress': -1,
                'message': '没有采集到有效语音内容',
                'meeting_id': None,
                'result': None,
                'updated_at': time.time(),
            }
        return info

    with _lock:
        _results[session_id] = {
            'status': 'analyzing',
            'progress': 10,
            'message': '正在分析...',
            'meeting_id': None,
            'result': None,
            'updated_at': time.time(),
        }

    def _worker():
        from modules.analysis_pipeline import analyze_audio
        from modules.funasr_transcriber import _persist

        def progress(percent, message):
            with _lock:
                entry = _results.get(session_id)
                if entry is not None:
                    entry.update(
                        status='failed' if percent == -1 else 'analyzing',
                        progress=percent,
                        message=message,
                        updated_at=time.time(),
                    )

        result = dict(info)
        try:
            with app.app_context():
                analysis = analyze_audio(
                    audio_file_path,
                    language=transcriber.language,
                    knowledge_items=transcriber.knowledge_items,
                    score_weights=transcriber.score_weights,
                    progress_callback=progress,
                    transcription_text=info['text'],
                    transcriptions=info['transcriptions'],
                    enable_diarization=True,
                    hotwords=transcriber.hotwords,
                )

            if analysis:
                result.update(analysis)
                if analysis.get('transcriptions'):
                    result['transcriptions'] = analysis['transcriptions']

            with app.app_context():
                meeting_id = _persist(None, session_id, transcriber, result,
                                      audio_file_path, app=app)

            result['meeting_id'] = meeting_id
            result['id'] = meeting_id

            with _lock:
                entry = _results.get(session_id)
                if entry is not None:
                    entry.update(
                        status='done' if meeting_id else 'failed',
                        progress=100 if meeting_id else -1,
                        message='分析完成' if meeting_id else '保存会议记录失败',
                        meeting_id=meeting_id,
                        result=result,
                        updated_at=time.time(),
                    )
            print(f'[实时转写] HTTP 会话 {session_id} 完成，meeting_id={meeting_id}')

            if on_done:
                try:
                    on_done(result)
                except Exception as exc:
                    print(f'[实时转写] on_done 回调异常: {exc}')

        except Exception as exc:
            print(f'[实时转写] HTTP 会话分析异常: {exc}')
            import traceback
            traceback.print_exc()
            with _lock:
                entry = _results.get(session_id)
                if entry is not None:
                    entry.update(status='failed', progress=-1,
                                 message=f'分析失败: {exc}', updated_at=time.time())

    threading.Thread(target=_worker, daemon=True).start()
    return info


def get_result(session_id):
    """查询会话状态与结果。"""
    entry = _results.get(session_id)
    if entry is None:
        raise KeyError(session_id)
    return {
        'status': entry['status'],
        'progress': entry['progress'],
        'message': entry['message'],
        'meeting_id': entry['meeting_id'],
        'result': entry['result'],
    }


def active_sessions():
    """当前进行中的会话数（用于健康检查/诊断）。"""
    return len(_sessions)


def discard(session_id):
    """放弃会话（用户中途关闭页面等），清理音频文件。"""
    transcriber = _sessions.pop(session_id, None)
    _results.pop(session_id, None)
    if transcriber and transcriber.audio_file_path and os.path.exists(transcriber.audio_file_path):
        try:
            os.remove(transcriber.audio_file_path)
        except OSError:
            pass
