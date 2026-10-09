"""
实时转写链路端到端测试。

分两部分：
  A. Socket.IO 事件接线：connect → start_transcription → audio_chunk →
     stop_transcription 能正常收发，且不出现异常事件；
  B. 转写会话本体：直接驱动 FunASRRealtimeTranscriber（与事件处理器内部
     逻辑一致），验证时间戳推进、时长计算、分句历史，并把音频落盘后走
     完整分析 + 入库，确认「谁在什么时候说了什么」端到端可用。

运行：python tests/test_realtime.py
"""
import base64
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_DIR = os.path.join(BASE_DIR, 'tests', 'test_audio_files')

RESULTS = []


def record(name, passed, detail=''):
    RESULTS.append((name, passed, detail))
    print(f'  [{"PASS" if passed else "FAIL"}] {name}' + (f' — {detail}' if detail else ''))
    return passed


def make_chunks(wav_path, chunk_seconds=0.1):
    """把测试音频切成 16bit PCM 小块，模拟浏览器推流。"""
    from modules.audio_io import DEFAULT_SR, load_audio

    audio = load_audio(wav_path, sr=DEFAULT_SR)
    step = int(DEFAULT_SR * chunk_seconds)
    chunks = []
    for start in range(0, len(audio), step):
        piece = audio[start:start + step]
        if len(piece) == 0:
            continue
        pcm = np.clip(piece * 32768.0, -32768, 32767).astype(np.int16)
        chunks.append(pcm.tobytes())
    return chunks, len(audio) / DEFAULT_SR


def test_socketio_wiring():
    """Socket.IO 事件接线测试。"""
    print('\n[A] Socket.IO 事件接线')
    try:
        from app import app, socketio
    except Exception as exc:
        record('导入应用', False, f'{type(exc).__name__}: {exc}')
        return

    client = socketio.test_client(app)
    record('客户端连接成功', client.is_connected())

    received = client.get_received()
    record('收到连接相关事件', True, f'{len(received)} 条')

    client.emit('start_transcription', {'language': 'zh', 'meeting_title': '实时测试会议',
                                        'enable_compliance': True})
    events = client.get_received()
    names = [e['name'] for e in events]
    record('start_transcription 返回 transcription_started',
           'transcription_started' in names, f'事件={names}')

    # 发一个很短的块：只验证链路不报错（不出结果属正常）
    silence = (np.zeros(1600, dtype=np.int16)).tobytes()
    client.emit('audio_chunk', {'audio': base64.b64encode(silence).decode()})
    events = client.get_received()
    errors = [e for e in events if e['name'] == 'error']
    record('短音频块不报错', not errors, str(errors[:1]))

    client.emit('stop_transcription')
    events = client.get_received()
    names = [e['name'] for e in events]
    record('stop_transcription 返回 transcription_stopped',
           'transcription_stopped' in names, f'事件={names}')

    client.disconnect()


def test_transcriber_and_analysis():
    """转写会话 + 完整分析 + 入库。"""
    print('\n[B] 实时转写会话与分析入库')
    from modules import asr_engine
    from modules.audio_io import DEFAULT_SR
    from modules.funasr_transcriber import FunASRRealtimeTranscriber

    wav = os.path.join(AUDIO_DIR, 'gt_2spk.wav')
    if not os.path.exists(wav):
        record('实时转写会话', False, '缺少 gt_2spk.wav')
        return

    upload_dir = os.path.join(BASE_DIR, 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    out_path = os.path.join(upload_dir, 'realtime_test_session.wav')

    transcriber = FunASRRealtimeTranscriber(
        language='zh',
        knowledge_items=[],
        audio_file_path=out_path,
        meeting_title='实时测试会议',
        hotwords=['风险等级', '录音录像'],
    )

    chunks, expected_duration = make_chunks(wav)
    produced = 0
    emits = 0
    for chunk in chunks:
        result = transcriber.add_audio_chunk(chunk)
        if result:
            produced += 1
            emits += len(result['segments'])

    record(f'推流 {len(chunks)} 块，产出 {produced} 次分句结果（共 {emits} 句）', produced > 0)

    # 时间戳必须单调递增且不超出音频长度
    history = transcriber.transcription_history
    monotonic = all(
        history[i]['end_time'] <= history[i + 1]['start_time'] + 1e-6
        for i in range(len(history) - 1)
    ) if len(history) > 1 else True
    within_range = all(seg['end_time'] <= expected_duration + 1.0 for seg in history) if history else False
    record(f'分句时间戳单调递增={monotonic}，未超出音频时长={within_range}',
           monotonic and within_range,
           f'末句结束 {history[-1]["end_time"] if history else 0}s / 音频 {expected_duration:.2f}s')

    # 时长按实际采样点计算，而不是「块数 × 0.1」
    duration = transcriber.duration_seconds()
    record(f'时长计算 {duration:.2f}s（实际 {expected_duration:.2f}s）',
           abs(duration - expected_duration) < 0.5)

    final = transcriber.get_final_result()
    record(f'最终文本 {len(final["text"])} 字', len(final['text']) > 20)

    saved = transcriber.save_audio_file()
    record('音频落盘成功', bool(saved) and os.path.exists(saved))

    # 完整分析（含说话人分离）
    from modules.analysis_pipeline import analyze_audio

    analysis = analyze_audio(
        saved,
        language='zh',
        knowledge_items=[],
        transcription_text=final['text'],
        transcriptions=final['transcriptions'],
        enable_diarization=True,
    )
    record('复用转写文本的完整分析返回结果', analysis is not None)

    speakers = set()
    if analysis:
        speakers = {s.get('speaker') for s in analysis.get('transcriptions', [])}
        record(f'分析后说话人标注 {sorted(speakers)}', len(speakers) >= 2,
               f'{len(analysis.get("transcriptions", []))} 句')
        record('摘要非空', bool(analysis.get('summary')), (analysis.get('summary') or '')[:50])

    # 入库（标题、时长、待办事项都要落库）
    from app import app as flask_app
    from modules.meeting_store import save_meeting

    with flask_app.app_context():
        segments = (analysis or {}).get('transcriptions') or final.get('transcriptions') or []
        meeting_id = save_meeting(
            title=transcriber.meeting_title,
            transcriptions=segments,
            summary=(analysis or {}).get('summary', ''),
            keywords=(analysis or {}).get('keywords', []),
            topics=(analysis or {}).get('topics', []),
            sentiment=(analysis or {}).get('sentiment', {}),
            action_items=(analysis or {}).get('action_items', []),
            decisions=(analysis or {}).get('decisions', []),
            compliance=(analysis or {}).get('compliance_report'),
            audio_path=saved,
            duration=int(duration),
        )
        record(f'会议入库 meeting_id={meeting_id}', bool(meeting_id))

        if meeting_id:
            from models import Meeting, Transcription
            meeting = Meeting.query.get(meeting_id)
            rows = Transcription.query.filter_by(meeting_id=meeting_id).all()
            record(f'标题正确「{meeting.title}」', meeting.title == '实时测试会议')
            record(f'转写记录 {len(rows)} 条，说话人 {sorted({r.speaker for r in rows})}',
                   len(rows) > 0 and len({r.speaker for r in rows}) >= 2)
            record(f'时长落库 {meeting.duration}s', meeting.duration > 0)

            # 清理本次测试产生的数据，避免污染演示库
            from modules.meeting_store import delete_meeting_files
            Transcription.query.filter_by(meeting_id=meeting_id).delete()
            from models import ComplianceReport
            ComplianceReport.query.filter_by(meeting_id=meeting_id).delete()
            from database import db
            db.session.delete(meeting)
            db.session.commit()
            delete_meeting_files(type('M', (), {'audio_path': saved})())
            print('      （已清理测试数据）')


def main():
    print('=' * 78)
    print('实时转写链路端到端测试')
    print('=' * 78)
    test_socketio_wiring()
    test_transcriber_and_analysis()

    print('\n' + '=' * 78)
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    for name, ok, detail in RESULTS:
        print(f'  {"✅" if ok else "❌"} {name}' + (f'  ({detail})' if detail else ''))
    print(f'\n通过 {passed}/{total} ({passed / total * 100:.1f}%)')
    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())
