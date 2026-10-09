"""
实时转写 HTTP 接口端到端测试。

覆盖新增的 /api/realtime/* 接口（内置页面使用，不依赖 socket.io 客户端）：
  - start / chunk / stop / result / discard 全流程
  - 分句时间戳与录音时长正确（含 48000Hz 采样率场景，验证重采样与时长换算）
  - 停止后的完整分析（说话人分离 + 合规）与入库
  - 空会话（全程静音）不写库、不留音频
  - /api/meetings?include_compliance=1 返回合规摘要

运行：python tests/test_realtime_http.py
"""
import base64
import io
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_DIR = os.path.join(BASE_DIR, 'tests', 'test_audio_files')

RESULTS = []


def record(name, passed, detail=''):
    RESULTS.append((name, passed, detail))
    print(f'  [{"PASS" if passed else "FAIL"}] {name}' + (f' — {detail}' if detail else ''))
    return passed


def pcm_chunks(path, sample_rate, chunk_seconds=0.5):
    """把音频切成 16bit 单声道 PCM 块（base64），模拟浏览器推流。"""
    from modules.audio_io import load_audio, resample

    audio = load_audio(path, sr=sample_rate)
    step = int(sample_rate * chunk_seconds)
    chunks = []
    for start in range(0, len(audio), step):
        piece = audio[start:start + step]
        if len(piece) == 0:
            continue
        pcm = np.clip(piece * 32768.0, -32768, 32767).astype(np.int16)
        chunks.append(base64.b64encode(pcm.tobytes()).decode())
    return chunks, len(audio) / sample_rate


def run_session(client, wav, sample_rate, title, chunk_seconds=0.5):
    """跑一次完整实时会话，返回 (会话信息, 推流产生的分句数, 结束响应, 最终结果)。"""
    response = client.post('/api/realtime/start', json={
        'language': 'zh',
        'sample_rate': sample_rate,
        'meeting_title': title,
        'enable_compliance': True,
    })
    payload = response.get_json() or {}
    session = (payload.get('data') or {}).get('session_id')
    if not session:
        return None, 0, payload, None

    chunks, expected = pcm_chunks(wav, sample_rate, chunk_seconds)
    produced = 0
    for chunk in chunks:
        resp = client.post('/api/realtime/chunk',
                           json={'session_id': session, 'audio': chunk})
        if resp.status_code != 200:
            print('      chunk 失败:', resp.get_data(as_text=True)[:200])
            break
        data = (resp.get_json() or {}).get('data')
        if data and data.get('segments'):
            produced += len(data['segments'])

    stop = client.post('/api/realtime/stop', json={'session_id': session})
    final = poll_result(client, session)
    return {'session_id': session, 'expected': expected}, produced, stop.get_json(), final


def poll_result(client, session_id, timeout=300):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        resp = client.get(f'/api/realtime/result/{session_id}')
        if resp.status_code != 200:
            return None
        data = (resp.get_json() or {}).get('data') or {}
        last = data
        if data.get('status') in ('done', 'failed'):
            return data
        time.sleep(2)
    return last


def test_realtime_http_flow():
    print('\n[1] 实时转写 HTTP 全流程（16000Hz）')
    from app import app

    wav = os.path.join(AUDIO_DIR, 'gt_2spk.wav')
    if not os.path.exists(wav):
        record('实时 HTTP 全流程', False, '缺少 gt_2spk.wav')
        return

    client = app.test_client()
    meeting_id = None
    try:
        session, produced, stop, final = run_session(
            client, wav, 16000, '实时HTTP测试-16k', chunk_seconds=0.5)

        record(f'POST /api/realtime/start -> session={bool(session)}', bool(session))
        record(f'推流产出 {produced} 句', produced > 0)
        if stop:
            record(f'POST /api/realtime/stop -> {stop.get("code")}',
                   stop.get('code') == 200)

        if not final:
            record('轮询最终结果', False, '未取到结果')
            return
        record(f'分析状态 status={final.get("status")}',
               final.get('status') == 'done', final.get('message', ''))
        meeting_id = final.get('meeting_id')
        record(f'入库 meeting_id={meeting_id}', bool(meeting_id))

        result = final.get('result') or {}
        transcriptions = result.get('transcriptions') or []
        speakers = {t.get('speaker') for t in transcriptions}
        record(f'分句 {len(transcriptions)} 条，说话人 {sorted(speakers)}',
               len(transcriptions) > 0 and len(speakers) >= 2)

        # 时长换算：16000Hz 应接近音频真实时长
        expected = session['expected'] if session else 0
        # 从 stop 响应取时长
        duration = (stop.get('data') or {}).get('audio_duration') if stop else None
        ok = duration is not None and abs(duration - expected) < 1.0
        record(f'录音时长 {duration}s（实际 {expected:.2f}s）', ok)

        # 时间戳不能超出音频长度
        max_end = max((t.get('end_time') or 0) for t in transcriptions) if transcriptions else 0
        record(f'末句结束 {max_end}s ≤ 音频时长', max_end <= expected + 1.0)

        report = result.get('compliance_report')
        record(f'合规报告存在 score={report.get("total_score") if report else None}',
               bool(report) and 'total_score' in report)
        record('待办事项/决议结论字段存在',
               'action_items' in result and 'decisions' in result)
    finally:
        if meeting_id:
            try:
                client.delete(f'/api/meetings/{meeting_id}')
            except Exception:
                pass


def test_realtime_http_resample():
    print('\n[2] 实时转写 HTTP 全流程（48000Hz，验证重采样与时长换算）')
    from app import app

    wav = os.path.join(AUDIO_DIR, 'gt_2spk.wav')
    if not os.path.exists(wav):
        record('48kHz 场景', False, '缺少 gt_2spk.wav')
        return

    client = app.test_client()
    meeting_id = None
    try:
        session, produced, stop, final = run_session(
            client, wav, 48000, '实时HTTP测试-48k', chunk_seconds=0.5)

        record(f'48kHz 推流产出 {produced} 句', produced > 0)
        expected = session['expected'] if session else 0
        duration = (stop.get('data') or {}).get('audio_duration') if stop else None
        # 关键：若按 16k 解释 48k 数据，时长会缩到约 1/3
        ok = duration is not None and abs(duration - expected) < 1.0
        record(f'报告时长 {duration}s（实际 {expected:.2f}s，若未重采样会是 {expected / 3:.1f}s 左右）', ok)

        if final and final.get('status') == 'done':
            meeting_id = final.get('meeting_id')
            result = final.get('result') or {}
            transcriptions = result.get('transcriptions') or []
            record(f'48kHz 下仍完成分析并入库 {len(transcriptions)} 句',
                   bool(meeting_id) and len(transcriptions) > 0)
        else:
            record('48kHz 下完成分析', False, str(final)[:150] if final else '无结果')
    finally:
        if meeting_id:
            try:
                client.delete(f'/api/meetings/{meeting_id}')
            except Exception:
                pass


def test_realtime_empty_session():
    print('\n[3] 空会话（全程静音）')
    from app import app
    from models import Meeting

    client = app.test_client()
    resp = client.post('/api/realtime/start', json={
        'language': 'zh', 'sample_rate': 16000,
        'meeting_title': '空会话测试', 'enable_compliance': True})
    session = ((resp.get_json() or {}).get('data') or {}).get('session_id')
    if not session:
        record('空会话', False, '未能创建会话')
        return

    with app.app_context():
        before = Meeting.query.count()

    silence = base64.b64encode(np.zeros(16000, dtype=np.int16).tobytes()).decode()
    client.post('/api/realtime/chunk', json={'session_id': session, 'audio': silence})
    client.post('/api/realtime/stop', json={'session_id': session})

    final = poll_result(client, session, timeout=120)
    status = (final or {}).get('status')
    record(f'空会话状态={status}（期望 failed）', status == 'failed')
    record('空会话 meeting_id 为空', not (final or {}).get('meeting_id'))

    with app.app_context():
        after = Meeting.query.count()
    record(f'未产生会议记录（{before} -> {after}）', before == after)

    # discard 不应报错（会话已结束）
    resp = client.post('/api/realtime/discard', json={'session_id': session})
    record(f'POST /api/realtime/discard -> {resp.status_code}', resp.status_code == 200)

    # 未知会话应返回 404
    resp = client.post('/api/realtime/stop', json={'session_id': 'no-such-session'})
    record(f'未知会话 stop -> {resp.status_code}（期望 404）', resp.status_code == 404)
    resp = client.get('/api/realtime/result/no-such-session')
    record(f'未知会话 result -> {resp.status_code}（期望 404）', resp.status_code == 404)


def test_meetings_include_compliance():
    print('\n[4] 会议列表合规摘要（报表页所需）')
    from app import app

    client = app.test_client()
    plain = (client.get('/api/meetings?page_size=5').get_json() or {})
    record('默认列表不含 compliance_summary',
           all('compliance_summary' not in m for m in plain.get('data', [])))

    resp = client.get('/api/meetings?page_size=5&include_compliance=1')
    payload = resp.get_json() or {}
    items = payload.get('data', [])
    record(f'include_compliance=1 返回 {len(items)} 条', resp.status_code == 200 and bool(items))

    has_field = all('compliance_summary' in m for m in items)
    record('每条都带 compliance_summary 字段', has_field)

    with_summary = [m for m in items if m.get('compliance_summary')]
    if with_summary:
        sample = with_summary[0]['compliance_summary']
        ok = all(k in sample for k in
                 ('missing_points_count', 'risk_keywords_count', 'first_suggestion'))
        record(f'摘要字段完整（示例 {json.dumps(sample, ensure_ascii=False)[:80]}）', ok)
    else:
        record('至少一条会议带合规摘要', False, '当前页面内没有已出合规报告的会议')
    record(f'顶层 total={payload.get("total")}', isinstance(payload.get('total'), int))


def main():
    print('=' * 78)
    print('实时转写 HTTP 接口端到端测试')
    print('=' * 78)
    for func in (test_realtime_http_flow, test_realtime_http_resample,
                 test_realtime_empty_session, test_meetings_include_compliance):
        try:
            func()
        except Exception as exc:
            record(func.__name__, False, f'用例异常 {type(exc).__name__}: {exc}')
            import traceback
            traceback.print_exc()

    print('\n' + '=' * 78)
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    for name, ok, detail in RESULTS:
        print(f'  {"✅" if ok else "❌"} {name}' + (f'  ({detail})' if detail else ''))
    print(f'\n通过 {passed}/{total} ({passed / total * 100:.1f}%)')
    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())
