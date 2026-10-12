"""
会议文档生成测试（会议纪要 / 会议记录概要）。

覆盖：
  1. 两份文件都会生成，目录按日期分月：records/<年-月>/<日期_标题_id>/
  2. 纪要包含正式纪要该有的章节（会议信息、概况、讨论内容、决议、待办、合规、附转写全文）
  3. 概要包含速览表格与摘要、待办/决议清单
  4. 标题里的非法字符被替换，重名会议不会互相覆盖
  5. 文件是 UTF-8 with BOM（Windows 记事本打开中文不乱码）
  6. 接口：GET/POST /api/meetings/<id>/documents、GET /api/documents（按月份分组）
  7. save_meeting / 上传分析链路会自动生成文档

运行：python tests/test_meeting_documents.py
"""
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESULTS = []


def record(name, passed, detail=''):
    RESULTS.append((name, passed, detail))
    print(f'  [{"PASS" if passed else "FAIL"}] {name}' + (f' — {detail}' if detail else ''))
    return passed


class FakeMeeting:
    """模拟数据库里的会议记录对象（文档模块只依赖属性）。"""

    def __init__(self, **kwargs):
        self.id = kwargs.get('id', 1)
        self.title = kwargs.get('title', '合规培训会议')
        self.date = kwargs.get('date', datetime(2026, 10, 12, 15, 30))
        self.created_at = self.date
        self.duration = kwargs.get('duration', 200)
        self.summary = kwargs.get('summary', '会议围绕理财产品合规销售展开，明确了风险揭示要求。')
        self.keywords = json.dumps(kwargs.get('keywords', ['风险等级', '投资限制']), ensure_ascii=False)
        self.topics = json.dumps(kwargs.get('topics', ['合规审查', '培训学习']), ensure_ascii=False)
        self.sentiment = json.dumps(kwargs.get('sentiment', {'sentiment': 'neutral'}), ensure_ascii=False)
        self.action_items = json.dumps(kwargs.get('action_items', ['下周组织全员合规培训']), ensure_ascii=False)
        self.decisions = json.dumps(kwargs.get('decisions', ['会议决定加强销售过程录音录像管理']), ensure_ascii=False)
        self.total_score = kwargs.get('total_score', 62.6)
        self.score_level = kwargs.get('score_level', '合格')
        self.audio_path = None
        self.status = 'finished'
        self.documents_dir = None


SEGMENTS = [
    {'speaker': 'SPEAKER_00', 'text': '各位同事，今天召开产品销售合规会议。', 'start_time': 0.0, 'end_time': 3.2},
    {'speaker': 'SPEAKER_01', 'text': '上周有客户投诉收益表述问题。', 'start_time': 3.5, 'end_time': 7.0},
    {'speaker': 'SPEAKER_00', 'text': '这个问题必须引起重视，重新梳理销售话术。', 'start_time': 7.4, 'end_time': 11.0},
]

COMPLIANCE = {
    'total_score': 62.6,
    'score_level': '合格',
    'components': {'semantic_similarity': 30.0, 'point_coverage': 20.0,
                   'risk_detection': 12.6, 'keyword_matching': 0.0},
    'missing_points': ['投资者风险测评'],
    'risk_keywords': ['保本保收益'],
    'risk_time_markers': [{'keyword': '保本保收益', 'start_time': 5.0, 'end_time': 6.0,
                           'text': '这个产品保本保收益'}],
    'suggestions': ['补充风险等级说明'],
}


def test_basic_generation():
    """两份文件生成、目录结构、章节完整性。"""
    print('\n[1] 基本生成与目录结构')
    from modules import minutes_writer

    root = tempfile.mkdtemp(prefix='records_test_')
    try:
        meeting = FakeMeeting()
        stored = minutes_writer.write_documents(meeting, SEGMENTS, compliance=COMPLIANCE, root=root)

        relative = os.path.relpath(stored['directory'], root)
        expected_dir = os.path.join('2026-10', '2026-10-12_合规培训会议_1')
        record(f'目录按「年-月/日期_标题_id」分层：{relative}', relative == expected_dir,
               f'期望 {expected_dir}')

        minutes = open(stored['minutes'], encoding='utf-8-sig').read()
        summary = open(stored['summary'], encoding='utf-8-sig').read()

        record('会议纪要.md 已生成', os.path.exists(stored['minutes']))
        record('会议记录概要.md 已生成', os.path.exists(stored['summary']))

        required = ['# 合规培训会议 会议纪要', '**会议名称**', '**会议时间**', '**会议时长**',
                    '**参会人员**', '## 一、会议概况', '## 二、主要讨论内容', '## 三、决议结论',
                    '## 四、待办事项', '## 五、合规与风险提示', '## 六、附：转写全文']
        missing = [item for item in required if item not in minutes]
        record('纪要章节齐全', not missing, f'缺少 {missing}' if missing else '11 项齐全')
        record('纪要含说话人标注的讨论内容', '[00:00] SPEAKER_00' in minutes and 'SPEAKER_01' in minutes)
        record('纪要含决议与待办', '加强销售过程录音录像管理' in minutes and '下周组织全员合规培训' in minutes)
        record('纪要含合规评分与遗漏要点', '62.6' in minutes and '投资者风险测评' in minutes)
        record('纪要含风险时间点', '保本保收益' in minutes and '[00:05]' in minutes)

        summary_required = ['# 合规培训会议 · 会议记录概要', '| 项目 | 内容 |', '| 会议时长 |',
                            '| 参会人员 |', '## 一、内容摘要', '## 二、待办事项', '## 三、决议结论']
        missing = [item for item in summary_required if item not in summary]
        record('概要章节齐全', not missing, f'缺少 {missing}' if missing else '7 项齐全')
        record('概要含参会人数与句数', '2 人' in summary and '3 句' in summary)
        record('概要含合规与建议', '62.6' in summary and '补充风险等级说明' in summary)

        with open(stored['minutes'], 'rb') as handle:
            record('文件为 UTF-8 with BOM', handle.read(3) == b'\xef\xbb\xbf')
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_filename_safety():
    """标题含非法字符 / 超长 / 重名时的文件名处理。"""
    print('\n[2] 文件名与重名处理')
    from modules import minutes_writer

    root = tempfile.mkdtemp(prefix='records_test_')
    try:
        tricky = FakeMeeting(id=7, title='2026/10/12 合规培训：A计划<>|*?')
        stored = minutes_writer.write_documents(tricky, SEGMENTS, root=root)
        name = os.path.basename(stored['directory'])
        record(f'非法字符被替换：{name}', not any(ch in name for ch in '\\/:*?"<>|'), name)

        same_title_a = FakeMeeting(id=11, title='同名会议')
        same_title_b = FakeMeeting(id=12, title='同名会议')
        dir_a = minutes_writer.write_documents(same_title_a, SEGMENTS, root=root)['directory']
        dir_b = minutes_writer.write_documents(same_title_b, SEGMENTS, root=root)['directory']
        record('同名会议因带 id 不会互相覆盖', dir_a != dir_b,
               f'{os.path.basename(dir_a)} / {os.path.basename(dir_b)}')

        long_title = FakeMeeting(id=13, title='很长' * 40)
        dir_c = minutes_writer.write_documents(long_title, SEGMENTS, root=root)['directory']
        record('超长标题被截断', len(os.path.basename(dir_c)) <= 60,
               f'{len(os.path.basename(dir_c))} 字符')

        empty = FakeMeeting(id=14, title='   ')
        dir_d = minutes_writer.write_documents(empty, SEGMENTS, root=root)['directory']
        record('空标题有兜底名', '未命名会议' in os.path.basename(dir_d),
               os.path.basename(dir_d))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_describe_and_regenerate():
    """describe_documents 与覆盖写（重新生成）行为。"""
    print('\n[3] 文档状态查询与重新生成')
    from modules import minutes_writer

    root = tempfile.mkdtemp(prefix='records_test_')
    try:
        meeting = FakeMeeting(id=21)
        described = minutes_writer.describe_documents(meeting, root=root)
        record('未生成时 exists=False', all(not item['exists'] for item in described['files']),
               str([item['kind'] for item in described['files']]))

        minutes_writer.write_documents(meeting, SEGMENTS, root=root)
        described = minutes_writer.describe_documents(meeting, root=root)
        record('生成后 exists=True 且带大小',
               all(item['exists'] and item['size'] > 0 for item in described['files']),
               f'{described["files"][0]["size"]} 字节')

        meeting.summary = '更新后的摘要内容：会议达成了合规培训安排。'
        minutes_writer.write_documents(meeting, SEGMENTS, root=root)
        content = open(described['files'][0]['path'], encoding='utf-8-sig').read()
        record('重新生成会覆盖为最新分析结果', '更新后的摘要内容' in content)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_api_endpoints():
    """接口：会议文档查看/下载/重新生成 + 按月份浏览。"""
    print('\n[4] 接口（Flask 测试客户端）')
    from app import app
    from models import Meeting, Transcription

    with app.app_context():
        meeting = (Meeting.query.join(Transcription, Transcription.meeting_id == Meeting.id)
                   .order_by(Meeting.id.desc()).first())
    if not meeting:
        record('接口用例', False, '数据库里没有带转写的会议')
        return
    meeting_id = meeting.id

    client = app.test_client()
    response = client.get(f'/api/meetings/{meeting_id}/documents')
    payload = response.get_json() or {}
    data = payload.get('data') or {}
    record(f'GET /api/meetings/{meeting_id}/documents -> {response.status_code}',
           response.status_code == 200 and len(data.get('files', [])) == 2,
           (data.get('directory') or '')[-40:])
    record('返回两份文件内容', set((data.get('contents') or {}).keys()) == {'minutes', 'summary'},
           str(list((data.get('contents') or {}).keys())))
    record('目录落在 records/ 下', 'records' in (data.get('directory') or ''))

    response = client.get(f'/api/meetings/{meeting_id}/documents/minutes')
    record(f'GET .../documents/minutes -> {response.status_code}',
           response.status_code == 200 and len(response.data) > 200,
           f'{len(response.data)} 字节')

    response = client.post(f'/api/meetings/{meeting_id}/documents')
    record(f'POST .../documents（重新生成）-> {response.status_code}',
           response.status_code == 200)

    response = client.get('/api/documents')
    payload = response.get_json() or {}
    months = ((payload.get('data') or {}).get('months') or [])
    record(f'GET /api/documents -> {response.status_code}，{len(months)} 个月份分组',
           response.status_code == 200 and months and all('month' in m for m in months),
           ', '.join(f'{m["month"]}({len(m["meetings"])})' for m in months[:4]))

    response = client.get(f'/api/meetings/999999/documents')
    record('不存在的会议 -> 404', response.status_code == 404)


def test_save_meeting_writes_documents():
    """save_meeting 会自动生成文档并记录目录。"""
    print('\n[5] save_meeting 自动生成文档')
    from app import app
    from database import db
    from models import Meeting, Transcription
    from modules.meeting_store import delete_meeting_files, save_meeting
    from modules import minutes_writer

    root = tempfile.mkdtemp(prefix='records_test_')
    with app.app_context():
        knowledge_segments = [dict(seg) for seg in SEGMENTS]
        meeting_id = save_meeting(
            title='文档自动生成测试会议',
            transcriptions=knowledge_segments,
            summary='本次会议用于验证文档自动生成。',
            action_items=['完成文档校验'],
            decisions=['通过文档方案'],
            duration=20,
            documents_root=root,
        )
        record(f'save_meeting 入库成功 meeting_id={meeting_id}', bool(meeting_id))
        if not meeting_id:
            shutil.rmtree(root, ignore_errors=True)
            return

        meeting = Meeting.query.get(meeting_id)
        record('meeting.documents_dir 已记录', bool(meeting.documents_dir),
               meeting.documents_dir or '')
        minutes_path = os.path.join(meeting.documents_dir, minutes_writer.MINUTES_NAME)
        record('会议纪要文件已落盘', os.path.exists(minutes_path))

        # 清理测试数据，避免污染演示库
        Transcription.query.filter_by(meeting_id=meeting_id).delete()
        db.session.delete(meeting)
        db.session.commit()
        delete_meeting_files(type('M', (), {'audio_path': None})())
    shutil.rmtree(root, ignore_errors=True)


def main():
    print('=' * 68)
    print('会议文档生成测试')
    print('=' * 68)
    for func in (test_basic_generation, test_filename_safety, test_describe_and_regenerate,
                 test_api_endpoints, test_save_meeting_writes_documents):
        try:
            func()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            record(func.__name__, False, f'异常: {exc}')

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print('\n' + '=' * 68)
    print(f'结果: {passed}/{len(RESULTS)} 通过')
    print('=' * 68)
    return 0 if passed == len(RESULTS) else 1


if __name__ == '__main__':
    sys.exit(main())
