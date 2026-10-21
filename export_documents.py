"""
批量生成历史会议的文档（会议纪要 / 会议记录概要 / 会议记录全文）。

为什么要单独有这个脚本：文档是在「分析完成」时自动生成的，之前存的会议
（或手工导入的记录）没有文件。这个脚本直接从数据库读转写与合规结果来写文件，
**不加载任何模型、不重新分析**，因此很快（一场会议几毫秒）。

用法：
    python export_documents.py                # 所有有内容的会议（纪要正文走规则归纳）
    python export_documents.py 93 94          # 指定会议 id
    python export_documents.py --month 2026-10  # 只导出某个月（按会议时间）
    python export_documents.py --distill        # 顺带用大模型重新归纳纪要正文（慢，每场 15~30s）

生成位置：records/<年-月>/<日期_标题_id>/
    会议纪要.md
    会议记录概要.md
    会议记录全文.md
"""
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)

from app import app                                    # noqa: E402
from database import db                                # noqa: E402
from models import ComplianceReport, Meeting, Transcription  # noqa: E402
from modules import meeting_store, minutes_writer      # noqa: E402


def compliance_payload(meeting_id):
    report = ComplianceReport.query.filter_by(meeting_id=meeting_id).first()
    if not report:
        return None
    raw = report.to_dict()
    return {
        'total_score': raw.get('total_score', 0),
        'score_level': raw.get('score_level', ''),
        'components': raw.get('detailed_scores') or {},
        'missing_points': raw.get('missing_points') or [],
        'risk_keywords': raw.get('risk_keywords') or [],
        'risk_time_markers': raw.get('risk_time_markers') or [],
        'suggestions': raw.get('suggestions') or [],
    }


def minutes_body_for(meeting, segments, distill=False):
    """
    纪要正文：优先用缓存；--distill 时用大模型重新归纳（慢，每场 15~60s）。

    默认**不跑大模型**：批量补历史文档要的是快（这里是 88 场，跑模型要半小时），
    规则归纳已经能产出「带公文引导语的分条纪要」。新会议在分析结束时会由
    后台线程自动跑一次模型改写（见 minutes_writer.schedule_refinement），
    单场想再来一遍就从页面点「重新生成」。
    """
    body = meeting_store._minutes_body_of(meeting)
    if body and not distill:
        return body
    try:
        from modules.text_analyzer import distill_minutes

        body = distill_minutes(
            ''.join(segment.get('text') or '' for segment in segments),
            decisions=json.loads(meeting.decisions) if meeting.decisions else [],
            actions=json.loads(meeting.action_items) if meeting.action_items else [],
            keywords=json.loads(meeting.keywords) if meeting.keywords else [],
            use_model=bool(distill),
        )
        if body:
            meeting.minutes_body = json.dumps(body, ensure_ascii=False)
    except Exception as exc:
        print(f'    [警告] 纪要正文归纳失败: {exc}')
        return body
    return body


def main():
    args = sys.argv[1:]
    month_filter = None
    distill = '--distill' in args
    ids = []
    index = 0
    while index < len(args):
        if args[index] == '--month' and index + 1 < len(args):
            month_filter = args[index + 1]
            index += 2
            continue
        if args[index].isdigit():
            ids.append(int(args[index]))
        index += 1

    with app.app_context():
        query = Meeting.query
        if ids:
            query = query.filter(Meeting.id.in_(ids))
        meetings = query.order_by(Meeting.date).all()

        done = skipped = 0
        for meeting in meetings:
            if month_filter:
                moment = meeting.date or meeting.created_at
                if not moment or moment.strftime('%Y-%m') != month_filter:
                    continue
            segments = [t.to_dict() for t in
                        Transcription.query.filter_by(meeting_id=meeting.id)
                        .order_by(Transcription.start_time).all()]
            try:
                body = minutes_body_for(meeting, segments, distill=distill)
                stored = minutes_writer.write_documents(
                    meeting, segments, compliance=compliance_payload(meeting.id),
                    body=body)
                meeting.documents_dir = stored['directory']
                db.session.commit()
                print(f'  [完成] 会议 {meeting.id}《{meeting.title}》'
                      f'（{len(segments)} 句）-> {os.path.relpath(stored["directory"], BASE_DIR)}')
                done += 1
            except Exception as exc:
                db.session.rollback()
                print(f'  [失败] 会议 {meeting.id}: {exc}')
                skipped += 1

        print(f'\n完成 {done} 场，跳过 {skipped} 场'
              f'{("（纪要正文已用大模型重新归纳）" if distill else "")}')
        print(f'文档根目录：{minutes_writer.DEFAULT_ROOT}')


if __name__ == '__main__':
    sys.exit(main())

