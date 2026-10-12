"""
批量生成历史会议的文档（会议纪要 / 会议记录概要）。

为什么要单独有这个脚本：文档是在「分析完成」时自动生成的，之前存的会议
（或手工导入的记录）没有文件。这个脚本直接从数据库读转写与合规结果来写文件，
**不加载任何模型、不重新分析**，因此很快（一场会议几毫秒）。

用法：
    python export_documents.py                # 所有有内容的会议
    python export_documents.py 93 94          # 指定会议 id
    python export_documents.py --month 2026-10  # 只导出某个月（按会议时间）

生成位置：records/<年-月>/<日期_标题_id>/
    会议纪要.md
    会议记录概要.md
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)

from app import app                                    # noqa: E402
from database import db                                # noqa: E402
from models import ComplianceReport, Meeting, Transcription  # noqa: E402
from modules import minutes_writer                     # noqa: E402


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


def main():
    args = sys.argv[1:]
    month_filter = None
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
            if not segments and not (meeting.summary or '').strip():
                print(f'  [跳过] 会议 {meeting.id}《{meeting.title}》没有转写内容')
                skipped += 1
                continue
            try:
                stored = minutes_writer.write_documents(
                    meeting, segments, compliance=compliance_payload(meeting.id))
                meeting.documents_dir = stored['directory']
                db.session.commit()
                print(f'  [完成] 会议 {meeting.id}《{meeting.title}》'
                      f'（{len(segments)} 句）-> {os.path.relpath(stored["directory"], BASE_DIR)}')
                done += 1
            except Exception as exc:
                db.session.rollback()
                print(f'  [失败] 会议 {meeting.id}: {exc}')
                skipped += 1

        print(f'\n完成 {done} 场，跳过 {skipped} 场')
        root = minutes_writer.DEFAULT_ROOT
        print(f'文档根目录：{root}')


if __name__ == '__main__':
    sys.exit(main())
