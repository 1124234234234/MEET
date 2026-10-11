"""
重新分析历史会议（用当前识别与分析链路刷新已有记录）。

用途：识别/分段链路升级后（例如修复实时转写「不准」之后），历史会议里存的
还是旧结果，界面上看着仍然是碎片文本。本脚本用**同一份音频**重跑一遍完整
分析（预处理 → 识别 → 说话人分离 → 文本分析 → 合规比对）并就地更新该会议
记录，不需要重新上传音频，也不会新建会议。

用法：
    python reanalyze_meetings.py 93 94          # 指定会议 id
    python reanalyze_meetings.py --realtime      # 所有实时转写会议
    python reanalyze_meetings.py --all           # 所有音频还在的会议

注意：
  - 会覆盖这些会议现有的转写、摘要、合规报告（音频文件不动）；
  - 每个会议都要跑完整分析，几秒钟一个；模型只需加载一次；
  - 运行前建议先停掉正在运行的服务（两边各占一套模型内存）。
"""
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)

from app import app                                    # noqa: E402
from database import db                                # noqa: E402
from models import ComplianceReport, KnowledgeBase, Meeting, ScoreWeight, Transcription  # noqa: E402
from modules.analysis_pipeline import analyze_audio     # noqa: E402
from modules.meeting_store import _as_json             # noqa: E402


def _score_weights():
    rows = ScoreWeight.query.all()
    if rows:
        return {row.weight_name: row.weight_value for row in rows}
    return app.config.get('SCORE_WEIGHTS')


def _update_meeting(meeting, analysis, duration):
    """把分析结果写回已有会议记录（先清掉旧的转写与合规报告）。"""
    Transcription.query.filter_by(meeting_id=meeting.id).delete()
    ComplianceReport.query.filter_by(meeting_id=meeting.id).delete()

    for segment in analysis.get('transcriptions') or []:
        db.session.add(Transcription(
            meeting_id=meeting.id,
            speaker=segment.get('speaker') or 'SPEAKER_00',
            text=segment.get('text', ''),
            start_time=float(segment.get('start_time', 0) or 0),
            end_time=float(segment.get('end_time', 0) or 0),
            confidence=float(segment.get('confidence', 0) or 0),
            language=analysis.get('language', 'zh'),
        ))

    meeting.summary = analysis.get('summary', '') or ''
    meeting.keywords = _as_json(analysis.get('keywords') or [])
    meeting.topics = _as_json(analysis.get('topics') or [])
    meeting.sentiment = _as_json(analysis.get('sentiment') or {})
    meeting.action_items = _as_json(analysis.get('action_items') or [])
    meeting.decisions = _as_json(analysis.get('decisions') or [])
    meeting.audio_quality = _as_json(analysis.get('audio_quality')) if analysis.get('audio_quality') else None
    if duration:
        meeting.duration = int(duration)

    compliance = analysis.get('compliance_report')
    if compliance:
        db.session.add(ComplianceReport(
            meeting_id=meeting.id,
            total_score=float(compliance.get('total_score', 0) or 0),
            score_level=compliance.get('score_level', ''),
            detailed_scores=_as_json(compliance.get('components', {})),
            missing_points=_as_json(compliance.get('missing_points', [])),
            risk_keywords=_as_json(compliance.get('risk_keywords_found', [])),
            risk_time_markers=_as_json(compliance.get('risk_time_markers', [])),
            point_time_markers=_as_json(compliance.get('point_time_markers', [])),
            matched_keywords=_as_json(compliance.get('matched_keywords', [])),
            suggestions=_as_json(compliance.get('suggestions', [])),
        ))
        meeting.total_score = float(compliance.get('total_score', 0) or 0)
        meeting.score_level = compliance.get('score_level', '')
    db.session.commit()


def _resolve_targets(args):
    if not args or args[0] == '--all':
        query = Meeting.query
    elif args[0] == '--realtime':
        query = Meeting.query.filter(Meeting.audio_path.like('%_realtime.wav'))
    else:
        ids = []
        for value in args:
            try:
                ids.append(int(value))
            except ValueError:
                print(f'忽略无法识别的参数: {value}')
        query = Meeting.query.filter(Meeting.id.in_(ids))
    return query.order_by(Meeting.id).all()


def main():
    args = sys.argv[1:]
    with app.app_context():
        targets = _resolve_targets(args)
        targets = [m for m in targets if m.audio_path and os.path.exists(m.audio_path)]
        if not targets:
            print('没有可重新分析的会议（会议不存在或音频已删除）')
            return 1

        knowledge_items = KnowledgeBase.query.filter_by(status='active').all()
        weights = _score_weights()
        print(f'待重新分析 {len(targets)} 场会议，知识库条目 {len(knowledge_items)} 条\n')

        ok = failed = 0
        for meeting in targets:
            before = Transcription.query.filter_by(meeting_id=meeting.id).count()
            old_text = ' '.join(r.text or '' for r in
                                Transcription.query.filter_by(meeting_id=meeting.id)
                                .order_by(Transcription.start_time).limit(3).all())
            print(f'--- 会议 {meeting.id}《{meeting.title}》{meeting.duration}s，'
                  f'音频 {os.path.basename(meeting.audio_path)}')
            print(f'    更新前 {before} 句: {old_text[:70]}')

            started = time.time()
            analysis = analyze_audio(
                meeting.audio_path,
                language='zh',
                knowledge_items=knowledge_items,
                score_weights=weights,
                enable_diarization=True,
            )
            if not analysis:
                print('    跳过：没识别到有效语音（音频可能已损坏或全程静音）\n')
                failed += 1
                continue

            segments = analysis.get('transcriptions') or []
            _update_meeting(meeting, analysis, analysis.get('duration'))
            after_text = ' '.join(s.get('text', '') for s in segments[:3])
            print(f'    更新后 {len(segments)} 句（{time.time() - started:.1f}s）: {after_text[:70]}')
            print(f'    摘要: {(analysis.get("summary") or "")[:60]}\n')
            ok += 1

        print(f'完成：成功 {ok}，跳过 {failed}')
        return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
