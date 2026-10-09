"""
会议数据持久化 + 轻量表结构自愈。

原先保存会议记录的逻辑塞在 modules/realtime_transcriber.py 里（那是已被
FunASR 取代的旧 Whisper 实时模块），既难复用也不好定位问题，这里独立出来。

另外：SQLite 下 `db.create_all()` 只建新表，不会给已有表补新列。为了让老库
（data/meeting_analysis.db）也能直接升级，这里做一次幂等的列检查与补列。
"""
import json
import os

from database import db
from models import ComplianceReport, Meeting, Transcription

# 记录需要保证存在的列：(列名, SQLite 类型)
_MEETING_COLUMNS = [
    ('action_items', 'TEXT'),
    ('decisions', 'TEXT'),
    ('audio_quality', 'TEXT'),
]


def ensure_schema():
    """
    补齐老数据库缺失的列（幂等，重复调用无副作用）。

    create_all 不会修改已存在的表，所以新增字段必须显式 ALTER TABLE。
    """
    try:
        from sqlalchemy import inspect, text

        inspector = inspect(db.engine)
        if 'meeting' not in inspector.get_table_names():
            return
        existing = {column['name'] for column in inspector.get_columns('meeting')}

        for name, column_type in _MEETING_COLUMNS:
            if name in existing:
                continue
            db.session.execute(text(f'ALTER TABLE meeting ADD COLUMN {name} {column_type}'))
            print(f'[数据库] 已补充字段 meeting.{name}')
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        print(f'[数据库] 结构检查失败（不影响运行）: {exc}')


def _as_json(value):
    """把 list/dict/None 安全地序列化为 JSON 文本。"""
    if value is None:
        return None
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps(str(value), ensure_ascii=False)


def save_meeting(
    *,
    title,
    transcriptions,
    summary='',
    keywords=None,
    topics=None,
    sentiment=None,
    action_items=None,
    decisions=None,
    compliance=None,
    audio_quality=None,
    audio_path=None,
    duration=0,
    language='zh',
    status='finished',
):
    """
    保存一场会议（会议主体 + 转写分句 + 合规报告）。

    参数：
        transcriptions: [{'speaker','text','start_time','end_time','confidence'}, ...]
        compliance:    合规结果字典（含 total_score/score_level/components 等）
        audio_quality: 音频质量报告字典（降噪量 / 处理前后信噪比）

    返回新建会议的 id；失败返回 None。
    """
    try:
        meeting = Meeting(
            title=(title or '未命名会议').strip() or '未命名会议',
            duration=int(duration or 0),
            status=status,
            audio_path=audio_path,
            summary=summary or '',
            keywords=_as_json(keywords or []),
            topics=_as_json(topics or []),
            sentiment=_as_json(sentiment or {}),
            action_items=_as_json(action_items or []),
            decisions=_as_json(decisions or []),
            audio_quality=_as_json(audio_quality) if audio_quality else None,
        )
        db.session.add(meeting)
        db.session.commit()

        for segment in transcriptions or []:
            db.session.add(Transcription(
                meeting_id=meeting.id,
                speaker=segment.get('speaker') or 'SPEAKER_00',
                text=segment.get('text', ''),
                start_time=float(segment.get('start_time', segment.get('start', 0)) or 0),
                end_time=float(segment.get('end_time', segment.get('end', 0)) or 0),
                confidence=float(segment.get('confidence', 0) or 0),
                language=language,
            ))
        db.session.commit()

        if compliance:
            report = ComplianceReport(
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
            )
            db.session.add(report)
            meeting.total_score = float(compliance.get('total_score', 0) or 0)
            meeting.score_level = compliance.get('score_level', '')
            db.session.commit()

        return meeting.id

    except Exception as exc:
        db.session.rollback()
        print(f'[数据库] 保存会议失败: {exc}')
        import traceback
        traceback.print_exc()
        return None


def delete_meeting_files(meeting):
    """删除会议关联的音频文件（原始 + 预处理产物），忽略不存在的情况。"""
    if not meeting or not meeting.audio_path:
        return
    candidates = {meeting.audio_path}
    # 预处理产物命名规则：<原名去扩展名>_processed.wav
    base = os.path.splitext(meeting.audio_path)[0]
    candidates.add(base + '_processed.wav')
    if base.endswith('_original'):
        candidates.add(base[: -len('_original')] + '_processed.wav')

    for path in candidates:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError as exc:
                print(f'[清理] 无法删除 {path}: {exc}')
