"""
会议文档生成：把一场会议写成两份文件，按日期分类存放，方便查阅。

产出的两份文件（需求：「会议记录概要和纪要两个需要生成对应文件」）：

    会议纪要.md      正式纪要：会议基本信息 + 会议概况 + 讨论要点 +
                     决议结论 + 待办事项 + 合规与风险提示 + 转写全文（附）
    会议记录概要.md  速览：一页表格列出时间/时长/参会人/评分/主题/关键词/
                     情绪 + 摘要 + 待办与决议清单

存储结构（按日期分月，一眼能找到某天的会）：

    records/
      2026-10/
        2026-10-12_合规培训会议_98/
          会议纪要.md
          会议记录概要.md

文件名由「日期_标题_会议id」组成：标题里的 Windows 非法字符会被替换，
再带 id 保证重名会议不会互相覆盖。文件用 UTF-8 with BOM 写盘，
Windows 记事本 / Excel 直接打开不会乱码。

本模块只依赖会议数据本身（数据库记录或分析结果字典），不加载任何模型，
因此可以离线批量补生成历史会议的文档（见 export_documents.py）。
"""
import json
import os
import re
from datetime import datetime

# 默认根目录：项目下的 records/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ROOT = os.path.join(BASE_DIR, 'records')

# 文件名里不能出现的字符（Windows 限制）+ 控制字符
_ILLEGAL = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
# 标题最多取多少个字符进文件名（太长在资源管理器里不好看）
MAX_TITLE_IN_NAME = 30

MINUTES_NAME = '会议纪要.md'
SUMMARY_NAME = '会议记录概要.md'


def _as_list(value):
    """把 JSON 字符串 / 列表 / None 统一成列表。"""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError):
            return [text]
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
        return [str(parsed)]
    return [value]


def _as_dict(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except (ValueError, TypeError):
            return {}
    return {}


def sanitize_name(name, max_length=MAX_TITLE_IN_NAME):
    """把标题变成安全的文件名片段。"""
    text = _ILLEGAL.sub('_', str(name or '').strip())
    text = re.sub(r'\s+', ' ', text).strip(' ._')
    if not text:
        text = '未命名会议'
    return text[:max_length]


def format_duration(seconds):
    """把秒数写成「X 分 Y 秒」。"""
    try:
        total = int(round(float(seconds or 0)))
    except (TypeError, ValueError):
        total = 0
    minutes, secs = divmod(total, 60)
    if minutes >= 60:
        hours, minutes = divmod(minutes, 60)
        return f'{hours} 小时 {minutes} 分 {secs} 秒'
    if minutes:
        return f'{minutes} 分 {secs} 秒'
    return f'{secs} 秒'


def format_clock(seconds):
    """秒 → mm:ss（转写全文里的时间戳）。"""
    try:
        total = int(max(0.0, float(seconds or 0)))
    except (TypeError, ValueError):
        total = 0
    return f'{total // 60:02d}:{total % 60:02d}'


def meeting_directory(root, created_at, title, meeting_id):
    """会议文档目录：records/<年-月>/<日期_标题_id>/。"""
    moment = created_at if isinstance(created_at, datetime) else datetime.now()
    month = moment.strftime('%Y-%m')
    day = moment.strftime('%Y-%m-%d')
    folder = f'{day}_{sanitize_name(title)}_{meeting_id}'
    return os.path.join(root, month, folder)


def _write(path, content):
    # utf-8-sig：带 BOM，Windows 记事本/dos 工具打开中文不乱码
    with open(path, 'w', encoding='utf-8-sig') as handle:
        handle.write(content)
    return path


# ------------------------------------------------------------------ 概要


def build_summary_document(meeting, transcriptions, compliance=None):
    """会议记录概要：一页速览。"""
    title = getattr(meeting, 'title', '') or '未命名会议'
    created = getattr(meeting, 'date', None) or getattr(meeting, 'created_at', None)
    created_text = created.strftime('%Y-%m-%d %H:%M') if isinstance(created, datetime) else ''
    duration = getattr(meeting, 'duration', 0) or 0

    keywords = [str(k.get('word') if isinstance(k, dict) else k)
                for k in _as_list(getattr(meeting, 'keywords', None))]
    topics = [str(t.get('topic') if isinstance(t, dict) else t)
              for t in _as_list(getattr(meeting, 'topics', None))]
    actions = [str(a) for a in _as_list(getattr(meeting, 'action_items', None))]
    decisions = [str(d) for d in _as_list(getattr(meeting, 'decisions', None))]
    sentiment = _as_dict(getattr(meeting, 'sentiment', None))

    speakers = {}
    for seg in transcriptions or []:
        speaker = seg.get('speaker') or 'SPEAKER_00'
        speakers[speaker] = speakers.get(speaker, 0) + 1
    speaker_text = '、'.join(f'{name}（{count} 句）' for name, count in sorted(speakers.items()))
    char_count = sum(len(seg.get('text') or '') for seg in transcriptions or [])

    score = getattr(meeting, 'total_score', None)
    level = getattr(meeting, 'score_level', '') or ''

    lines = [f'# {title} · 会议记录概要', '']
    lines.append('| 项目 | 内容 |')
    lines.append('| --- | --- |')
    lines.append(f'| 会议名称 | {title} |')
    if created_text:
        lines.append(f'| 会议时间 | {created_text} |')
    lines.append(f'| 会议时长 | {format_duration(duration)} |')
    lines.append(f'| 参会人员 | {len(speakers) or 1} 人（{speaker_text or "SPEAKER_00"}） |')
    lines.append(f'| 发言规模 | {len(transcriptions or [])} 句，共 {char_count} 字 |')
    if score is not None:
        lines.append(f'| 合规评分 | {float(score):.1f} 分{("（" + level + "）") if level else ""} |')
    if topics:
        lines.append(f'| 会议主题 | {"、".join(topics[:5])} |')
    if keywords:
        lines.append(f'| 关键词 | {"、".join(keywords[:12])} |')
    if sentiment:
        label = sentiment.get('label') or sentiment.get('sentiment') or ''
        if label:
            lines.append(f'| 情绪倾向 | {label} |')
    lines.append(f'| 待办事项 | {len(actions)} 条 |')
    lines.append(f'| 决议结论 | {len(decisions)} 条 |')
    lines.append('')

    lines.append('## 一、内容摘要')
    lines.append((getattr(meeting, 'summary', '') or '（暂无摘要）').strip())
    lines.append('')

    lines.append('## 二、待办事项')
    if actions:
        lines.extend(f'{index}. {item}' for index, item in enumerate(actions, 1))
    else:
        lines.append('（本次会议未提取到明确的待办事项）')
    lines.append('')

    lines.append('## 三、决议结论')
    if decisions:
        lines.extend(f'{index}. {item}' for index, item in enumerate(decisions, 1))
    else:
        lines.append('（本次会议未提取到明确的决议结论）')
    lines.append('')

    lines.append('## 四、合规与风险')
    if compliance:
        lines.append(f'- 合规评分：{float(compliance.get("total_score", 0)):.1f} 分'
                     f'{("（" + compliance.get("score_level", "") + "）") if compliance.get("score_level") else ""}')
        risks = compliance.get('risk_keywords') or []
        missing = compliance.get('missing_points') or []
        suggestions = compliance.get('suggestions') or []
        if risks:
            lines.append(f'- 风险内容：{"、".join(str(r) for r in risks[:10])}')
        else:
            lines.append('- 风险内容：未发现风险表述')
        if missing:
            lines.append(f'- 遗漏要点：{"、".join(str(m) for m in missing[:10])}')
        if suggestions:
            lines.append('- 改进建议：')
            lines.extend(f'  - {str(s)}' for s in suggestions[:5])
    else:
        lines.append('（本次会议未执行合规比对）')

    lines.append('')
    lines.append('---')
    lines.append(f'*本文件由会议室智能合规分析系统自动生成于 {datetime.now().strftime("%Y-%m-%d %H:%M")}*')
    return '\n'.join(lines) + '\n'


# ------------------------------------------------------------------ 纪要


def _discussion_points(transcriptions, compliance=None, max_points=12):
    """按说话人把发言整理成「谁说了什么」的要点，供纪要的讨论内容章节使用。"""
    points = []
    for seg in transcriptions or []:
        text = (seg.get('text') or '').strip()
        if not text:
            continue
        points.append({
            'time': format_clock(seg.get('start_time')),
            'speaker': seg.get('speaker') or 'SPEAKER_00',
            'text': text,
            'start': float(seg.get('start_time') or 0),
        })
    return points


def build_minutes_document(meeting, transcriptions, compliance=None):
    """会议纪要：正式文书（结构参考公开的会议纪要模板）。"""
    title = getattr(meeting, 'title', '') or '未命名会议'
    created = getattr(meeting, 'date', None) or getattr(meeting, 'created_at', None)
    created_text = created.strftime('%Y年%m月%d日 %H:%M') if isinstance(created, datetime) else ''
    duration = getattr(meeting, 'duration', 0) or 0

    actions = [str(a) for a in _as_list(getattr(meeting, 'action_items', None))]
    decisions = [str(d) for d in _as_list(getattr(meeting, 'decisions', None))]
    topics = [str(t.get('topic') if isinstance(t, dict) else t)
              for t in _as_list(getattr(meeting, 'topics', None))]
    keywords = [str(k.get('word') if isinstance(k, dict) else k)
                for k in _as_list(getattr(meeting, 'keywords', None))]

    speakers = {}
    for seg in transcriptions or []:
        speaker = seg.get('speaker') or 'SPEAKER_00'
        info = speakers.setdefault(speaker, {'count': 0, 'chars': 0})
        info['count'] += 1
        info['chars'] += len(seg.get('text') or '')
    speaker_parts = []
    for name, info in sorted(speakers.items()):
        share = (info['chars'] / max(1, sum(v['chars'] for v in speakers.values()))) * 100
        speaker_parts.append(f'{name}（发言 {info["count"]} 句，占 {share:.0f}%）')

    lines = [f'# {title} 会议纪要', '']
    lines.append(f'- **会议名称**：{title}')
    if created_text:
        lines.append(f'- **会议时间**：{created_text}')
    lines.append(f'- **会议时长**：{format_duration(duration)}')
    lines.append(f'- **参会人员**：共 {len(speakers) or 1} 人'
                 + (f'（{"；".join(speaker_parts)}）' if speaker_parts else ''))
    if topics:
        lines.append(f'- **会议主题**：{"、".join(topics[:5])}')
    lines.append('- **记录方式**：会议室智能合规分析系统自动转写与整理'
                 '（说话人按声纹聚类编号，人员姓名需人工核对）')
    lines.append('')

    lines.append('## 一、会议概况')
    lines.append((getattr(meeting, 'summary', '') or '（暂无摘要）').strip())
    lines.append('')

    lines.append('## 二、主要讨论内容')
    points = _discussion_points(transcriptions)
    if points:
        for item in points:
            lines.append(f'- **[{item["time"]}] {item["speaker"]}**：{item["text"]}')
    else:
        lines.append('（本次会议没有可用的转写内容）')
    lines.append('')

    lines.append('## 三、决议结论')
    if decisions:
        lines.extend(f'{index}. {item}' for index, item in enumerate(decisions, 1))
    else:
        lines.append('（本次会议未形成明确的决议结论）')
    lines.append('')

    lines.append('## 四、待办事项')
    if actions:
        lines.extend(f'{index}. {item}' for index, item in enumerate(actions, 1))
    else:
        lines.append('（本次会议未提取到明确的待办事项）')
    lines.append('')

    lines.append('## 五、合规与风险提示')
    if compliance:
        lines.append(f'- 合规评分：{float(compliance.get("total_score", 0)):.1f} 分'
                     f'{("（" + compliance.get("score_level", "") + "）") if compliance.get("score_level") else ""}')
        components = compliance.get('components') or compliance.get('detailed_scores') or {}
        if components:
            detail = '、'.join(f'{key} {float(value):.1f}' for key, value in components.items())
            lines.append(f'- 分项得分：{detail}')
        risks = compliance.get('risk_keywords') or []
        markers = compliance.get('risk_time_markers') or []
        if risks:
            lines.append('- 风险内容：')
            if markers:
                for marker in markers[:10]:
                    if isinstance(marker, dict):
                        lines.append(f'  - [{format_clock(marker.get("start_time"))}] '
                                     f'{marker.get("keyword", "")}：{marker.get("text", "")}')
                    else:
                        lines.append(f'  - {marker}')
            else:
                lines.append(f'  - {"、".join(str(r) for r in risks[:10])}')
        else:
            lines.append('- 风险内容：未发现风险表述')
        missing = compliance.get('missing_points') or []
        if missing:
            lines.append(f'- 遗漏要点：{"、".join(str(m) for m in missing[:10])}')
        suggestions = compliance.get('suggestions') or []
        if suggestions:
            lines.append('- 改进建议：')
            lines.extend(f'  - {str(s)}' for s in suggestions[:5])
    else:
        lines.append('（本次会议未执行合规比对）')
    lines.append('')

    lines.append('## 六、附：转写全文')
    if points:
        for item in points:
            lines.append(f'[{item["time"]}] {item["speaker"]}：{item["text"]}')
    else:
        lines.append('（无）')
    lines.append('')
    if keywords:
        lines.append(f'> 关键词：{"、".join(keywords[:15])}')
        lines.append('')

    lines.append('---')
    lines.append(f'*本纪要由会议室智能合规分析系统自动生成于 {datetime.now().strftime("%Y-%m-%d %H:%M")}，'
                 f'说话人编号（SPEAKER_00…）为系统聚类结果，请人工核对具体人员与结论。*')
    return '\n'.join(lines) + '\n'


# ------------------------------------------------------------------ 写盘


def write_documents(meeting, transcriptions=None, compliance=None, root=None):
    """
    为一场会议写出两份文件，返回 {directory, minutes, summary}。

    meeting 可以是数据库模型对象，也可以是带同名属性的普通对象；
    transcriptions 支持 [{'speaker','text','start_time','end_time'}, ...]。
    """
    root = root or DEFAULT_ROOT
    created = getattr(meeting, 'date', None) or getattr(meeting, 'created_at', None)
    directory = meeting_directory(root, created, getattr(meeting, 'title', ''), getattr(meeting, 'id', 0))
    os.makedirs(directory, exist_ok=True)

    minutes_path = os.path.join(directory, MINUTES_NAME)
    summary_path = os.path.join(directory, SUMMARY_NAME)
    _write(minutes_path, build_minutes_document(meeting, transcriptions, compliance))
    _write(summary_path, build_summary_document(meeting, transcriptions, compliance))
    return {'directory': directory, 'minutes': minutes_path, 'summary': summary_path}


def describe_documents(meeting, root=None):
    """返回该会议两份文档的现状（不生成），供接口展示。"""
    root = root or DEFAULT_ROOT
    created = getattr(meeting, 'date', None) or getattr(meeting, 'created_at', None)
    directory = meeting_directory(root, created, getattr(meeting, 'title', ''), getattr(meeting, 'id', 0))
    files = []
    for kind, name in (('minutes', MINUTES_NAME), ('summary', SUMMARY_NAME)):
        path = os.path.join(directory, name)
        exists = os.path.exists(path)
        files.append({
            'kind': kind,
            'name': name,
            'path': path,
            'exists': exists,
            'size': os.path.getsize(path) if exists else 0,
        })
    return {'directory': directory, 'files': files}
