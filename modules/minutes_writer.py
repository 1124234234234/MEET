"""
会议文档生成：把一场会议写成三份文件，按日期分类存放，方便查阅与归档。

产出的文件（需求：「会议记录概要和纪要两个需要生成对应文件」）：

    会议纪要.md      正式纪要，按公文通用格式：
                     标题 + 会议名称/时间/地点/主持人/出席人员/记录人/会议议题
                     + 一、会议概况 + 二、会议听取了下列发言 + 三、会议决定
                     + 四、会议要求 + 五、合规与风险提示 + 落款
                     （引导语用「会议听取了/会议决定/会议要求/会议强调」这套公文写法，
                      不含逐字全文 —— 全文另存为会议记录，不混进纪要）
    会议记录概要.md  速览：一页表格列出时间/时长/参会人/评分/主题/关键词/情绪
                     + 摘要 + 待办与决议清单
    会议记录全文.md  逐字记录：按说话人与时间排列的转写全文（纪要的附件）

存储结构（按日期分月，一眼能找到某天的会）：

    records/
      2026-10/
        2026-10-12_合规培训会议_98/
          会议纪要.md
          会议记录概要.md
          会议记录全文.md

文件名由「日期_标题_会议id」组成：标题里的 Windows 非法字符会被替换，
再带 id 保证重名会议不会互相覆盖。文件用 UTF-8 with BOM 写盘，
Windows 记事本 / Excel 直接打开不会乱码。

本模块只依赖会议数据本身（数据库记录或分析结果字典），不加载任何模型，
因此可以离线批量补生成历史会议的文档（见 export_documents.py）。
"""
import json
import os
import re
import time
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
TRANSCRIPT_NAME = '会议记录全文.md'
# 三类文档（kind → 文件名），接口与前端按这个顺序展示
DOCUMENT_KINDS = (
    ('minutes', MINUTES_NAME),
    ('summary', SUMMARY_NAME),
    ('transcript', TRANSCRIPT_NAME),
)
# 带日期的中文时间戳（公文体里用「2026年10月12日 15时30分」）
_MOMENT_PLACEHOLDER = '（待补充）'
_RECORDER_FALLBACK = '会议室智能合规分析系统（自动转写）'
# 公文分条用中文序号
_CN_NUMBERS = ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十',
               '十一', '十二', '十三', '十四', '十五', '十六', '十七', '十八', '十九', '二十']


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
    lines.append(f'*本文件由会议室智能合规分析系统自动生成于 {datetime.now().strftime("%Y-%m-%d %H:%M")}，'
                 f'与《会议纪要》《会议记录全文》同目录存放。*')
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


def _extract_owner(item):
    """从待办事项里猜负责人（「由X负责」「请X」「X跟进」这类写法）。"""
    patterns = (
        r'由\s*([^\s，。；、]{2,6}?)\s*(?:负责|牵头|跟进|落实|对接|准备|同步)',
        r'请\s*([^\s，。；、]{2,6}?)\s*(?:负责|牵头|跟进|落实|准备|同步)',
        r'([^\s，。；、]{2,6}?)\s*(?:负责|牵头|跟进|落实)',
    )
    for pattern in patterns:
        match = re.search(pattern, item)
        if match:
            candidate = match.group(1).strip()
            if candidate and candidate not in ('相关', '有关', '我们', '大家', '部门'):
                return candidate
    return ''


def _extract_deadline(item):
    """从待办事项里猜时间节点（本周五 / 下周三 / 月底 / 12月15日…）。"""
    patterns = (
        r'(本|下|这)?周[一二三四五六日天]',
        r'(今|明|后)天',
        r'\d{1,2}\s*月\s*\d{1,2}\s*[日号]',
        r'(本|下)?月(底|初|中)',
        r'(两周|一周|三天|五天|十天|半个月|一个半月)(内|之内|以内)?',
        r'[Vv]?\d+(\.\d+)?\s*版本?',
    )
    found = []
    for pattern in patterns:
        for match in re.finditer(pattern, item):
            text = match.group(0).strip()
            if text and text not in found:
                found.append(text)
    return '、'.join(found[:2])


def _action_table(actions):
    """待办事项做成「事项 / 负责人 / 时间」表格（负责人与时间从文字里提，提不到留空）。"""
    rows = [['事项', '负责人', '时间']]
    for item in actions:
        text = str(item).strip().rstrip('。；')
        if not _looks_like_action(text):
            continue
        rows.append([text, _extract_owner(text) or '—', _extract_deadline(text) or '—'])
    return rows if len(rows) > 1 else None


# 待办事项必须带「动作」——否则会把「完成了百分之四十」「请介绍一下技术情况」
# 这类陈述句/提问当成待办，纪要的表格就成了一句句话的堆砌
_ACTION_VERBS = (
    '完成', '提交', '提供', '组织', '准备', '上线', '对接', '同步', '优化', '调整',
    '落实', '跟进', '安排', '培训', '评审', '整改', '汇报', '确认', '补充', '部署',
    '测试', '修复', '更新', '梳理', '协调', '整理', '编写', '制定', '推进', '排查',
)
# 明显不是待办的开头（陈述已完成、或者提问）
_NOT_ACTION_HEADS = (
    '完成', '目前', '大概', '可能', '已经', '因为', '但是', '所以', '如果', '这个',
    '那', '嗯', '啊', '好的', '对', '是',
)
# 待办要带「要做什么」的语气；只有动词没有任务语气的多为状态汇报
_TASK_MARKERS = (
    '需要', '需', '必须', '要求', '请', '计划', '预计', '安排', '负责', '牵头',
    '跟进', '落实', '下周', '本周', '月底', '之前', '以前', '尽快', '后续',
    '下一步', '待', '将于', '前完成', '前提交', '内完成',
)
# 动作词后面接这些词时是名词（「推进会」「测试结果」「优化方案」），不是动作
_NOUN_SUFFIXES = (
    '会', '方案', '结果', '报告', '工作', '情况', '进度', '问题', '版本',
    '模块', '系统', '功能', '计划', '目标', '指标', '环节', '流程',
)


def _has_action_verb(text):
    """找出真正当动词用的动作词，跳过「推进会」「测试结果」这类名词搭配。"""
    for verb in _ACTION_VERBS:
        start = 0
        while True:
            index = text.find(verb, start)
            if index < 0:
                break
            tail = text[index + len(verb):index + len(verb) + 2]
            if not any(tail.startswith(suffix) for suffix in _NOUN_SUFFIXES):
                return True
            start = index + 1
    return False


def _looks_like_action(text):
    """粗筛待办：要有动作词、任务语气、长度合适，且不是「已完成…」「大概…」这类陈述。"""
    body = (text or '').strip()
    if not (8 <= len(body) <= 60):
        return False
    if any(body.startswith(head) for head in _NOT_ACTION_HEADS):
        return False
    if body.endswith(('？', '?')):
        return False
    if not _has_action_verb(body):
        return False
    return any(marker in body for marker in _TASK_MARKERS)


def build_minutes_document(meeting, transcriptions, compliance=None, body=None):
    """
    会议纪要：标题 + 会议信息 + 摘要 + 讨论要点 + 决议 + 待办 + 合规 + 落款。

    版式参考飞书妙记：先给一段摘要，再把「讨论要点」做成短条目，
    决议与待办单独成节，待办用表格（事项 / 负责人 / 时间）。
    会议组织情况（名称/时间/地点/主持人/出席人员/记录人）保留公文要素。

    body 是已经归纳好的要点列表（由 text_analyzer.distill_minutes 生成并缓存）。
    """
    title = getattr(meeting, 'title', '') or '未命名会议'
    created = getattr(meeting, 'date', None) or getattr(meeting, 'created_at', None)
    created_text = (created.strftime('%Y年%m月%d日 %H:%M')
                    if isinstance(created, datetime) else _MOMENT_PLACEHOLDER)
    stamp_date = (created.strftime('%Y年%m月%d日')
                  if isinstance(created, datetime) else datetime.now().strftime('%Y年%m月%d日'))
    duration = getattr(meeting, 'duration', 0) or 0

    actions = [str(a) for a in _as_list(getattr(meeting, 'action_items', None))]
    decisions = [str(d) for d in _as_list(getattr(meeting, 'decisions', None))]
    topics = [str(t.get('topic') if isinstance(t, dict) else t)
              for t in _as_list(getattr(meeting, 'topics', None))]
    keywords = [str(k.get('word') if isinstance(k, dict) else k)
                for k in _as_list(getattr(meeting, 'keywords', None))]

    if body is None:
        body = _as_list(getattr(meeting, 'minutes_body', None))
        if isinstance(body, str):
            body = [line for line in body.split('\n') if line.strip()]

    speakers = {}
    for seg in transcriptions or []:
        speaker = seg.get('speaker') or 'SPEAKER_00'
        info = speakers.setdefault(speaker, {'count': 0, 'chars': 0})
        info['count'] += 1
        info['chars'] += len(seg.get('text') or '')
    total_chars = max(1, sum(v['chars'] for v in speakers.values()))
    speaker_parts = [f'{name}（{info["count"]} 句，'
                     f'{info["chars"] / total_chars * 100:.0f}%）'
                     for name, info in sorted(speakers.items())]

    lines = [f'# {title} · 会议纪要', '']

    # ---- 会议信息（表格，紧凑好读）----
    lines.append('| 项目 | 内容 |')
    lines.append('| --- | --- |')
    lines.append(f'| 会议时间 | {created_text}（{format_duration(duration)}） |')
    lines.append(f'| 会议地点 | {getattr(meeting, "location", None) or "—"} |')
    lines.append(f'| 主 持 人 | {getattr(meeting, "host", None) or "—"} |')
    lines.append(f'| 参会人员 | {len(speakers) or 1} 人：{"；".join(speaker_parts) or "SPEAKER_00"} |')
    lines.append(f'| 记 录 人 | {getattr(meeting, "recorder", None) or _RECORDER_FALLBACK} |')
    if topics:
        lines.append(f'| 会议议题 | {"、".join(topics[:5])} |')
    lines.append('')

    # ---- 一、会议摘要 ----
    lines.append('## 一、会议摘要')
    lines.append((getattr(meeting, 'summary', '') or '（暂无摘要）').strip())
    lines.append('')

    # ---- 二、讨论要点（飞书妙记式短条目）----
    lines.append('## 二、讨论要点')
    if body:
        bullets = []
        for item in body:
            text = str(item).strip()
            # 缓存里可能是旧版带编号/引导语的条目，这里剥掉，避免「1. 1. 会议听取了，…」
            text = re.sub(r'^\d+[.、)]\s*', '', text)
            for lead in ('会议听取了', '会议认为', '会议指出', '会议强调',
                         '会议明确', '会议讨论了'):
                if text.startswith(lead):
                    text = text[len(lead):].lstrip('，,：: ')
                    break
            text = text.strip().strip('。；; ')
            if not text:
                continue
            if not text.endswith(('！', '？', '…')):
                text += '。'
            bullets.append(f'{len(bullets) + 1}. {text}')
        if bullets:
            lines.extend(bullets)
        else:
            lines.append('本次会议没有可用的转写内容。')
    else:
        points = _discussion_points(transcriptions)
        if points:
            for index, item in enumerate(points, 1):
                lines.append(f'{index}. {item["text"]}')
        else:
            lines.append('本次会议没有可用的转写内容。')
    lines.append('')

    # ---- 三、决议事项 ----
    lines.append('## 三、决议事项')
    if decisions:
        for index, item in enumerate(decisions, 1):
            lines.append(f'{index}. {str(item).rstrip("。；")}。')
    else:
        lines.append('本次会议未形成明确的决议事项。')
    lines.append('')

    # ---- 四、待办事项（表格）----
    lines.append('## 四、待办事项')
    table = _action_table(actions) if actions else None
    if table:
        for index, row in enumerate(table):
            lines.append('| ' + ' | '.join(row) + ' |')
            if index == 0:
                lines.append('| --- | --- | --- |')
    else:
        lines.append('本次会议未提出明确的待办事项。')
    lines.append('')

    # ---- 五、合规与风险提示 ----
    lines.append('## 五、合规与风险提示')
    if compliance:
        lines.append(f'合规得分 **{float(compliance.get("total_score", 0)):.1f} 分**'
                     f'{("（" + compliance.get("score_level", "") + "）") if compliance.get("score_level") else ""}。')
        components = compliance.get('components') or compliance.get('detailed_scores') or {}
        if components:
            detail = '、'.join(f'{key} {float(value):.1f}' for key, value in components.items())
            lines.append('')
            lines.append(f'- 分项得分：{detail}')
        risks = compliance.get('risk_keywords') or []
        markers = compliance.get('risk_time_markers') or []
        if risks:
            lines.append('- 风险表述：')
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
            lines.append('- 未发现风险表述。')
        missing = compliance.get('missing_points') or []
        if missing:
            lines.append(f'- 未覆盖的必传要点：{"、".join(str(m) for m in missing[:10])}')
        suggestions = compliance.get('suggestions') or []
        if suggestions:
            lines.append('- 改进建议：')
            lines.extend(f'  - {str(s)}' for s in suggestions[:5])
    else:
        lines.append('本次会议未执行合规比对。')
    lines.append('')

    # ---- 落款 ----
    lines.append('---')
    lines.append('')
    lines.append(f'记录整理：{_RECORDER_FALLBACK}')
    lines.append(f'成文日期：{stamp_date}')
    if keywords:
        lines.append(f'关键词：{"、".join(keywords[:12])}')
    lines.append('')
    lines.append('> 本纪要由会议室智能合规分析系统依据录音自动整理生成，'
                 '议定事项与人员姓名请以会议实际表决结果为准；逐字发言见同目录《会议记录全文》。')
    return '\n'.join(lines) + '\n'


def build_transcript_document(meeting, transcriptions):
    """会议记录全文：按说话人与时间排列的逐字记录（纪要的附件）。"""
    title = getattr(meeting, 'title', '') or '未命名会议'
    created = getattr(meeting, 'date', None) or getattr(meeting, 'created_at', None)
    created_text = (created.strftime('%Y年%m月%d日 %H时%M分')
                    if isinstance(created, datetime) else _MOMENT_PLACEHOLDER)
    duration = getattr(meeting, 'duration', 0) or 0

    speakers = sorted({(seg.get('speaker') or 'SPEAKER_00') for seg in transcriptions or []})
    char_count = sum(len(seg.get('text') or '') for seg in transcriptions or [])

    lines = [f'# {title} 会议记录（全文）', '']
    lines.append(f'会议名称：{title}')
    lines.append(f'时　　间：{created_text}（时长 {format_duration(duration)}）')
    lines.append(f'出席人员：共 {len(speakers) or 1} 人（{"、".join(speakers) or "SPEAKER_00"}）')
    lines.append('记录方式：FunASR paraformer-large 自动转写 + 标点恢复；'
                 '说话人由声纹聚类自动编号')
    lines.append(f'记录规模：{len(transcriptions or [])} 句，共 {char_count} 字')
    lines.append('')
    lines.append('---')
    lines.append('')
    for seg in transcriptions or []:
        text = (seg.get('text') or '').strip()
        if not text:
            continue
        lines.append(f'[{format_clock(seg.get("start_time"))}] '
                     f'{seg.get("speaker") or "SPEAKER_00"}：{text}')
    if not transcriptions:
        lines.append('（无转写内容）')
    lines.append('')
    lines.append('---')
    lines.append(f'*全文由系统自动转写，未经人工校对；纪要见同目录《会议纪要》。*')
    return '\n'.join(lines) + '\n'


# ------------------------------------------------------------------ 写盘


def write_documents(meeting, transcriptions=None, compliance=None, root=None, body=None):
    """
    为一场会议写出三份文件，返回 {directory, minutes, summary, transcript}。

    meeting 可以是数据库模型对象，也可以是带同名属性的普通对象；
    transcriptions 支持 [{'speaker','text','start_time','end_time'}, ...]；
    body 是归纳好的纪要正文条目（不传时用 meeting.minutes_body 缓存）。
    """
    root = root or DEFAULT_ROOT
    created = getattr(meeting, 'date', None) or getattr(meeting, 'created_at', None)
    directory = meeting_directory(root, created, getattr(meeting, 'title', ''), getattr(meeting, 'id', 0))
    os.makedirs(directory, exist_ok=True)

    minutes_path = os.path.join(directory, MINUTES_NAME)
    summary_path = os.path.join(directory, SUMMARY_NAME)
    transcript_path = os.path.join(directory, TRANSCRIPT_NAME)
    _write(minutes_path, build_minutes_document(meeting, transcriptions, compliance, body=body))
    _write(summary_path, build_summary_document(meeting, transcriptions, compliance))
    _write(transcript_path, build_transcript_document(meeting, transcriptions))
    return {'directory': directory, 'minutes': minutes_path, 'summary': summary_path,
            'transcript': transcript_path}


def describe_documents(meeting, root=None):
    """返回该会议各文档的现状（不生成），供接口与前端展示。"""
    root = root or DEFAULT_ROOT
    created = getattr(meeting, 'date', None) or getattr(meeting, 'created_at', None)
    directory = meeting_directory(root, created, getattr(meeting, 'title', ''), getattr(meeting, 'id', 0))
    files = []
    for kind, name in DOCUMENT_KINDS:
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


def schedule_refinement(meeting_id, delay=0.0):
    """
    后台用大模型重新归纳纪要正文，完成后刷新文档文件。

    为什么放后台：1.5B 模型改写要 30~60 秒（本机 CPU），让用户干等不值得 ——
    分析一结束就先用规则归纳写出一份可用纪要（瞬间完成），这里再在后台把措辞
    换成公文体（口语转规范语），改完自动刷新文件。用户不需要做任何操作，
    再次打开纪要时看到的就是改写版。

    Returns:
        True 表示已排入后台线程；False 表示没排（没有转写内容 / 线程创建失败）。
    """
    import threading

    def _worker():
        try:
            time.sleep(delay)
        except Exception:
            pass
        try:
            from app import app as flask_app
            from database import db
            from models import ComplianceReport, Meeting, Transcription
        except Exception as exc:
            print(f'[纪要] 后台改写不可用（无法导入应用上下文）: {exc}')
            return

        try:
            with flask_app.app_context():
                meeting = Meeting.query.get(meeting_id)
                if not meeting:
                    return
                segments = [t.to_dict() for t in
                            Transcription.query.filter_by(meeting_id=meeting_id)
                            .order_by(Transcription.start_time).all()]
                if not segments:
                    return

                from modules.text_analyzer import distill_minutes

                keywords = _as_list(getattr(meeting, 'keywords', None))
                body = distill_minutes(
                    ''.join(segment.get('text') or '' for segment in segments),
                    decisions=[str(item) for item in _as_list(meeting.decisions)],
                    actions=[str(item) for item in _as_list(meeting.action_items)],
                    keywords=keywords,
                    use_model=True,
                )
                if not body:
                    print(f'[纪要] 会议 {meeting_id} 后台改写没有结果，保留规则版')
                    return

                meeting.minutes_body = json.dumps(body, ensure_ascii=False)
                report = ComplianceReport.query.filter_by(meeting_id=meeting_id).first()
                compliance = None
                if report:
                    raw = report.to_dict()
                    compliance = {
                        'total_score': raw.get('total_score', 0),
                        'score_level': raw.get('score_level', ''),
                        'components': raw.get('detailed_scores') or {},
                        'missing_points': raw.get('missing_points') or [],
                        'risk_keywords': raw.get('risk_keywords') or [],
                        'risk_time_markers': raw.get('risk_time_markers') or [],
                        'suggestions': raw.get('suggestions') or [],
                    }
                stored = write_documents(meeting, segments, compliance=compliance, body=body)
                meeting.documents_dir = stored['directory']
                db.session.commit()
                print(f'[纪要] 会议 {meeting_id} 纪要正文已改写并刷新文档')
        except Exception as exc:
            print(f'[纪要] 会议 {meeting_id} 后台改写失败（保留规则版）: {exc}')

    try:
        thread = threading.Thread(target=_worker, daemon=True,
                                 name=f'minutes-refine-{meeting_id}')
        thread.start()
        return True
    except Exception as exc:
        print(f'[纪要] 后台改写线程创建失败: {exc}')
        return False
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
