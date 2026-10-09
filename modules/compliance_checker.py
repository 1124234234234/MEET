import json
import os
import re
from datetime import datetime


def _load_list(raw):
    """
    安全解析知识库里的 JSON 数组字段。

    知识库由接口写入，客户端可能传 null、字符串或对象，直接 json.loads 会得到
    None / str / dict，后续迭代就会抛 TypeError 或按字符逐个匹配（把「合规」
    拆成「合」「规」当关键词）。这里统一收敛为字符串列表。
    """
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(item) for item in raw if item is not None and str(item).strip()]
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            value = json.loads(text)
        except (TypeError, ValueError):
            return [text]
        if isinstance(value, list):
            return [str(item) for item in value if item is not None and str(item).strip()]
        if isinstance(value, str):
            return [value] if value.strip() else []
        return []
    return [str(raw)]


_compliance_model = None

def _get_compliance_model():
    """
    获取中文句向量模型（本地优先）。

    与 text_analyzer 共用 model_registry 的单例：旧实现各自加载一份 bge，
    白白多占约 200MB 内存和一次加载时间。
    """
    global _compliance_model
    if _compliance_model is None:
        try:
            from modules import model_registry

            _compliance_model = model_registry.get_sentence_transformer('zh')
        except Exception as e:
            print(f"Failed to load compliance model: {e}")
            return None
    return _compliance_model


# 必传要点模板的适用性门控阈值（转写文本 ↔ 模板的语义相似度）。
# 实测：不对口的组合 0.36~0.44（如「项目例会要点」套到保险销售通话上得 0.4435），
# 对口组合 0.52~0.72（如「理财销售必传要点」对理财文本得 0.7163）。
# 低于阈值就判定「没有适用的模板」，而不是宣称所有要点都遗漏——
# 否则一通正常的销售电话也会被判成「不合格、遗漏进度汇报/问题讨论」。
POINT_TEMPLATE_MIN_RELEVANCE = 0.50


def _template_text(item):
    """把要点模板拼成一段可比对的文本（标题 + 内容 + 各要点）。"""
    parts = [getattr(item, 'title', '') or '', getattr(item, 'content', '') or '']
    parts.extend(_load_list(getattr(item, 'required_points', None)))
    return ' '.join(part for part in parts if part)


def calculate_compliance_score(transcription_text, knowledge_base_items, score_weights=None, transcription_segments=None):
    """
    计算合规评分 - 智能匹配相关知识库，避免被不相关内容拉低分数
    
    参数:
        transcription_text: 转写文本
        knowledge_base_items: 知识库条目列表
        score_weights: 评分权重（可选）
        transcription_segments: 转写段落列表（包含时间戳），用于标记风险内容时间节点
    """
    weights = score_weights or {
        'semantic_similarity': 40,
        'point_coverage': 30,
        'risk_detection': 20,
        'keyword_matching': 10
    }
    
    score_components = {
        'semantic_similarity': 0,
        'point_coverage': 0,
        'risk_detection': 0,
        'keyword_matching': 0
    }
    
    active_items = [item for item in knowledge_base_items if item.status == 'active']
    
    if not active_items:
        return {
            'total_score': 0,
            'components': score_components,
            'covered_points': [],
            'missing_points': [],
            'risk_keywords_found': [],
            'risk_time_markers': [],
            # 与正常返回保持同样的键集合，避免调用方 KeyError
            'point_time_markers': [],
            'matched_keywords': [],
            'suggestions': ['知识库为空，请先添加合规检查规则']
        }
    
    # 1. 计算每个知识库条目与文本的相关性，找出最相关的
    policy_items = [item for item in active_items if item.item_type in ['policy', 'meeting_spirit']]
    point_items = [item for item in active_items if item.item_type in ['required', 'key_points']]
    risk_items = [item for item in active_items if item.item_type in ['risk_keywords', 'forbidden']]
    
    # 计算政策类条目的相似度，找出最相关的
    policy_similarities = []
    for item in policy_items:
        sim = compute_semantic_similarity(transcription_text, item.content)
        policy_similarities.append((item, sim))
    
    # 按相似度排序，取最相关的
    policy_similarities.sort(key=lambda x: x[1], reverse=True)
    best_policy_sim = policy_similarities[0][1] if policy_similarities else 0.0
    
    # 1. 语义相似度得分（40分）- 取最相关政策的相似度，而不是平均
    score_components['semantic_similarity'] = best_policy_sim * weights['semantic_similarity']
    
    # 2. 必传要点覆盖率（30分）- 只考虑与内容相关的要点模板
    #    策略：优先用关键词命中数挑要点模板；一个都没命中时，退回到「与最相关
    #    政策语义最接近」的要点模板。否则一份完全没提要点的会议会因为
    #    best_point_item 为空而报出「零遗漏」，遗漏检测形同虚设。
    candidates = []
    for item in point_items:
        keywords = _load_list(item.keywords)
        match_count = sum(1 for kw in keywords if kw and kw.lower() in transcription_text.lower())
        candidates.append((item, match_count))

    best_point_item = None
    if candidates:
        best_item, best_count = max(candidates, key=lambda pair: pair[1])
        if best_count > 0:
            best_point_item = best_item

    if best_point_item is None and point_items:
        # 没有关键词命中：用语义相似度找该政策对应的要点模板
        reference = policy_similarities[0][0] if policy_similarities else None
        best_score = 0.0
        for item in point_items:
            if reference is not None:
                text = f'{reference.title} {reference.content}'
                score = compute_semantic_similarity(text, f'{item.title} {item.content}')
            else:
                score = simple_similarity(transcription_text, f'{item.title} {item.content}')
            if score > best_score:
                best_score = score
                best_point_item = item

        # 只有一个要点模板时直接采用，避免误配到完全无关的模板
        if len(point_items) == 1:
            best_point_item = point_items[0]

    # 适用性门控：模板与本次会议内容不相关时，不套用它。
    # 否则一通与模板无关的会议会被判成「所有必传要点都遗漏」，
    # 用户看到的是「不合格 + 遗漏进度汇报/问题讨论」这类不相关结论。
    template_rejected = False
    template_relevance = None
    if best_point_item is not None:
        template_relevance = compute_semantic_similarity(
            transcription_text, _template_text(best_point_item))
        if template_relevance < POINT_TEMPLATE_MIN_RELEVANCE:
            print(f'  [合规] 要点模板「{best_point_item.title}」与本次内容相关性过低'
                  f'（{template_relevance:.2f} < {POINT_TEMPLATE_MIN_RELEVANCE}），不套用')
            template_rejected = True
            best_point_item = None
    
    covered_points = []
    point_time_markers = []
    all_required_points = []

    if best_point_item:
        points = _load_list(best_point_item.required_points)
        keywords = _load_list(best_point_item.keywords)
        all_required_points = [p for p in points if p]

        # 逐个要点判定是否被覆盖。
        # 旧实现把「任一关键词命中」映射成「该条目的全部要点都被覆盖」，
        # 结果是一句话提到「通知」就让所有必传要点都算完成、missing_points
        # 永远为空——遗漏事项检测这个核心功能等于失效。
        # 现在按要点自身文本（或与要点同名的关键词）逐个匹配。
        for point in all_required_points:
            point_text = str(point).strip()
            if not point_text:
                continue

            matched_keyword = None
            if point_text.lower() in transcription_text.lower():
                matched_keyword = point_text
            else:
                for kw in keywords:
                    if kw == point_text and kw.lower() in transcription_text.lower():
                        matched_keyword = kw
                        break

            if not matched_keyword:
                continue

            covered_points.append(point)
            if transcription_segments:
                time_marker = find_point_time_marker(matched_keyword, transcription_segments)
                if time_marker:
                    point_time_markers.append({
                        'point': point_text,
                        'keyword': matched_keyword,
                        'source': best_point_item.title,
                        'start_time': time_marker['start'],
                        'end_time': time_marker['end'],
                        'text': time_marker['text'],
                    })
    
    if all_required_points:
        coverage_rate = len(covered_points) / len(all_required_points)
        score_components['point_coverage'] = coverage_rate * weights['point_coverage']
    else:
        # 没有配置必传要点时，给一个基础分（按关键词匹配度）
        if policy_similarities and best_policy_sim > 0:
            score_components['point_coverage'] = best_policy_sim * weights['point_coverage'] * 0.8
    
    # 3. 风险内容检测（20分）- 增加时间节点标记和语义层面风险检测
    # 改进：区分禁止语境和实际使用，避免将合规培训内容误判为风险
    all_risk_keywords = []
    risk_categories = {}
    for item in active_items:
        if item.item_type in ['risk_keywords', 'forbidden']:
            keywords = _load_list(item.keywords)
            for kw in keywords:
                if kw:
                    all_risk_keywords.append(kw)
                    risk_categories[kw] = item.title
    
    risk_keywords_found = []
    risk_time_markers = []
    for kw in all_risk_keywords:
        if kw and kw.strip().lower() in transcription_text.lower():
            if is_negated_context(transcription_text, kw):
                continue
            risk_keywords_found.append(kw)
            if transcription_segments:
                time_markers = find_risk_time_markers(kw, transcription_segments)
                for marker in time_markers:
                    if not is_negated_context(marker['text'], kw):
                        risk_time_markers.append({
                            'keyword': kw,
                            'category': risk_categories.get(kw, '风险内容'),
                            'start_time': marker['start'],
                            'end_time': marker['end'],
                            'text': marker['text'],
                            'severity': assess_risk_severity(kw)
                        })
    
    semantic_risks = detect_semantic_risks(transcription_text, transcription_segments)
    risk_time_markers.extend(semantic_risks)
    
    risk_score = max(0, weights['risk_detection'] - len(risk_keywords_found) * 5 - len(semantic_risks) * 3)
    score_components['risk_detection'] = risk_score
    
    # 4. 关键词命中（10分）- 只统计最相关政策的关键词
    relevant_keywords = []
    if policy_similarities and best_policy_sim > 0.1:
        best_policy = policy_similarities[0][0]
        keywords = _load_list(best_policy.keywords)
        relevant_keywords = [kw for kw in keywords if kw]
    
    if not relevant_keywords:
        # 如果没有相关政策，取所有关键词
        for item in active_items:
            keywords = _load_list(item.keywords)
            relevant_keywords.extend(keywords)
    
    matched_keywords = []
    for kw in relevant_keywords:
        if kw and kw.strip().lower() in transcription_text.lower():
            matched_keywords.append(kw)
    
    keyword_rate = len(matched_keywords) / max(len(relevant_keywords), 1)
    score_components['keyword_matching'] = keyword_rate * weights['keyword_matching']
    
    # 总分
    total_score = sum(score_components.values())
    
    # 生成建议
    suggestions = generate_suggestions(total_score, covered_points, all_required_points, risk_keywords_found)
    if template_rejected:
        # 明确告知「没有适用的模板」，并给出可操作的方向，而不是让用户以为真的漏了一堆要点
        suggestions.insert(0, (
            f'本次会议内容与知识库中现有的必传要点模板相关性较低'
            f'（{template_relevance:.2f}），未套用任何模板进行遗漏核查。'
            '建议在知识库中按会议类型补充对应的必传要点模板。'
        ))

    return {
        'total_score': round(total_score, 2),
        'components': score_components,
        'covered_points': covered_points,
        'point_time_markers': point_time_markers,
        'missing_points': [p for p in all_required_points if p not in covered_points],
        'risk_keywords_found': risk_keywords_found,
        'risk_time_markers': risk_time_markers,
        'matched_keywords': matched_keywords,
        'suggestions': suggestions,
        'point_template': None if best_point_item is None else best_point_item.title,
        'point_template_rejected': template_rejected,
        'point_template_relevance': None if template_relevance is None else round(template_relevance, 4),
    }


def find_point_time_marker(point, segments):
    """找到必传要点出现的时间节点"""
    point_lower = point.strip().lower()
    for seg in segments:
        if point_lower in seg.get('text', '').lower():
            return {
                'start': seg.get('start_time', 0),
                'end': seg.get('end_time', 0),
                'text': seg.get('text', '')
            }
    return None


def find_risk_time_markers(keyword, segments):
    """找到风险关键词出现的所有时间节点"""
    markers = []
    keyword_lower = keyword.strip().lower()
    for seg in segments:
        text = seg.get('text', '')
        if keyword_lower in text.lower():
            markers.append({
                'start': seg.get('start_time', 0),
                'end': seg.get('end_time', 0),
                'text': text
            })
    return markers


def assess_risk_severity(keyword):
    """评估风险严重程度"""
    high_risk = ['抵制', '反对', '拒绝', '消极', '不满']
    medium_risk = ['抱怨', '不行', '不可能', '做不到']
    
    keyword_lower = keyword.strip().lower()
    if any(kw in keyword_lower for kw in high_risk):
        return 'high'
    elif any(kw in keyword_lower for kw in medium_risk):
        return 'medium'
    else:
        return 'low'


def is_negated_context(text, keyword, window=15):
    """
    检查关键词是否在否定/禁止的语境中。
    关键词前面出现禁止/不得/不能/不要/严禁/不允许等否定词时，说明是在强调
    「不要这么做」，属于合规表述而不是风险内容。
    """
    negation_words = [
        '禁止', '不得', '不能', '不要', '严禁', '不允许', '不可以', '不准',
        '反对', '拒绝', '纠正', '不对', '错误', '不应该', '不应当',
        '必须避免', '不能有', '不允许有', '禁止使用', '禁止承诺',
        '不允许承诺', '不得承诺', '不得使用', '不可', '千万别',
        '绝对不能', '一定不要', '坚决禁止', '严格禁止',
        # 英文语境同样需要识别，否则 "Do Not Use LIBOR" 会被判成风险
        'do not', "don't", 'must not', 'should not', 'never', 'avoid',
        'prohibited', 'forbidden', 'no ',
    ]

    # 命中判定统一做了小写归一化，这里也必须同样处理：否则英文/混合大小写
    # 关键词（如 "LIBOR"）在禁止语境中 find 返回 -1，会被当成真实风险上报
    lowered = text.lower()
    keyword_pos = lowered.find(str(keyword).strip().lower())
    if keyword_pos == -1:
        return False

    prefix = lowered[max(0, keyword_pos - window * 2):keyword_pos]

    for neg in negation_words:
        if neg in prefix:
            return True

    negation_patterns = [
        r'.{0,10}(禁止|不得|不能|不要|严禁|不允许|不准|不可以).{0,5}$',
        r'.{0,10}(反对|拒绝|纠正|不对|错误).{0,5}$',
        r'.{0,15}\b(do not|must not|should not|never)\b.{0,10}$',
    ]
    for pattern in negation_patterns:
        if re.search(pattern, prefix):
            return True

    return False


def _scan_risk_patterns(text, segments, patterns):
    """
    扫描风险句式，返回命中的风险条目。

    旧实现的写法是「先全文 finditer 判断是否出现，再对每个句子重新 finditer」，
    同一个命中会被重复累加（全文命中 N 次 × 句内命中 M 次，实测同一句话重复
    三次会产生 9 条相同风险），导致 risk_detection 分数被重复扣到 0。
    这里改为每条命中只计一次：有分句时按分句定位，没有分句时用全文定位。
    """
    risks = []
    seen = set()

    for pattern, category, severity in patterns:
        if segments:
            for seg in segments:
                seg_text = seg.get('text', '') or ''
                for match in re.finditer(pattern, seg_text):
                    matched = match.group()
                    if is_negated_context(seg_text, matched):
                        continue
                    key = (matched, category, seg.get('start_time', 0))
                    if key in seen:
                        continue
                    seen.add(key)
                    risks.append({
                        'keyword': matched,
                        'category': category,
                        'start_time': seg.get('start_time', 0),
                        'end_time': seg.get('end_time', 0),
                        'text': seg_text,
                        'severity': severity,
                    })
        else:
            for match in re.finditer(pattern, text):
                matched = match.group()
                if is_negated_context(text, matched):
                    continue
                key = (matched, category)
                if key in seen:
                    continue
                seen.add(key)
                risks.append({
                    'keyword': matched,
                    'category': category,
                    'start_time': 0,
                    'end_time': 0,
                    'text': text[:50],
                    'severity': severity,
                })

    return risks


def detect_semantic_risks(text, segments=None):
    """
    语义层面的风险检测
    检测：偏离主题、表述不当、消极负面等内容
    区分禁止语境和实际使用，避免将合规培训内容误判为风险
    """
    inappropriate_patterns = [
        (r'肯定会赚|一定赚|稳赚|保本保收益|零风险|无风险|绝对安全', '表述不当', 'high'),
        (r'忽悠|骗|蒙|糊弄|坑', '表述不当', 'high'),
        (r'随便说说|无所谓|不用管', '表述不当', 'medium'),
        (r'不行吧|不太可能|估计悬', '消极负面', 'medium'),
        (r'太麻烦了|不想做|懒得弄', '消极负面', 'medium'),
        (r'这东西没用|没必要|浪费时间', '消极负面', 'high'),
        (r'客户傻|客户不懂|客户好忽悠', '表述不当', 'high'),
        (r'上级不知道|领导不会查|没人知道', '表述不当', 'high'),
    ]

    negative_sentiment_patterns = [
        (r'问题太多|麻烦不断|一团糟|混乱', '消极负面', 'high'),
        (r'没办法|无解|搞不定|束手无策', '消极负面', 'medium'),
        (r'不乐观|堪忧|危险|隐患', '消极负面', 'medium'),
        (r'反对意见|抵制|拒绝执行', '消极负面', 'high'),
        (r'抱怨|不满|牢骚|指责', '消极负面', 'medium'),
    ]

    return (_scan_risk_patterns(text, segments, inappropriate_patterns)
            + _scan_risk_patterns(text, segments, negative_sentiment_patterns))


def compute_semantic_similarity(text1, text2):
    """计算语义相似度"""
    try:
        text1 = (text1 or '').strip()
        text2 = (text2 or '').strip()
        
        if not text1 or not text2:
            return 0.0
        
        from sentence_transformers import util
        
        model = _get_compliance_model()
        if model is None:
            return simple_similarity(text1, text2)
        
        embedding1 = model.encode(text1, convert_to_tensor=True)
        embedding2 = model.encode(text2, convert_to_tensor=True)
        
        cosine_score = util.cos_sim(embedding1, embedding2).item()
        return cosine_score
    
    except Exception as e:
        print(f"Semantic similarity failed: {e}, using simple similarity")
        return simple_similarity(text1, text2)


def simple_similarity(text1, text2):
    """简单相似度（降级方案）- 使用jieba分词支持中文"""
    import jieba
    
    words1 = set(w for w in jieba.lcut(text1.lower()) if len(w) > 1)
    words2 = set(w for w in jieba.lcut(text2.lower()) if len(w) > 1)
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union


def get_score_level(score):
    """获取评分等级"""
    if score >= 90:
        return '优秀'
    elif score >= 75:
        return '良好'
    elif score >= 60:
        return '合格'
    else:
        return '不合格'


def generate_suggestions(total_score, covered_points, all_required_points, risk_keywords_found):
    """生成建议"""
    suggestions = []
    
    if total_score < 60:
        suggestions.append('会议内容与政策要求差距较大，建议重新传达')
    
    missing_points = [p for p in all_required_points if p not in covered_points]
    if missing_points:
        suggestions.append(f'以下必传要点未覆盖：{", ".join(missing_points[:5])}{"..." if len(missing_points) > 5 else ""}')
    
    if risk_keywords_found:
        suggestions.append(f'发现风险内容：{", ".join(risk_keywords_found[:3])}{"..." if len(risk_keywords_found) > 3 else ""}，请核查')
    
    if not suggestions:
        suggestions.append('会议内容合规，继续保持')
    
    return suggestions


def realtime_compliance_check(text_segment, knowledge_base_items, start_time, end_time):
    """
    实时合规检查：对一段转写文本做风险与必传要点核查。

    返回结构（与前端实时提示、第三方客户端约定一致）：
        {
          'has_risk': bool,
          'risk_keywords': ['保本保收益', ...],           # 便于直接展示
          'risk_items': [{'keyword','severity','start_time','end_time','text'}, ...],
          'covered_points': [{'point','keyword','start_time','end_time'}, ...],
          'alerts': [{'type','level','message','time'}, ...],
          'start_time': float, 'end_time': float           # 本段文本的时间范围
        }
    """
    active_items = [item for item in knowledge_base_items if item.status == 'active']

    result = {
        'has_risk': False,
        'risk_keywords': [],
        'risk_items': [],
        'covered_points': [],
        'alerts': [],
        'start_time': start_time,
        'end_time': end_time,
    }

    if not active_items:
        return result

    text_lower = text_segment.lower()
    seen_risk = set()
    seen_points = set()

    # 1. 风险关键词
    for item in active_items:
        if item.item_type in ['risk_keywords', 'forbidden']:
            for kw in _load_list(item.keywords):
                if not kw or kw.strip().lower() not in text_lower:
                    continue
                if is_negated_context(text_segment, kw):
                    continue
                if kw in seen_risk:
                    continue
                seen_risk.add(kw)
                result['has_risk'] = True
                result['risk_keywords'].append(kw)
                result['risk_items'].append({
                    'keyword': kw,
                    'start_time': start_time,
                    'end_time': end_time,
                    'text': text_segment,
                    'severity': assess_risk_severity(kw),
                })

    # 2. 必传要点：逐条判定，避免「命中一个关键词就把该条目的全部要点都算覆盖」
    #    （旧实现会把所有要点一次性标记为已覆盖，实时提示因此失去意义）
    for item in active_items:
        if item.item_type in ['required', 'key_points']:
            keywords = _load_list(item.keywords)
            points = _load_list(item.required_points)
            for point in points:
                if not point:
                    continue
                point_text = str(point).strip()
                matched_keyword = None
                if point_text.lower() in text_lower:
                    matched_keyword = point_text
                else:
                    for kw in keywords:
                        if kw == point_text and kw.strip().lower() in text_lower:
                            matched_keyword = kw
                            break
                if not matched_keyword or point in seen_points:
                    continue
                seen_points.add(point)
                result['covered_points'].append({
                    'point': point,
                    'keyword': matched_keyword,
                    'start_time': start_time,
                    'end_time': end_time,
                })

    # 3. 高风险告警
    high_risks = [r for r in result['risk_items'] if r['severity'] == 'high']
    if high_risks:
        result['alerts'].append({
            'type': 'risk_alert',
            'level': 'high',
            'message': f'检测到高风险内容：{", ".join(r["keyword"] for r in high_risks)}',
            'time': start_time,
        })

    return result