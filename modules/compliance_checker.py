import json
import os
from datetime import datetime

_compliance_model = None

def _get_compliance_model():
    global _compliance_model
    if _compliance_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            
            # 优先使用本地中文向量模型 bge-small-zh-v1.5
            local_model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'models', 'bge-small-zh-v1.5'
            )
            
            if os.path.exists(local_model_path):
                _compliance_model = SentenceTransformer(local_model_path)
                print(f'Loaded bge-small-zh-v1.5 from local: {local_model_path}')
            else:
                # 降级：尝试从HuggingFace下载
                os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
                _compliance_model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
                print('Loaded bge-small-zh-v1.5 from HuggingFace')
        except Exception as e:
            print(f"Failed to load compliance model: {e}")
            return None
    return _compliance_model


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
    #    策略：找出关键词匹配最多的key_points条目，用它来计算覆盖率
    best_point_item = None
    best_point_match_count = 0
    
    for item in point_items:
        keywords = json.loads(item.keywords) if item.keywords else []
        match_count = sum(1 for kw in keywords if kw and kw.lower() in transcription_text.lower())
        if match_count > best_point_match_count:
            best_point_match_count = match_count
            best_point_item = item
    
    # 另外：也根据政策相似度来找对应的要点（如果政策和要点标题相似）
    if policy_similarities and policy_similarities[0][1] > 0.1:
        best_policy_title = policy_similarities[0][0].title
        # 找标题最匹配的要点条目
        for item in point_items:
            title_sim = simple_similarity(best_policy_title, item.title)
            if title_sim > 0.3:
                best_point_item = item
                break
    
    covered_points = []
    point_time_markers = []
    all_required_points = []
    
    if best_point_item:
        points = json.loads(best_point_item.required_points) if best_point_item.required_points else []
        keywords = json.loads(best_point_item.keywords) if best_point_item.keywords else []
        all_required_points = [p for p in points if p]
        
        point_keywords_map = {}
        for kw in keywords:
            if kw:
                point_keywords_map[kw] = points
        
        for kw, kw_points in point_keywords_map.items():
            kw_lower = kw.strip().lower()
            if kw_lower in transcription_text.lower():
                for point in kw_points:
                    if point and point not in covered_points:
                        covered_points.append(point)
                
                if transcription_segments:
                    time_marker = find_point_time_marker(kw, transcription_segments)
                    if time_marker:
                        point_time_markers.append({
                            'point': kw_points[0] if kw_points else kw,
                            'keyword': kw,
                            'source': best_point_item.title,
                            'start_time': time_marker['start'],
                            'end_time': time_marker['end'],
                            'text': time_marker['text']
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
            keywords = json.loads(item.keywords) if item.keywords else []
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
        keywords = json.loads(best_policy.keywords) if best_policy.keywords else []
        relevant_keywords = [kw for kw in keywords if kw]
    
    if not relevant_keywords:
        # 如果没有相关政策，取所有关键词
        for item in active_items:
            keywords = json.loads(item.keywords) if item.keywords else []
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
    
    return {
        'total_score': round(total_score, 2),
        'components': score_components,
        'covered_points': covered_points,
        'point_time_markers': point_time_markers,
        'missing_points': [p for p in all_required_points if p not in covered_points],
        'risk_keywords_found': risk_keywords_found,
        'risk_time_markers': risk_time_markers,
        'matched_keywords': matched_keywords,
        'suggestions': suggestions
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
    检查关键词是否在否定/禁止的语境中
    如果关键词前面有禁止/不得/不能/不要/严禁/不允许等否定词，则不是风险
    """
    import re
    
    negation_words = [
        '禁止', '不得', '不能', '不要', '严禁', '不允许', '不可以', '不准',
        '反对', '拒绝', '纠正', '不对', '错误', '不应该', '不应当',
        '必须避免', '不能有', '不允许有', '禁止使用', '禁止承诺',
        '不允许承诺', '不得承诺', '不得使用', '不可', '千万别',
        '绝对不能', '一定不要', '坚决禁止', '严格禁止',
    ]
    
    keyword_pos = text.find(keyword)
    if keyword_pos == -1:
        return False
    
    prefix = text[max(0, keyword_pos - window * 2):keyword_pos]
    
    for neg in negation_words:
        if neg in prefix:
            return True
    
    negation_patterns = [
        r'.{0,10}(禁止|不得|不能|不要|严禁|不允许|不准|不可以).{0,5}$',
        r'.{0,10}(反对|拒绝|纠正|不对|错误).{0,5}$',
    ]
    for pattern in negation_patterns:
        if re.search(pattern, prefix):
            return True
    
    return False


def detect_semantic_risks(text, segments=None):
    """
    语义层面的风险检测
    检测：偏离主题、表述不当、消极负面等内容
    改进：区分禁止语境和实际使用，避免将合规培训内容误判为风险
    """
    risks = []
    
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
    
    for pattern, category, severity in inappropriate_patterns:
        import re
        matches = re.finditer(pattern, text)
        for match in matches:
            matched_text = match.group()
            if is_negated_context(text, matched_text):
                continue
            
            if segments:
                for seg in segments:
                    seg_text = seg.get('text', '')
                    seg_matches = list(re.finditer(pattern, seg_text))
                    for seg_match in seg_matches:
                        seg_matched = seg_match.group()
                        if not is_negated_context(seg_text, seg_matched):
                            risks.append({
                                'keyword': seg_matched,
                                'category': category,
                                'start_time': seg.get('start_time', 0),
                                'end_time': seg.get('end_time', 0),
                                'text': seg_text,
                                'severity': severity
                            })
            else:
                risks.append({
                    'keyword': matched_text,
                    'category': category,
                    'start_time': 0,
                    'end_time': 0,
                    'text': text[:50],
                    'severity': severity
                })
    
    negative_sentiment_patterns = [
        (r'问题太多|麻烦不断|一团糟|混乱', '消极负面', 'high'),
        (r'没办法|无解|搞不定|束手无策', '消极负面', 'medium'),
        (r'不乐观|堪忧|危险|隐患', '消极负面', 'medium'),
        (r'反对意见|抵制|拒绝执行', '消极负面', 'high'),
        (r'抱怨|不满|牢骚|指责', '消极负面', 'medium'),
    ]
    
    for pattern, category, severity in negative_sentiment_patterns:
        import re
        matches = re.finditer(pattern, text)
        for match in matches:
            matched_text = match.group()
            if is_negated_context(text, matched_text):
                continue
            
            if segments:
                for seg in segments:
                    seg_text = seg.get('text', '')
                    seg_matches = list(re.finditer(pattern, seg_text))
                    for seg_match in seg_matches:
                        seg_matched = seg_match.group()
                        if not is_negated_context(seg_text, seg_matched):
                            risks.append({
                                'keyword': seg_matched,
                                'category': category,
                                'start_time': seg.get('start_time', 0),
                                'end_time': seg.get('end_time', 0),
                                'text': seg_text,
                                'severity': severity
                            })
            else:
                risks.append({
                    'keyword': matched_text,
                    'category': category,
                    'start_time': 0,
                    'end_time': 0,
                    'text': text[:50],
                    'severity': severity
                })
    
    return risks


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


def generate_compliance_report(meeting_id, transcription_text, compliance_result):
    """生成合规报告"""
    report = {
        'meeting_id': meeting_id,
        'report_time': datetime.now().isoformat(),
        'overall_score': compliance_result['total_score'],
        'score_level': get_score_level(compliance_result['total_score']),
        'detailed_scores': compliance_result['components'],
        'summary': {
            'covered_points_count': len(compliance_result['covered_points']),
            'total_points_count': len(compliance_result['covered_points']) + len(compliance_result['missing_points']),
            'risk_items_count': len(compliance_result['risk_keywords_found']),
            'matched_keywords_count': len(compliance_result['matched_keywords'])
        },
        'covered_points': compliance_result['covered_points'],
        'point_time_markers': compliance_result.get('point_time_markers', []),
        'missing_points': compliance_result['missing_points'],
        'risk_keywords': compliance_result['risk_keywords_found'],
        'risk_time_markers': compliance_result.get('risk_time_markers', []),
        'matched_keywords': compliance_result['matched_keywords'],
        'suggestions': compliance_result['suggestions']
    }
    
    return report


def realtime_compliance_check(text_segment, knowledge_base_items, start_time, end_time):
    """
    实时合规检查（用于WebSocket实时转写）
    每收到一段转写文本就进行合规检查
    
    返回：
        - 是否有风险内容
        - 是否覆盖了必传要点
        - 需要告警的内容
    """
    active_items = [item for item in knowledge_base_items if item.status == 'active']
    
    result = {
        'has_risk': False,
        'risk_items': [],
        'covered_points': [],
        'alerts': []
    }
    
    if not active_items:
        return result
    
    text_lower = text_segment.lower()
    
    # 检查风险关键词
    for item in active_items:
        # 支持 risk_keywords 和 forbidden 类型
        if item.item_type in ['risk_keywords', 'forbidden']:
            keywords = json.loads(item.keywords) if item.keywords else []
            for kw in keywords:
                if kw and kw.strip().lower() in text_lower:
                    if is_negated_context(text_segment, kw):
                        continue
                    result['has_risk'] = True
                    result['risk_items'].append({
                        'keyword': kw,
                        'start_time': start_time,
                        'end_time': end_time,
                        'text': text_segment,
                        'severity': assess_risk_severity(kw)
                    })
    
    # 检查必传要点 - 使用关键词匹配
    for item in active_items:
        # 支持 required 和 key_points 两种类型
        if item.item_type in ['required', 'key_points']:
            keywords = json.loads(item.keywords) if item.keywords else []
            points = json.loads(item.required_points) if item.required_points else []
            
            for kw in keywords:
                if kw and kw.strip().lower() in text_lower:
                    # 关键词匹配成功，记录对应的要点
                    for point in points:
                        if point:
                            result['covered_points'].append({
                                'point': point,
                                'keyword': kw,
                                'start_time': start_time,
                                'end_time': end_time
                            })
    
    # 生成实时告警
    if result['risk_items']:
        high_risks = [r for r in result['risk_items'] if r['severity'] == 'high']
        if high_risks:
            result['alerts'].append({
                'type': 'risk_alert',
                'level': 'high',
                'message': f'检测到高风险内容：{", ".join([r["keyword"] for r in high_risks])}',
                'time': start_time
            })
    
    return result