import json
from datetime import datetime


def calculate_compliance_score(transcription_text, knowledge_base_items, score_weights=None, transcription_segments=None):
    """
    计算合规评分
    
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
            'risk_time_markers': [],  # 新增：风险内容时间节点
            'matched_keywords': [],
            'suggestions': ['知识库为空，请先添加合规检查规则']
        }
    
    # 1. 语义相似度匹配（40分）
    total_similarity = 0
    for item in active_items:
        similarity = compute_semantic_similarity(transcription_text, item.content)
        total_similarity += similarity
    
    score_components['semantic_similarity'] = (total_similarity / len(active_items)) * weights['semantic_similarity']
    
    # 2. 必传要点覆盖率（30分）- 使用关键词匹配
    all_required_points = []
    point_keywords_map = {}  # 关键词到要点的映射
    point_sources = {}  # 记录每个要点来自哪个政策
    
    for item in active_items:
        # 只处理 required 类型
        if item.item_type == 'required':
            points = json.loads(item.required_points) if item.required_points else []
            keywords = json.loads(item.keywords) if item.keywords else []
            
            for point in points:
                if point:
                    all_required_points.append(point)
                    point_sources[point] = item.title
            
            # 将关键词映射到要点
            for kw in keywords:
                if kw:
                    point_keywords_map[kw] = {'points': points, 'title': item.title}
    
    covered_points = []
    point_time_markers = []  # 要点出现的时间节点
    
    # 用关键词来检查要点覆盖（更灵活的匹配）
    for kw, mapping in point_keywords_map.items():
        kw_lower = kw.strip().lower()
        if kw_lower in transcription_text.lower():
            # 关键词匹配成功，记录对应的要点
            for point in mapping['points']:
                if point and point not in covered_points:
                    covered_points.append(point)
            
            # 尝试找到关键词出现的时间节点
            if transcription_segments:
                time_marker = find_point_time_marker(kw, transcription_segments)
                if time_marker:
                    point_time_markers.append({
                        'point': mapping['points'][0] if mapping['points'] else kw,  # 显示完整要点描述
                        'keyword': kw,  # 实际匹配的关键词
                        'source': mapping['title'],
                        'start_time': time_marker['start'],
                        'end_time': time_marker['end'],
                        'text': time_marker['text']
                    })
    
    coverage_rate = len(covered_points) / max(len(all_required_points), 1)
    score_components['point_coverage'] = coverage_rate * weights['point_coverage']
    
    # 3. 风险内容检测（20分）- 增加时间节点标记
    all_risk_keywords = []
    risk_categories = {}  # 风险类别
    for item in active_items:
        # 支持 risk_keywords 和 forbidden 类型
        if item.item_type in ['risk_keywords', 'forbidden']:
            keywords = json.loads(item.keywords) if item.keywords else []
            for kw in keywords:
                if kw:
                    all_risk_keywords.append(kw)
                    risk_categories[kw] = item.title
    
    risk_keywords_found = []
    risk_time_markers = []  # 风险内容时间节点
    for kw in all_risk_keywords:
        if kw and kw.strip().lower() in transcription_text.lower():
            risk_keywords_found.append(kw)
            # 找到风险关键词出现的时间节点
            if transcription_segments:
                time_markers = find_risk_time_markers(kw, transcription_segments)
                for marker in time_markers:
                    risk_time_markers.append({
                        'keyword': kw,
                        'category': risk_categories.get(kw, '风险内容'),
                        'start_time': marker['start'],
                        'end_time': marker['end'],
                        'text': marker['text'],
                        'severity': assess_risk_severity(kw)
                    })
    
    risk_score = max(0, weights['risk_detection'] - len(risk_keywords_found) * 5)
    score_components['risk_detection'] = risk_score
    
    # 4. 关键词命中（10分）
    all_keywords = []
    for item in active_items:
        keywords = json.loads(item.keywords) if item.keywords else []
        all_keywords.extend(keywords)
    
    matched_keywords = []
    for kw in all_keywords:
        if kw and kw.strip().lower() in transcription_text.lower():
            matched_keywords.append(kw)
    
    keyword_rate = len(matched_keywords) / max(len(all_keywords), 1)
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


def compute_semantic_similarity(text1, text2):
    """计算语义相似度"""
    try:
        from sentence_transformers import SentenceTransformer, util
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embedding1 = model.encode(text1, convert_to_tensor=True)
        embedding2 = model.encode(text2, convert_to_tensor=True)
        
        cosine_score = util.cos_sim(embedding1, embedding2).item()
        return cosine_score
    
    except Exception as e:
        print(f"Semantic similarity failed: {e}, using simple similarity")
        return simple_similarity(text1, text2)


def simple_similarity(text1, text2):
    """简单相似度（降级方案）"""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
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
        if item.item_type == 'required':
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