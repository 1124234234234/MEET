"""
报表生成模块
生成会议统计报表、合规分析报表
支持导出为PDF和Excel格式
"""
from datetime import datetime
from io import BytesIO
import json


def generate_meeting_summary_report(meeting_data):
    """
    生成会议摘要报告
    """
    report = {
        'report_id': meeting_data.get('meeting_id', 'unknown'),
        'report_time': datetime.now().isoformat(),
        'title': meeting_data.get('title', '未命名会议'),
        'date': meeting_data.get('date', datetime.now().isoformat()),
        'duration': meeting_data.get('duration', 0),
        'participant_count': meeting_data.get('participant_count', 0),
        'summary': meeting_data.get('summary', ''),
        'topics': meeting_data.get('topics', []),
        'keywords': meeting_data.get('keywords', []),
        'sentiment': meeting_data.get('sentiment', {}),
        'compliance': meeting_data.get('compliance', {}),
        'transcription_count': len(meeting_data.get('transcriptions', []))
    }
    return report


def generate_compliance_trend_report(meetings_data):
    """
    生成合规趋势报表
    """
    if not meetings_data:
        return None
    
    total_meetings = len(meetings_data)
    total_score = sum(m.get('total_score', 0) for m in meetings_data)
    avg_score = total_score / total_meetings
    
    score_distribution = {'excellent': 0, 'good': 0, 'pass': 0, 'fail': 0}
    for m in meetings_data:
        level = m.get('score_level', 'fail')
        if level == '优秀':
            score_distribution['excellent'] += 1
        elif level == '良好':
            score_distribution['good'] += 1
        elif level == '合格':
            score_distribution['pass'] += 1
        else:
            score_distribution['fail'] += 1
    
    risk_count = 0
    missing_points_count = 0
    for m in meetings_data:
        compliance = m.get('compliance_report')
        if compliance:
            risk_keywords = json.loads(compliance.risk_keywords) if compliance.risk_keywords else []
            missing_points = json.loads(compliance.missing_points) if compliance.missing_points else []
            risk_count += len(risk_keywords)
            missing_points_count += len(missing_points)
    
    report = {
        'report_time': datetime.now().isoformat(),
        'time_range': {
            'start': min(m.get('date', datetime.now().isoformat()) for m in meetings_data),
            'end': max(m.get('date', datetime.now().isoformat()) for m in meetings_data)
        },
        'statistics': {
            'total_meetings': total_meetings,
            'avg_score': round(avg_score, 2),
            'avg_duration': round(sum(m.get('duration', 0) for m in meetings_data) / total_meetings, 0),
            'total_risk_items': risk_count,
            'total_missing_points': missing_points_count,
            'score_distribution': score_distribution
        },
        'trend_data': [
            {
                'date': m.get('date', ''),
                'score': m.get('total_score', 0),
                'level': m.get('score_level', '')
            } for m in meetings_data
        ],
        'recommendations': generate_recommendations(avg_score, risk_count, missing_points_count)
    }
    
    return report


def generate_recommendations(avg_score, risk_count, missing_points_count):
    """
    根据统计数据生成改进建议
    """
    recommendations = []
    
    if avg_score < 60:
        recommendations.append('整体合规评分偏低，建议加强合规培训')
    elif avg_score < 80:
        recommendations.append('合规评分有提升空间，建议优化传达流程')
    
    if risk_count > 5:
        recommendations.append('风险内容较多，建议加强风险预警监控')
    
    if missing_points_count > 10:
        recommendations.append('必传要点覆盖不足，建议完善知识库内容')
    
    if not recommendations:
        recommendations.append('整体合规情况良好，继续保持')
    
    return recommendations


def generate_report_html(report_type, data):
    """
    生成HTML格式报表
    """
    if report_type == 'meeting_summary':
        return _generate_meeting_summary_html(data)
    elif report_type == 'compliance_trend':
        return _generate_compliance_trend_html(data)
    return '<html><body><h1>未知报表类型</h1></body></html>'


def _generate_meeting_summary_html(data):
    """生成会议摘要HTML报表"""
    html = f"""
    <html>
    <head>
        <title>会议摘要报告 - {data['title']}</title>
        <style>
            body {{ font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 40px; }}
            .header {{ border-bottom: 2px solid #667eea; padding-bottom: 20px; margin-bottom: 30px; }}
            .header h1 {{ color: #667eea; }}
            .info-row {{ display: flex; margin-bottom: 12px; }}
            .info-label {{ width: 120px; font-weight: bold; color: #666; }}
            .info-value {{ color: #333; }}
            .section {{ margin-bottom: 30px; }}
            .section-title {{ font-size: 18px; font-weight: bold; color: #667eea; margin-bottom: 15px; padding-bottom: 5px; border-bottom: 1px solid #eee; }}
            .keyword {{ display: inline-block; background: #e8f0fe; padding: 4px 12px; border-radius: 20px; margin: 4px; font-size: 13px; }}
            .topic {{ margin-bottom: 10px; }}
            .topic-name {{ font-weight: bold; }}
            .topic-bar {{ width: 200px; height: 8px; background: #eee; border-radius: 4px; margin-top: 4px; }}
            .topic-fill {{ height: 100%; background: #667eea; border-radius: 4px; }}
            .score-box {{ display: inline-block; padding: 20px; border-radius: 12px; }}
            .score-excellent {{ background: #f6ffed; border: 1px solid #b7eb8f; }}
            .score-good {{ background: #e6f7ff; border: 1px solid #91d5ff; }}
            .score-pass {{ background: #fffbe6; border: 1px solid #ffe58f; }}
            .score-fail {{ background: #fff2f0; border: 1px solid #ffccc7; }}
            .score-value {{ font-size: 36px; font-weight: bold; }}
            .score-excellent .score-value {{ color: #52c41a; }}
            .score-good .score-value {{ color: #1890ff; }}
            .score-pass .score-value {{ color: #faad14; }}
            .score-fail .score-value {{ color: #ff4d4f; }}
            .risk-item {{ background: #fff1f0; padding: 8px; margin: 4px 0; border-radius: 4px; }}
            .missing-item {{ background: #fffbe6; padding: 8px; margin: 4px 0; border-radius: 4px; }}
            .suggestion {{ background: #f6ffed; padding: 8px; margin: 4px 0; border-radius: 4px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>会议摘要报告</h1>
            <div class="info-row"><span class="info-label">会议标题:</span><span class="info-value">{data['title']}</span></div>
            <div class="info-row"><span class="info-label">会议时间:</span><span class="info-value">{data['date']}</span></div>
            <div class="info-row"><span class="info-label">会议时长:</span><span class="info-value">{data['duration']} 分钟</span></div>
            <div class="info-row"><span class="info-label">参会人数:</span><span class="info-value">{data['participant_count']} 人</span></div>
        </div>
        
        <div class="section">
            <div class="section-title">会议摘要</div>
            <p>{data['summary']}</p>
        </div>
        
        <div class="section">
            <div class="section-title">讨论主题</div>
            {''.join([f'<div class="topic"><span class="topic-name">{t["topic"]}</span><div class="topic-bar"><div class="topic-fill" style="width:{t["score"]*100}%"></div></div></div>' for t in data.get('topics', [])])}
        </div>
        
        <div class="section">
            <div class="section-title">关键词</div>
            {''.join([f'<span class="keyword">{k["word"]} ({k["frequency"]})</span>' for k in data.get('keywords', [])])}
        </div>
        
        <div class="section">
            <div class="section-title">合规评分</div>
            {_get_score_html(data.get('compliance', {}))}
        </div>
    </body>
    </html>
    """
    return html


def _get_score_html(compliance):
    """生成评分HTML"""
    if not compliance:
        return '<p>未进行合规检查</p>'
    
    level_map = {'优秀': 'excellent', '良好': 'good', '合格': 'pass', '不合格': 'fail'}
    level = compliance.get('score_level', '不合格')
    class_name = level_map.get(level, 'fail')
    
    html = f"""
    <div class="score-box score-{class_name}">
        <div class="score-value">{compliance.get('total_score', 0)}</div>
        <div style="color:#666;">{level}</div>
    </div>
    """
    
    if compliance.get('risk_keywords'):
        html += '<div style="margin-top:15px;"><h4>风险内容:</h4>'
        for kw in compliance['risk_keywords']:
            html += f'<div class="risk-item">⚠ {kw}</div>'
        html += '</div>'
    
    if compliance.get('missing_points'):
        html += '<div style="margin-top:15px;"><h4>缺失要点:</h4>'
        for point in compliance['missing_points']:
            html += f'<div class="missing-item">✗ {point}</div>'
        html += '</div>'
    
    if compliance.get('suggestions'):
        html += '<div style="margin-top:15px;"><h4>改进建议:</h4>'
        for suggestion in compliance['suggestions']:
            html += f'<div class="suggestion">→ {suggestion}</div>'
        html += '</div>'
    
    return html


def _generate_compliance_trend_html(data):
    """生成合规趋势HTML报表"""
    stats = data['statistics']
    html = f"""
    <html>
    <head>
        <title>合规趋势报表</title>
        <style>
            body {{ font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 40px; }}
            .header {{ border-bottom: 2px solid #667eea; padding-bottom: 20px; margin-bottom: 30px; }}
            .header h1 {{ color: #667eea; }}
            .stat-card {{ display: inline-block; width: 200px; padding: 20px; margin: 10px; border-radius: 12px; text-align: center; }}
            .stat-card-blue {{ background: #e6f7ff; }}
            .stat-card-green {{ background: #f6ffed; }}
            .stat-card-yellow {{ background: #fffbe6; }}
            .stat-card-red {{ background: #fff2f0; }}
            .stat-value {{ font-size: 32px; font-weight: bold; }}
            .stat-label {{ font-size: 14px; color: #666; margin-top: 8px; }}
            .section {{ margin-bottom: 30px; }}
            .section-title {{ font-size: 18px; font-weight: bold; color: #667eea; margin-bottom: 15px; }}
            .distribution-bar {{ height: 30px; background: #eee; border-radius: 4px; margin-bottom: 8px; }}
            .dist-excellent {{ background: #52c41a; }}
            .dist-good {{ background: #1890ff; }}
            .dist-pass {{ background: #faad14; }}
            .dist-fail {{ background: #ff4d4f; }}
            .trend-table {{ width: 100%; border-collapse: collapse; }}
            .trend-table th, .trend-table td {{ border: 1px solid #eee; padding: 10px; text-align: center; }}
            .trend-table th {{ background: #f8f9fa; }}
            .suggestion {{ background: #f6ffed; padding: 10px; margin: 8px 0; border-radius: 4px; border-left: 4px solid #52c41a; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>合规趋势报表</h1>
            <p>统计周期: {data['time_range']['start']} ~ {data['time_range']['end']}</p>
        </div>
        
        <div class="section">
            <div class="section-title">总体统计</div>
            <div class="stat-card stat-card-blue"><div class="stat-value">{stats['total_meetings']}</div><div class="stat-label">会议总数</div></div>
            <div class="stat-card stat-card-green"><div class="stat-value">{stats['avg_score']}</div><div class="stat-label">平均评分</div></div>
            <div class="stat-card stat-card-yellow"><div class="stat-value">{stats['avg_duration']}</div><div class="stat-label">平均时长(分钟)</div></div>
            <div class="stat-card stat-card-red"><div class="stat-value">{stats['total_risk_items']}</div><div class="stat-label">风险内容总数</div></div>
        </div>
        
        <div class="section">
            <div class="section-title">评分分布</div>
            <div>优秀 ({stats['score_distribution']['excellent']}): <div class="distribution-bar" style="width:{stats['score_distribution']['excellent']/stats['total_meetings']*100}%" class="dist-excellent"></div></div>
            <div>良好 ({stats['score_distribution']['good']}): <div class="distribution-bar" style="width:{stats['score_distribution']['good']/stats['total_meetings']*100}%" class="dist-good"></div></div>
            <div>合格 ({stats['score_distribution']['pass']}): <div class="distribution-bar" style="width:{stats['score_distribution']['pass']/stats['total_meetings']*100}%" class="dist-pass"></div></div>
            <div>不合格 ({stats['score_distribution']['fail']}): <div class="distribution-bar" style="width:{stats['score_distribution']['fail']/stats['total_meetings']*100}%" class="dist-fail"></div></div>
        </div>
        
        <div class="section">
            <div class="section-title">改进建议</div>
            {''.join([f'<div class="suggestion">{s}</div>' for s in data.get('recommendations', [])])}
        </div>
    </body>
    </html>
    """
    return html