import jieba
from collections import Counter
import re
import numpy as np


def extract_keywords(text, top_n=10, language='zh'):
    """自动抓取会议核心关键词和高频讨论内容"""
    if language == 'zh':
        words = jieba.lcut(text)

        stopwords = set([
            '的', '是', '在', '和', '了', '有', '我', '他', '她', '它',
            '这', '那', '什么', '怎么', '为什么', '因为', '所以', '但是',
            '一个', '一些', '这个', '那个', '这些', '那些', '我们', '你们', '他们',
            '可以', '可能', '应该', '需要', '会', '不会', '能', '不能',
            '说', '话', '听', '看', '想', '做', '知道', '觉得', '认为',
            '今天', '明天', '昨天', '现在', '刚才', '然后', '最后', '首先',
            '会议', '工作', '问题', '事情', '东西', '地方', '时间', '人',
            '这个', '那个', '大家', '一下', '一直', '一些', '那么', '其实',
            '可能', '关于', '通过', '进行', '方面', '情况', '这样', '那样'
        ])

        filtered_words = [
            w for w in words
            if w not in stopwords and len(w) > 1 and re.match(r'[\u4e00-\u9fa5a-zA-Z0-9]+', w)
        ]

        word_freq = Counter(filtered_words)
        keywords = word_freq.most_common(top_n)

        return [{'word': kw, 'frequency': freq} for kw, freq in keywords]
    else:
        text = re.sub(r'[^\w\s]', '', text.lower())
        words = text.split()

        stopwords_en = set([
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'shall', 'need', 'dare',
            'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
            'from', 'up', 'about', 'into', 'over', 'after', 'and', 'but', 'or',
            'as', 'if', 'when', 'than', 'because', 'while', 'although', 'though',
            'that', 'which', 'who', 'whom', 'this', 'these', 'those', 'it', 'its'
        ])

        filtered_words = [w for w in words if w not in stopwords_en and len(w) > 2]
        word_freq = Counter(filtered_words)
        keywords = word_freq.most_common(top_n)

        return [{'word': kw, 'frequency': freq} for kw, freq in keywords]


def extract_key_sentences(text, top_n=5, language='zh'):
    """提取关键句子（基于TextRank思路的简化实现）"""
    if language == 'zh':
        # 中文分句
        sentences = re.split(r'[。！？\n]', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    else:
        sentences = re.split(r'[.!?\n]', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    if len(sentences) <= top_n:
        return sentences

    # 计算每句的关键词得分
    all_keywords = extract_keywords(text, top_n=20, language=language)
    keyword_set = {kw['word'] for kw in all_keywords}

    sentence_scores = []
    for idx, sent in enumerate(sentences):
        words = jieba.lcut(sent) if language == 'zh' else sent.split()
        keyword_hits = sum(1 for w in words if w in keyword_set)

        # 归一化：关键词命中数 / 句子长度
        score = keyword_hits / (len(words) + 1)

        # 位置加权：前1/3和后1/6的句子更重要
        position = idx / len(sentences)
        if position < 0.33:
            score *= 1.2
        elif position > 0.83:
            score *= 1.1

        sentence_scores.append((idx, sent, score))

    # 按得分排序
    sentence_scores.sort(key=lambda x: x[2], reverse=True)

    # 取top_n，再按原文顺序排列
    selected = sorted(sentence_scores[:top_n], key=lambda x: x[0])

    return [s[1] for s in selected]


def generate_summary(text, max_length=300, language='zh'):
    """
    生成会议摘要：快速提炼会议重点信息
    使用抽取式摘要 + 关键信息整合
    """
    if not text or len(text.strip()) < 20:
        return "内容过短，无法生成摘要"

    # 1. 提取关键句
    key_sentences = extract_key_sentences(text, top_n=5, language=language)

    # 2. 提取关键词
    keywords = extract_keywords(text, top_n=10, language=language)
    keyword_str = "、".join([kw['word'] for kw in keywords[:5]])

    # 3. 分析主题
    topics = analyze_topic(text, language=language)

    # 4. 组装摘要
    summary_parts = []

    # 主题概括
    if topics:
        top_topic = topics[0]['topic']
        summary_parts.append(f"会议主要围绕{top_topic}展开讨论")

    # 关键词概括
    if keywords:
        summary_parts.append(f"核心议题包括{keyword_str}")

    # 关键句
    if key_sentences:
        summary_parts.append("主要内容：" + "。".join(key_sentences[:3]))

    summary = "。".join(summary_parts)

    # 控制长度
    if len(summary) > max_length:
        summary = summary[:max_length] + "..."

    return summary


def analyze_topic(text, language='zh', candidate_topics=None):
    """对全文内容进行语义分析，识别讨论主题和核心方向"""
    default_topics = ["工作汇报", "项目讨论", "问题解决", "决策制定",
                      "进度跟进", "计划安排", "意见交流", "培训学习"]
    topics = candidate_topics or default_topics

    try:
        from sentence_transformers import SentenceTransformer, util

        model_name = "BAAI/bge-small-zh-v1.5" if language == 'zh' else "all-MiniLM-L6-v2"

        model = SentenceTransformer(model_name)
        text_embedding = model.encode(text, convert_to_tensor=True)
        topic_embeddings = model.encode(topics, convert_to_tensor=True)

        cosine_scores = util.cos_sim(text_embedding, topic_embeddings)
        topic_scores = list(zip(topics, cosine_scores[0].tolist()))

        sorted_topics = sorted(topic_scores, key=lambda x: x[1], reverse=True)

        return [{'topic': topic, 'score': round(score, 4)} for topic, score in sorted_topics[:5]]

    except Exception as e:
        print(f"语义分析失败，使用关键词匹配: {e}")
        return keyword_based_topic(text, topics)


def keyword_based_topic(text, topics):
    """基于关键词的主题分析（降级方案）"""
    topic_keywords = {
        "工作汇报": ["汇报", "总结", "报告", "进度", "完成"],
        "项目讨论": ["项目", "方案", "设计", "开发", "实施"],
        "问题解决": ["问题", "困难", "解决", "处理", "方案"],
        "决策制定": ["决定", "决策", "批准", "同意", "方案"],
        "进度跟进": ["进度", "进展", "跟进", "更新", "状态"],
        "计划安排": ["计划", "安排", "规划", "部署", "安排"],
        "意见交流": ["意见", "建议", "讨论", "交流", "想法"],
        "培训学习": ["培训", "学习", "课程", "讲座", "知识"]
    }

    scores = []
    for topic in topics:
        keywords = topic_keywords.get(topic, [])
        count = sum(1 for kw in keywords if kw in text)
        score = count / max(len(keywords), 1)
        scores.append((topic, score))

    sorted_topics = sorted(scores, key=lambda x: x[1], reverse=True)
    return [{'topic': topic, 'score': round(score, 4)} for topic, score in sorted_topics[:5]]


def analyze_sentiment(text, language='zh'):
    """分析会议情绪倾向（辅助风险识别）"""
    if language == 'zh':
        positive_words = ["好", "优秀", "完成", "成功", "进步", "提升", "积极", "满意", "支持", "同意"]
        negative_words = ["消极", "反对", "抵制", "抱怨", "不满", "拒绝", "不行", "不可能", "做不到", "失败"]

        words = jieba.lcut(text)
        positive_count = sum(1 for w in words if w in positive_words)
        negative_count = sum(1 for w in words if w in negative_words)

        total = positive_count + negative_count
        if total == 0:
            return {'sentiment': 'neutral', 'score': 0.5, 'positive': 0, 'negative': 0}

        score = positive_count / total
        if score > 0.6:
            sentiment = 'positive'
        elif score < 0.4:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'

        return {
            'sentiment': sentiment,
            'score': round(score, 4),
            'positive': positive_count,
            'negative': negative_count
        }

    return {'sentiment': 'neutral', 'score': 0.5, 'positive': 0, 'negative': 0}