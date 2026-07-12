"""
文本分析模块
关键词提取、摘要生成、主题分析、情绪分析
使用TF-IDF、TextRank、词性过滤等技术提升精准度
"""
import jieba
import jieba.posseg as pseg
from collections import Counter, defaultdict
import re
import numpy as np
import math

# 缓存sentence_transformers模型，避免每次分析都重新加载
_sentence_transformer_models = {}

def get_sentence_transformer_model(language='zh'):
    """获取缓存的sentence_transformers模型"""
    model_name = "BAAI/bge-small-zh-v1.5" if language == 'zh' else "all-MiniLM-L6-v2"
    
    if model_name not in _sentence_transformer_models:
        try:
            from sentence_transformers import SentenceTransformer
            _sentence_transformer_models[model_name] = SentenceTransformer(model_name)
            print(f"Loaded sentence_transformer model: {model_name}")
        except Exception as e:
            print(f"Failed to load sentence_transformer model: {e}")
            return None
    
    return _sentence_transformer_models[model_name]


# 停用词表
STOPWORDS_ZH = set([
    '的', '是', '在', '和', '了', '有', '我', '他', '她', '它',
    '这', '那', '什么', '怎么', '为什么', '因为', '所以', '但是',
    '一个', '一些', '这个', '那个', '这些', '那些', '我们', '你们', '他们',
    '可以', '可能', '应该', '需要', '会', '不会', '能', '不能',
    '说', '话', '听', '看', '想', '做', '知道', '觉得', '认为',
    '今天', '明天', '昨天', '现在', '刚才', '然后', '最后', '首先',
    '这个', '那个', '大家', '一下', '一直', '一些', '那么', '其实',
    '可能', '关于', '通过', '进行', '方面', '情况', '这样', '那样',
    '就是', '还是', '或者', '而且', '并且', '以及', '等', '的话',
    '起来', '下来', '出去', '过来', '过去', '回来', '回去',
    '一下', '一些', '一点', '一切', '一边', '一面',
    '它们', '她们', '它们', '自己', '别人', '大家', '咱们', '您们',
    '这里', '那里', '哪里', '到处', '处处',
    '这个', '那个', '这些', '那些', '这样', '那样',
    '什么', '怎么', '怎样', '多少', '为什么', '如何',
    '是', '有', '在', '不', '了', '着', '过', '的', '地', '得',
    '吗', '呢', '吧', '啊', '哦', '嗯', '呀', '哈', '嘿',
    '就', '都', '也', '还', '只', '才', '又', '再', '更', '最',
    '很', '非常', '特别', '十分', '比较', '相当', '极其', '格外',
    '已经', '曾经', '正在', '将要', '即将', '马上', '立刻',
    '没有', '没', '别', '莫', '勿', '休',
])


def extract_keywords(text, top_n=10, language='zh'):
    """
    自动抓取会议核心关键词和高频讨论内容
    改进：使用TF-IDF + 词性过滤，只保留名词、动词、专有名词
    """
    if language == 'zh':
        # 使用jieba.posseg进行词性标注
        words_with_pos = pseg.lcut(text)

        # 只保留有意义的词性：名词、动词、专有名词
        valid_pos = ['n', 'nr', 'ns', 'nt', 'nz', 'v', 'vn', 'an', 'a', 'i', 'l', 'g']
        
        # 扩展停用词：过滤常见的无意义高频词
        domain_stopwords = set([
            '会议', '内容', '方面', '问题', '事情', '东西', '地方', '时间', '人员',
            '情况', '状态', '方式', '方法', '过程', '结果', '原因', '目的',
            '大家', '各位', '领导', '同事', '今天', '现在', '然后', '觉得',
            '进行', '通过', '关于', '对于', '根据', '按照', '作为', '为了',
            '这个', '那个', '这些', '那些', '这样', '那样', '什么', '怎么',
            # 过滤代词和人称
            '我', '你', '他', '她', '我们', '你们', '他们', '她们',
            '自己', '别人', '本人', '有人',
            # 过滤常见无意义词
            '一般', '有点', '一下', '一些', '一点', '一直', '一定', '一样',
            '可以', '可能', '应该', '会', '不会', '能', '不能',
            '没有', '没什么',
            # 过滤常见人名模式（小名+姓氏）
            '小王', '小张', '小李', '小刘', '小陈', '小杨', '小黄', '小吴',
            '小赵', '小钱', '小孙', '小周', '小徐', '小朱', '小胡', '小郭',
            '小林', '小马', '小罗', '小高', '小郑', '小梁', '小宋', '小唐',
        ])

        filtered_words = []
        for word, pos in words_with_pos:
            if (pos in valid_pos and 
                word not in STOPWORDS_ZH and 
                word not in domain_stopwords and
                len(word) >= 2 and
                re.match(r'[\u4e00-\u9fa5a-zA-Z0-9]+', word)):
                filtered_words.append(word)

        if not filtered_words:
            return []

        # TF-IDF计算
        word_freq = Counter(filtered_words)
        total_words = len(filtered_words)

        # 计算每个词的TF-IDF分数
        # IDF用文档频率近似（这里用词在文中的段落分布）
        sentences = re.split(r'[。！？\n]', text)
        total_sentences = max(len(sentences), 1)
        
        word_doc_freq = defaultdict(int)
        for word in set(filtered_words):
            for sent in sentences:
                if word in sent:
                    word_doc_freq[word] += 1
        
        keywords_with_scores = []
        for word, freq in word_freq.items():
            tf = freq / total_words
            # IDF: 词在多少句子中出现，越集中越重要
            doc_freq = word_doc_freq[word]
            idf = math.log((total_sentences + 1) / (doc_freq + 1)) + 1
            
            tfidf_score = tf * idf
            
            # 词性加权：专有名词和名词权重更高
            pos_bonus = 1.0
            for w, p in words_with_pos:
                if w == word:
                    if p in ['nz', 'nr', 'ns', 'nt']:  # 专有名词
                        pos_bonus = 1.5
                    elif p in ['n', 'vn']:  # 名词
                        pos_bonus = 1.3
                    elif p in ['v']:  # 动词
                        pos_bonus = 1.1
                    break
            
            final_score = tfidf_score * pos_bonus
            keywords_with_scores.append({
                'word': word, 
                'frequency': freq, 
                'score': round(final_score, 4)
            })

        keywords_with_scores.sort(key=lambda x: x['score'], reverse=True)
        return keywords_with_scores[:top_n]

    else:
        text = re.sub(r'[^\w\s]', '', text.lower())
        words = text.split()

        stopwords_en = set([
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'shall', 'need', 'dare',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'up',
            'about', 'into', 'over', 'after', 'and', 'but', 'or', 'as', 'if',
            'when', 'than', 'because', 'while', 'although', 'though', 'that',
            'which', 'who', 'whom', 'this', 'these', 'those', 'it', 'its',
            'meeting', 'time', 'people', 'thing', 'something', 'anything',
        ])

        filtered_words = [w for w in words if w not in stopwords_en and len(w) > 2]
        word_freq = Counter(filtered_words)
        keywords = word_freq.most_common(top_n)
        return [{'word': kw, 'frequency': freq, 'score': freq / len(filtered_words)} for kw, freq in keywords]


def extract_key_sentences(text, top_n=5, language='zh'):
    """
    提取关键句子 - 基于TextRank算法
    改进：使用句子相似度构建图，通过PageRank迭代计算重要性
    """
    if language == 'zh':
        sentences = re.split(r'[。！？\n]', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 8]
    else:
        sentences = re.split(r'[.!?\n]', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    if len(sentences) <= top_n:
        return sentences

    # 1. 对每个句子分词
    sentence_words = []
    for sent in sentences:
        words = [w for w in jieba.lcut(sent) if len(w) > 1 and w not in STOPWORDS_ZH]
        sentence_words.append(words)

    # 2. 计算句子间的相似度（基于共同词的比例）
    n = len(sentences)
    similarity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            # Jaccard相似度的改进版：用共同词数 / 较短句子长度
            common = len(set(sentence_words[i]) & set(sentence_words[j]))
            shorter = min(len(sentence_words[i]), len(sentence_words[j]))
            if shorter > 0:
                sim = common / shorter
            else:
                sim = 0
            similarity_matrix[i][j] = sim
            similarity_matrix[j][i] = sim

    # 3. TextRank / PageRank 迭代
    damping = 0.85
    scores = np.ones(n) / n
    
    for _ in range(50):  # 最多迭代50次
        new_scores = np.zeros(n)
        for i in range(n):
            total_sim = sum(similarity_matrix[i])
            if total_sim > 0:
                for j in range(n):
                    if i != j and similarity_matrix[j][i] > 0:
                        new_scores[i] += (similarity_matrix[j][i] / sum(similarity_matrix[j])) * scores[j]
            new_scores[i] = (1 - damping) / n + damping * new_scores[i]
        
        # 检查收敛
        if np.sum(np.abs(new_scores - scores)) < 0.0001:
            break
        scores = new_scores

    # 4. 位置加权：开头和结尾的句子通常更重要
    for i in range(n):
        position = i / n
        if position < 0.2:  # 前20%的句子
            scores[i] *= 1.3
        elif position > 0.8:  # 后20%的句子
            scores[i] *= 1.1

    # 5. 关键词命中加权
    all_keywords = extract_keywords(text, top_n=20, language=language)
    keyword_set = {kw['word'] for kw in all_keywords}

    for i, sent_words in enumerate(sentence_words):
        keyword_hits = sum(1 for w in sent_words if w in keyword_set)
        if keyword_hits > 0:
            scores[i] *= (1 + keyword_hits * 0.1)

    # 6. 选取得分最高的句子，按原文顺序排列
    ranked_indices = sorted(range(n), key=lambda x: scores[x], reverse=True)
    selected_indices = sorted(ranked_indices[:top_n])

    return [sentences[i] for i in selected_indices]


def _is_low_quality_sentence(sentence):
    """判断句子是否为低质量（重复、无意义）"""
    if not sentence or len(sentence) < 5:
        return True
    if re.search(r'(.{2,})\1{2,}', sentence):
        return True
    if re.search(r'(.)\1{3,}', sentence):
        return True
    words = jieba.lcut(sentence)
    unique_ratio = len(set(words)) / max(len(words), 1)
    if unique_ratio < 0.3 and len(words) > 10:
        return True
    return False


def _convert_person(text):
    """
    将第一人称转换为第三人称，将对话式语气转换为叙述式
    """
    if not text:
        return ""
    
    # 第一人称转换
    text = text.replace('我', '发言者')
    text = text.replace('我们', '会议')
    text = text.replace('咱们', '参会人员')
    text = text.replace('大家', '参会人员')
    text = text.replace('同事', '参会人员')
    
    # 问候语和开场白过滤
    greeting_patterns = [
        r'大家好[，。！]',
        r'各位同事[，。！]',
        r'今天我们[来]?讨论',
        r'今天我们[来]?开个会',
        r'好的[，。]',
        r'那么[，。]',
        r'首先[，。]',
        r'最后[，。]',
        r'好的，今天的会议就到这里',
        r'散会[，。！]',
        r'谢谢大家[，。！]',
        r'感谢大家[，。！]',
        r'非常感谢[，。！]',
    ]
    for pattern in greeting_patterns:
        text = re.sub(pattern, '', text)
    
    # 语气词去除
    text = re.sub(r'[呢吧啊哦嗯呀哈嘿嘛咯]', '', text)
    
    # 去除多余标点和空白
    text = re.sub(r'[，,]+', '，', text)
    text = re.sub(r'[。.]+', '。', text)
    text = re.sub(r'\s+', '', text)
    
    return text.strip()


def _summarize_content(key_sentences):
    """
    真正的总结提炼：不是简单拼接，而是用第三人称叙述，重新组织语言
    """
    if not key_sentences:
        return ""
    
    converted = [_convert_person(s) for s in key_sentences]
    converted = [s for s in converted if s and len(s) > 8]
    
    if not converted:
        return ""
    
    summary_parts = []
    
    # 1. 识别讨论的主题/对象
    topic_patterns = [
        r'(讨论|研究|审议|探讨)\s+(.{2,20})',
        r'(关于|针对)\s+(.{2,20})的讨论',
        r'(介绍|讲解|说明)\s+(.{2,20})',
    ]
    topics_found = []
    for sent in converted:
        for pattern in topic_patterns:
            match = re.search(pattern, sent)
            if match:
                topic = match.group(2).strip()
                if topic not in topics_found:
                    topics_found.append(topic)
    
    if topics_found:
        summary_parts.append(f"围绕{'、'.join(topics_found)}等议题进行讨论")
    
    # 2. 识别主要观点/决定
    decision_patterns = [
        r'(决定|决议|同意|批准|通过)\s+(.{2,30})',
        r'(明确|确定)\s+(.{2,20})',
        r'(达成|形成)\s+(共识|决议|一致意见)',
    ]
    decisions_found = []
    for sent in converted:
        for pattern in decision_patterns:
            match = re.search(pattern, sent)
            if match:
                content = match.group(0).strip()
                if content not in decisions_found:
                    decisions_found.append(content)
    
    if decisions_found:
        summary_parts.append("会议" + "，".join(decisions_found))
    
    # 3. 识别要求/行动项
    requirement_patterns = [
        r'(要求|必须|应当|应该)\s+(.{2,40})',
        r'(提出|建议)\s+(.{2,30})',
    ]
    requirements_found = []
    for sent in converted:
        for pattern in requirement_patterns:
            match = re.search(pattern, sent)
            if match:
                content = match.group(0).strip()
                if content not in requirements_found:
                    requirements_found.append(content)
    
    if requirements_found:
        summary_parts.append("强调" + "；".join(requirements_found))
    
    # 4. 如果没有匹配到模式，用更智能的方式总结
    if not summary_parts:
        all_text = "。".join(converted)
        keywords = extract_keywords(all_text, top_n=5)
        keyword_str = "、".join([kw['word'] for kw in keywords])
        
        action_words = ['讨论', '分析', '决定', '要求', '提出', '确认', '明确', '达成']
        actions_found = []
        for sent in converted:
            for action in action_words:
                if action in sent:
                    idx = sent.find(action)
                    action_part = sent[idx:idx+30].strip()
                    if action_part not in actions_found:
                        actions_found.append(action_part)
        
        if actions_found:
            summary_parts.append(f"会议讨论了{keyword_str}等内容" + "；".join(actions_found[:2]))
        else:
            if len(converted) <= 2:
                summary_parts.append("会议讨论了" + "；".join(converted))
            else:
                merged = []
                for sent in converted:
                    is_duplicate = False
                    for existing in merged:
                        if sent in existing or existing in sent:
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        merged.append(sent)
                summary_parts.append("会议讨论了" + "；".join(merged[:2]))
    
    return "。".join(summary_parts)


def generate_summary(text, max_length=500, language='zh'):
    """
    生成会议摘要：快速提炼会议重点信息
    改进：第三人称叙述、真正的总结提炼、过滤对话语气
    """
    if not text or len(text.strip()) < 20:
        return "内容过短，无法生成摘要"

    # 1. 提取关键句（TextRank）
    key_sentences = extract_key_sentences(text, top_n=10, language=language)
    key_sentences = [s for s in key_sentences if not _is_low_quality_sentence(s)]

    # 2. 提取关键词
    keywords = extract_keywords(text, top_n=8, language=language)

    # 3. 分析主题
    topics = analyze_topic(text, language=language)

    # 4. 分析情绪
    sentiment = analyze_sentiment(text, language=language)

    # 5. 识别会议动作项（需要做的事）
    action_items = extract_action_items(text, language=language)
    action_items = [_convert_person(item) for item in action_items if _convert_person(item)]

    # 6. 组装结构化摘要
    summary_parts = []

    # 主题概括
    if topics and topics[0]['score'] > 0:
        top_topics = [t['topic'] for t in topics[:3] if t['score'] > 0]
        if top_topics:
            summary_parts.append(f"本次会议主要围绕{'、'.join(top_topics)}展开")

    # 关键议题
    if keywords:
        keyword_str = "、".join([kw['word'] for kw in keywords[:6]])
        summary_parts.append(f"核心议题包括：{keyword_str}")

    # 主要内容（真正的总结提炼，不是原句拼接）
    summarized_content = _summarize_content(key_sentences)
    if summarized_content:
        summary_parts.append(f"主要内容：{summarized_content}")

    # 动作项
    if action_items:
        action_str = "；".join(action_items[:3])
        summary_parts.append(f"后续行动：{action_str}")

    # 情绪倾向
    if sentiment['sentiment'] != 'neutral':
        sentiment_desc = "积极" if sentiment['sentiment'] == 'positive' else "消极"
        summary_parts.append(f"整体氛围：{sentiment_desc}")

    summary = "。".join(summary_parts)

    # 最后确保没有繁体字
    from modules.whisper_utils import fix_traditional_chinese
    summary = fix_traditional_chinese(summary)

    # 控制长度
    if len(summary) > max_length:
        summary = summary[:max_length - 3] + "..."

    return summary


def extract_action_items(text, language='zh'):
    """
    从文本中提取行动项/决议
    识别"需要"、"应该"、"决定"、"负责"、"完成"等关键词引导的句子
    """
    if language == 'zh':
        action_patterns = [
            r'[需要必须应该要求][^。！？]*[。！？]',
            r'[决定决议同意批准][^。！？]*[。！？]',
            r'[负责跟进落实执行][^。！？]*[。！？]',
            r'[计划预计安排][^。！？]*[。！？]',
            r'[下周下月近期][^。！？]*[。！？]',
        ]
    else:
        action_patterns = [
            r'[Nn]eed to[^.!?]*[.!?]',
            r'[Mm]ust[^.!?]*[.!?]',
            r'[Ss]hould[^.!?]*[.!?]',
            r'[Dd]ecide[^.!?]*[.!?]',
            r'[Pp]lan[^.!?]*[.!?]',
        ]

    action_items = []
    for pattern in action_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            clean = match.strip('。！？. \n')
            if len(clean) > 5 and clean not in action_items:
                action_items.append(clean)

    return action_items[:5]


def analyze_topic(text, language='zh', candidate_topics=None):
    """
    对全文内容进行语义分析，识别讨论主题和核心方向
    改进：更细化的主题词库 + 语义相似度
    """
    # 扩展主题词库（更细粒度）
    default_topics = [
        "工作汇报", "项目讨论", "问题解决", "决策制定",
        "进度跟进", "计划安排", "意见交流", "培训学习",
        "风险讨论", "合规审查", "财务预算", "人事安排",
        "客户服务", "产品设计", "技术方案", "市场营销"
    ]
    topics = candidate_topics or default_topics

    # 更详细的主题关键词映射
    topic_keywords = {
        "工作汇报": ["汇报", "总结", "报告", "完成", "进展", "成果", "回顾", "梳理", "概况", "数据"],
        "项目讨论": ["项目", "方案", "设计", "开发", "实施", "规划", "推进", "启动", "阶段", "里程碑"],
        "问题解决": ["问题", "困难", "解决", "处理", "应对", "排除", "修复", "改善", "优化", "突破"],
        "决策制定": ["决定", "决策", "批准", "同意", "通过", "审议", "表决", "决议", "采纳", "确认"],
        "进度跟进": ["进度", "进展", "跟进", "更新", "状态", "检查", "追踪", "督办", "落实", "执行"],
        "计划安排": ["计划", "安排", "规划", "部署", "排期", "时间表", "日程", "路线图", "目标", "任务"],
        "意见交流": ["意见", "建议", "讨论", "交流", "想法", "看法", "观点", "反馈", "提议", "倡议"],
        "培训学习": ["培训", "学习", "课程", "讲座", "知识", "教学", "指导", "培训", "分享", "研讨"],
        "风险讨论": ["风险", "隐患", "问题", "预警", "管控", "防范", "评估", "识别", "应对", "控制"],
        "合规审查": ["合规", "规定", "制度", "政策", "规范", "标准", "要求", "检查", "审核", "遵从"],
        "财务预算": ["预算", "费用", "成本", "支出", "收入", "资金", "财务", "报销", "拨款", "核算"],
        "人事安排": ["人员", "招聘", "分配", "调动", "任命", "职责", "团队", "岗位", "编制", "考核"],
        "客户服务": ["客户", "服务", "满意度", "需求", "反馈", "投诉", "售后", "维护", "回访", "支持"],
        "产品设计": ["产品", "设计", "功能", "体验", "界面", "需求", "原型", "测试", "迭代", "用户"],
        "技术方案": ["技术", "架构", "系统", "方案", "实现", "接口", "模块", "部署", "性能", "优化"],
        "市场营销": ["市场", "营销", "推广", "品牌", "宣传", "渠道", "客户", "销售", "策略", "活动"],
    }

    # 先尝试语义相似度
    try:
        from sentence_transformers import util

        model = get_sentence_transformer_model(language)
        if model is None:
            raise Exception("sentence_transformer model not available")

        # 将文本截取（避免过长文本导致编码慢）
        text_segment = text[:2000] if len(text) > 2000 else text
        text_embedding = model.encode(text_segment, convert_to_tensor=True)
        topic_embeddings = model.encode(topics, convert_to_tensor=True)

        cosine_scores = util.cos_sim(text_embedding, topic_embeddings)
        topic_scores = list(zip(topics, cosine_scores[0].tolist()))

        # 结合关键词匹配的得分
        enhanced_scores = []
        for topic, sim_score in topic_scores:
            # 关键词命中得分
            kw_list = topic_keywords.get(topic, [])
            kw_hits = sum(1 for kw in kw_list if kw in text)
            kw_score = min(kw_hits / max(len(kw_list), 1), 1.0)

            # 综合得分：语义相似度70% + 关键词匹配30%
            combined = sim_score * 0.7 + kw_score * 0.3
            enhanced_scores.append((topic, combined, sim_score, kw_score))

        enhanced_scores.sort(key=lambda x: x[1], reverse=True)

        result = []
        for topic, combined, sim, kw in enhanced_scores[:5]:
            result.append({
                'topic': topic,
                'score': round(combined, 4),
                'semantic_score': round(sim, 4),
                'keyword_score': round(kw, 4)
            })
        return result

    except Exception as e:
        print(f"语义分析失败，使用关键词匹配: {e}")
        return keyword_based_topic(text, topics, topic_keywords)


def keyword_based_topic(text, topics, topic_keywords=None):
    """基于关键词的主题分析（降级方案）- 改进版"""
    if topic_keywords is None:
        topic_keywords = {
            "工作汇报": ["汇报", "总结", "报告", "完成", "进展", "成果"],
            "项目讨论": ["项目", "方案", "设计", "开发", "实施", "推进"],
            "问题解决": ["问题", "困难", "解决", "处理", "应对", "修复"],
            "决策制定": ["决定", "决策", "批准", "同意", "通过", "决议"],
            "进度跟进": ["进度", "进展", "跟进", "更新", "状态", "追踪"],
            "计划安排": ["计划", "安排", "规划", "部署", "排期", "目标"],
            "意见交流": ["意见", "建议", "讨论", "交流", "想法", "反馈"],
            "培训学习": ["培训", "学习", "课程", "讲座", "知识", "研讨"],
            "风险讨论": ["风险", "隐患", "预警", "管控", "防范", "评估"],
            "合规审查": ["合规", "规定", "制度", "政策", "规范", "审核"],
            "财务预算": ["预算", "费用", "成本", "支出", "资金", "财务"],
            "人事安排": ["人员", "招聘", "分配", "调动", "任命", "职责"],
        }

    scores = []
    for topic in topics:
        keywords = topic_keywords.get(topic, [])
        # 统计关键词出现次数（不只是出现与否）
        total_hits = sum(text.count(kw) for kw in keywords)
        unique_hits = sum(1 for kw in keywords if kw in text)
        
        # 得分 = 唯一命中率 * 0.6 + 总命中密度 * 0.4
        unique_rate = unique_hits / max(len(keywords), 1)
        density = min(total_hits / max(len(text) / 100, 1), 1.0)
        score = unique_rate * 0.6 + density * 0.4
        
        scores.append((topic, score, unique_hits, total_hits))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [{'topic': t, 'score': round(s, 4), 'keyword_hits': h, 'total_hits': th} 
            for t, s, h, th in scores[:5]]


def analyze_sentiment(text, language='zh'):
    """
    分析会议情绪倾向（辅助风险识别）
    改进：扩充情感词典 + 否定词处理 + 程度副词加权
    """
    if language == 'zh':
        # 扩充的情感词典
        positive_words = {
            '好': 1, '优秀': 2, '完成': 1, '成功': 2, '进步': 1, '提升': 1, '积极': 1,
            '满意': 2, '支持': 1, '同意': 1, '通过': 1, '批准': 1, '赞': 1, '赞成': 1,
            '顺利': 1, '突破': 2, '创新': 1, '高效': 1, '优化': 1, '改善': 1,
            '增长': 1, '盈利': 1, '超越': 1, '达成': 1, '实现': 1, '解决': 1,
            '加强': 1, '推进': 1, '落实': 1, '成效': 1, '成果': 1, '收获': 1,
        }
        
        negative_words = {
            '消极': 1, '反对': 1, '抵制': 2, '抱怨': 1, '不满': 1, '拒绝': 1,
            '不行': 1, '不可能': 1, '做不到': 1, '失败': 2, '延误': 1, '延迟': 1,
            '风险': 1, '隐患': 1, '问题': 1, '困难': 1, '障碍': 1, '阻碍': 1,
            '下降': 1, '减少': 1, '亏损': 2, '流失': 1, '违规': 2, '违纪': 2,
            '偏离': 1, '不当': 1, '不足': 1, '缺乏': 1, '缺失': 1, '遗漏': 1,
            '警告': 1, '处罚': 1, '违规': 2, '违法': 2, '投诉': 1, '纠纷': 1,
        }
        
        # 否定词
        negation_words = {'不', '没', '无', '非', '未', '别', '莫', '勿', '否'}
        # 程度副词
        degree_words = {'非常': 1.5, '十分': 1.5, '特别': 1.5, '极其': 2, '格外': 1.5,
                       '比较': 1.2, '相当': 1.3, '很': 1.2, '太': 1.5, '极为': 2}

        words = jieba.lcut(text)
        
        positive_score = 0
        negative_score = 0
        positive_count = 0
        negative_count = 0

        i = 0
        while i < len(words):
            w = words[i]
            
            # 检查是否是程度副词
            degree = 1.0
            if w in degree_words:
                degree = degree_words[w]
                i += 1
                if i >= len(words):
                    break
                w = words[i]
            
            # 检查否定词
            negated = False
            if w in negation_words:
                negated = True
                i += 1
                if i >= len(words):
                    break
                w = words[i]
            
            # 计算情感得分
            if w in positive_words:
                score = positive_words[w] * degree
                if negated:
                    negative_score += score
                    negative_count += 1
                else:
                    positive_score += score
                    positive_count += 1
            elif w in negative_words:
                score = negative_words[w] * degree
                if negated:
                    positive_score += score
                    positive_count += 1
                else:
                    negative_score += score
                    negative_count += 1
            
            i += 1

        total = positive_score + negative_score
        if total == 0:
            return {'sentiment': 'neutral', 'score': 0.5, 'positive': 0, 'negative': 0}

        score = positive_score / total
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
            'negative': negative_count,
            'positive_score': round(positive_score, 2),
            'negative_score': round(negative_score, 2)
        }

    return {'sentiment': 'neutral', 'score': 0.5, 'positive': 0, 'negative': 0}
