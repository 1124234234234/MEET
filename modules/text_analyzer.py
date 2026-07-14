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

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

_sumy_summarizer = None

def get_sumy_summarizer():
    global _sumy_summarizer
    if _sumy_summarizer is None:
        _sumy_summarizer = LsaSummarizer()
    return _sumy_summarizer

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


def _smart_segment(text):
    """
    智能分句：处理无标点或标点不全的中文文本
    优先使用现有标点，不足时基于词汇和模式进行补充分句
    """
    text = text.strip()
    if not text:
        return []
    
    raw_sentences = re.split(r'[。！？!?\n]', text)
    raw_sentences = [s.strip() for s in raw_sentences if s.strip() and len(s.strip()) > 5]
    
    if len(raw_sentences) >= 8:
        return raw_sentences
    
    sentence_break_words = [
        '我认为', '我觉得', '我建议', '我同意', '我强调', '我补充',
        '另外，', '另外', '还有，', '还有', '对了，', '对了',
        '而且', '但是', '不过', '所以', '因此',
        '第一，', '第一', '第二，', '第二', '第三，', '第三',
        '第四，', '第四', '第五，', '第五',
        '首先', '其次', '最后',
        '这款', '这个产品', '那这个',
        '好的，', '好的',
        '必须', '要求', '禁止', '需要', '建议', '决定',
        '会议', '讨论', '总结',
        '今天我们', '今天',
        '我们禁止', '我们必须', '我们要求', '我们决定', '我们建议',
        '提醒客户', '提醒大家',
    ]
    
    result = []
    for sent in raw_sentences:
        if len(sent) < 40:
            result.append(sent)
            continue
        
        sub_parts = [sent]
        for word in sentence_break_words:
            new_parts = []
            for part in sub_parts:
                if len(part) < 40:
                    new_parts.append(part)
                    continue
                split_parts = re.split(f'(?={re.escape(word)})', part)
                new_parts.extend([p for p in split_parts if p.strip()])
            sub_parts = new_parts
        
        for part in sub_parts:
            part = part.strip('，,、；; ')
            if len(part) > 8:
                result.append(part)
    
    result = [s for s in result if len(s) > 10]
    return result if result else raw_sentences


def extract_key_sentences_from_list(sentences, top_n=5):
    """
    从句子列表中提取关键句 - 基于TextRank算法
    """
    if len(sentences) <= top_n:
        return sentences

    sentence_words = []
    for sent in sentences:
        words = [w for w in jieba.lcut(sent) if len(w) > 1 and w not in STOPWORDS_ZH]
        sentence_words.append(words)

    n = len(sentences)
    similarity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            common = len(set(sentence_words[i]) & set(sentence_words[j]))
            shorter = min(len(sentence_words[i]), len(sentence_words[j]))
            if shorter > 0:
                sim = common / shorter
            else:
                sim = 0
            similarity_matrix[i][j] = sim
            similarity_matrix[j][i] = sim

    damping = 0.85
    scores = np.ones(n) / n

    for _ in range(20):
        new_scores = np.zeros(n)
        for i in range(n):
            total_sim = sum(similarity_matrix[i])
            if total_sim > 0:
                for j in range(n):
                    if i != j and similarity_matrix[j][i] > 0:
                        new_scores[i] += (similarity_matrix[j][i] / sum(similarity_matrix[j])) * scores[j]
            new_scores[i] = (1 - damping) / n + damping * new_scores[i]

        if np.sum(np.abs(new_scores - scores)) < 0.001:
            break
        scores = new_scores

    for i in range(n):
        position = i / n
        if position < 0.2:
            scores[i] *= 1.3
        elif position > 0.8:
            scores[i] *= 1.1

    all_text = ''.join(sentences)
    all_keywords = extract_keywords(all_text, top_n=20)
    keyword_set = {kw['word'] for kw in all_keywords}

    for i, sent_words in enumerate(sentence_words):
        keyword_hits = sum(1 for w in sent_words if w in keyword_set)
        if keyword_hits > 0:
            scores[i] *= (1 + keyword_hits * 0.1)

    ranked_indices = sorted(range(n), key=lambda x: scores[x], reverse=True)
    selected_indices = sorted(ranked_indices[:top_n])

    return [sentences[i] for i in selected_indices]


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

    for _ in range(20):
        new_scores = np.zeros(n)
        for i in range(n):
            total_sim = sum(similarity_matrix[i])
            if total_sim > 0:
                for j in range(n):
                    if i != j and similarity_matrix[j][i] > 0:
                        new_scores[i] += (similarity_matrix[j][i] / sum(similarity_matrix[j])) * scores[j]
            new_scores[i] = (1 - damping) / n + damping * new_scores[i]

        if np.sum(np.abs(new_scores - scores)) < 0.001:
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


def _convert_to_narrative(sentence):
    """
    将单句转换为叙述式第三人称 - 增强版
    """
    if not sentence:
        return ""
    
    text = sentence.strip()
    
    greeting_remove = [
        r'^各位同事大家好[，,。.]?\s*',
        r'^大家好[，,。.]?\s*',
        r'^好的，谢谢大家[，,。.]?\s*',
        r'^谢谢大家[，,。.]?\s*',
        r'^好的[，,。.]\s*',
        r'^嗯[，,。.]\s*',
        r'^对[，,。.]\s*',
        r'^是[，,。.]\s*',
        r'^没错[，,。.]\s*',
        r'^行[，,。.]\s*',
        r'^可以[，,。.]\s*',
        r'^明白了[，,。.]\s*',
        r'^知道了[，,。.]\s*',
    ]
    for pattern in greeting_remove:
        text = re.sub(pattern, '', text)
    
    text = re.sub(r'今天我们来讨论', '会议讨论', text)
    text = re.sub(r'今天我们讨论', '会议讨论', text)
    text = re.sub(r'今天主要讲', '会议主要讨论', text)
    text = re.sub(r'今天主要说', '会议主要讨论', text)
    text = re.sub(r'我有个问题', '有观点提出疑问', text)
    text = re.sub(r'我有个疑问', '有观点提出疑问', text)
    text = re.sub(r'我想问一下', '有观点提出疑问', text)
    text = re.sub(r'我想请教', '有观点提出疑问', text)
    
    text = re.sub(r'^还有[，,，]?\s*', '', text)
    text = re.sub(r'^另外[，,，]?\s*', '', text)
    text = re.sub(r'^而且[，,，]?\s*', '', text)
    text = re.sub(r'^但是[，,，]?\s*', '', text)
    text = re.sub(r'^不过[，,，]?\s*', '', text)
    text = re.sub(r'^那么[，,，]?\s*', '', text)
    text = re.sub(r'^所以[，,，]?\s*', '', text)
    text = re.sub(r'^因此[，,，]?\s*', '', text)
    text = re.sub(r'^因为[，,，]?\s*', '', text)
    text = re.sub(r'^其实[，,，]?\s*', '', text)
    text = re.sub(r'^然后[，,，]?\s*', '', text)
    text = re.sub(r'^接下来[，,，]?\s*', '', text)
    text = re.sub(r'^首先[，,，]?\s*', '', text)
    text = re.sub(r'^其次[，,，]?\s*', '', text)
    text = re.sub(r'^最后[，,，]?\s*', '', text)
    text = re.sub(r'^还有', '', text)
    text = re.sub(r'^另外', '', text)
    text = re.sub(r'^而且', '', text)
    text = re.sub(r'^但是', '', text)
    text = re.sub(r'^不过', '', text)
    text = re.sub(r'^那么', '', text)
    text = re.sub(r'^所以', '', text)
    text = re.sub(r'^因此', '', text)
    text = re.sub(r'^因为', '', text)
    text = re.sub(r'^其实', '', text)
    text = re.sub(r'^然后', '', text)
    text = re.sub(r'^接下来', '', text)
    
    patterns = [
        (r'我们公司', '公司'),
        (r'咱们公司', '公司'),
        (r'我个人认为', '有观点认为'),
        (r'我个人觉得', '有观点认为'),
        (r'我个人建议', '建议'),
        (r'我认为', '有观点认为'),
        (r'我觉得', '有观点认为'),
        (r'我看', '有观点认为'),
        (r'我想', '有观点提出'),
        (r'我建议', '建议'),
        (r'我提议', '提议'),
        (r'我同意', '同意'),
        (r'我不同意', '不同意'),
        (r'我反对', '反对'),
        (r'我支持', '支持'),
        (r'我强调', '强调'),
        (r'我们认为', '会议认为'),
        (r'我们觉得', '会议认为'),
        (r'我们建议', '会议建议'),
        (r'我们提议', '会议提议'),
        (r'我们决定', '会议决定'),
        (r'我们同意', '会议同意'),
        (r'我们反对', '会议反对'),
        (r'我们强调', '会议强调'),
        (r'我们需要', '需要'),
        (r'我们必须', '必须'),
        (r'我们应当', '应当'),
        (r'我们应该', '应该'),
        (r'我们禁止', '禁止'),
        (r'我们要求', '要求'),
        (r'我们明确', '明确'),
        (r'我说', '发言称'),
        (r'他说', '发言称'),
        (r'她说', '发言称'),
        (r'大家说', '参会人员表示'),
        (r'大家认为', '参会人员认为'),
        (r'大家觉得', '参会人员认为'),
        (r'各位同事', '参会人员'),
        (r'各位领导', '参会人员'),
        (r'同事们', '参会人员'),
        (r'大家好', ''),
        (r'大家', '参会人员'),
        (r'咱们', '参会人员'),
        (r'我们', '会议'),
        (r'我', '发言者'),
    ]
    
    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text)
    
    text = re.sub(r'发言者们', '参会人员', text)
    text = re.sub(r'参会人员参会人员', '参会人员', text)
    text = re.sub(r'发言者发言者', '发言者', text)
    text = re.sub(r'会议会议', '会议', text)
    text = re.sub(r'公司公司', '公司', text)
    text = re.sub(r'[“”"\']', '', text)
    
    text = re.sub(r'([\u4e00-\u9fa5])\1{2,}', r'\1', text)
    text = re.sub(r'([\u4e00-\u9fa5]{2,})\1{1,}', r'\1', text)
    
    text = re.sub(r'，而且', '，', text)
    text = re.sub(r'，不过', '，但', text)
    text = re.sub(r'，那么', '，', text)
    text = re.sub(r'，所以', '，因此', text)
    
    return text.strip()


def _is_quality_sentence(sentence, keywords_set):
    """
    判断句子是否适合作为摘要内容
    过滤掉：纯介绍、纯举例、纯数字描述、寒暄等
    """
    if not sentence or len(sentence) < 15:
        return False
    
    text = sentence
    
    empty_patterns = [
        r'^好的$', r'^谢谢$', r'^大家好$', r'^明白了$', r'^知道了$',
        r'^没问题$', r'^可以$', r'^行$', r'^对$', r'^是$',
        r'还有什么问题', r'没什么问题', r'没了', r'没有了',
        r'散会', r'结束',
    ]
    for p in empty_patterns:
        if re.search(p, text):
            return False
    
    if re.search(r'(比如|例如|举个例子|比方说|就像)', text):
        if len(text) < 30:
            return False
    
    num_chars = len(re.findall(r'[0-9.%]', text))
    if num_chars > 0 and num_chars / len(text) > 0.15 and len(text) < 40:
        return False
    
    if re.search(r'(我叫|我是|他叫|他是|她叫|她是|这位是)', text) and len(text) < 30:
        return False
    
    kw_hits = sum(1 for kw in keywords_set if kw in text)
    if kw_hits == 0 and len(text) < 25:
        return False
    
    return True


def _extract_core_points(sentences, keywords_set, max_points=3):
    """
    从句子中提取核心讨论点
    优先选择：规范性表述（禁止/必须/要求等）、讨论性词汇、关键词命中多的句子
    过滤掉：纯产品介绍、被否定的陈述、举例说明等
    """
    discussion_patterns = [
        r'讨论', r'认为', r'强调', r'指出', r'提出', r'表示', r'觉得',
        r'分析', r'探讨', r'研究', r'关注', r'重视', r'考虑',
        r'问题', r'难点', r'重点', r'核心', r'关键',
        r'会议认为', r'有观点认为', r'发言称', r'参会人员',
    ]
    
    normative_patterns = [
        r'禁止', r'不得', r'必须', r'应当', r'要求', r'严禁',
        r'需要', r'明确', r'强调', r'决定', r'同意', r'反对',
    ]
    
    negative_statement_patterns = [
        r'承诺.*保本', r'承诺.*收益', r'保证.*收益', r'保证.*保本',
        r'绝对安全', r'无风险', r'零风险', r'稳赚不赔',
        r'放心推荐', r'肯定赚', r'一定赚',
    ]
    
    scored = []
    seen = set()
    
    for sent in sentences:
        narrative = _convert_to_narrative(sent)
        if not narrative or len(narrative) < 15:
            continue
        
        key = narrative[:20]
        if key in seen:
            continue
        seen.add(key)
        
        if not _is_quality_sentence(narrative, keywords_set):
            continue
        
        is_negative_statement = False
        for p in negative_statement_patterns:
            if re.search(p, narrative):
                is_negative_statement = True
                break
        
        score = 0
        
        is_normative = False
        for p in normative_patterns:
            if re.search(p, narrative):
                is_normative = True
                score += 5
                break
        
        if not is_normative and is_negative_statement:
            score -= 10
        
        for p in discussion_patterns:
            if re.search(p, narrative):
                score += 2
                break
        
        kw_hits = sum(1 for kw in keywords_set if kw in narrative)
        score += kw_hits
        
        if 20 <= len(narrative) <= 60:
            score += 1
        
        scored.append((narrative, score))
    
    scored.sort(key=lambda x: x[1], reverse=True)
    
    result = []
    for narrative, score in scored:
        if score <= 0:
            continue
        is_dup = False
        for existing in result:
            overlap = len(set(narrative) & set(existing)) / max(len(set(narrative)), 1)
            if overlap > 0.6:
                is_dup = True
                break
            if narrative[:15] in existing or existing[:15] in narrative:
                is_dup = True
                break
        if not is_dup:
            result.append(narrative)
            if len(result) >= max_points:
                break
    
    return result


def generate_summary(text, max_length=500, language='zh'):
    """
    生成会议摘要：使用TextRank提取关键句，转换为第三人称叙述
    预处理：智能分句，确保句子边界正确
    """
    print(f'=== generate_summary called === text length: {len(text)}')
    if not text or len(text.strip()) < 20:
        return "内容过短，无法生成摘要"

    segmented_text = _smart_segment(text)
    print(f'=== segmented sentences: {len(segmented_text)} ===')
    
    key_sents = extract_key_sentences_from_list(segmented_text, top_n=10)
    
    if not key_sents:
        return text[:max_length]

    keywords = extract_keywords(text, top_n=8, language=language)
    
    processed_sents = []
    seen = set()
    
    for sent in key_sents:
        narrative = _convert_to_narrative(sent)
        
        if not narrative or len(narrative) < 12:
            continue
        
        is_greeting = False
        greeting_patterns = [
            r'好的$', r'谢谢$', r'大家好', r'散会', r'明白了', r'知道了',
            r'还有什么问题', r'没了', r'没有了', r'没问题', r'可以', r'行',
            r'请.*介绍', r'请.*来讲', r'今天就到这里',
        ]
        for gp in greeting_patterns:
            if re.search(gp, narrative):
                is_greeting = True
                break
        if is_greeting:
            continue
        
        if narrative.endswith('？') or narrative.endswith('?'):
            continue
        
        question_words = ['怎么', '什么', '谁', '哪', '几', '多少', '为什么', '是否', '吗']
        has_question = False
        for qw in question_words:
            if qw in narrative:
                has_question = True
                break
        if has_question:
            continue
        
        negative_patterns = [
            r'承诺.*保本', r'承诺.*收益', r'保证.*收益', r'绝对安全',
            r'无风险', r'零风险', r'稳赚不赔', r'放心推荐',
        ]
        is_negative = False
        for np in negative_patterns:
            if re.search(np, narrative):
                is_negative = True
                break
        if is_negative:
            continue
        
        key = narrative[:15]
        if key in seen:
            continue
        
        is_dup = False
        for existing in processed_sents:
            common_chars = len(set(narrative) & set(existing))
            total_chars = len(set(narrative) | set(existing))
            if total_chars > 0 and common_chars / total_chars > 0.5:
                is_dup = True
                break
            if narrative[:10] in existing or existing[:10] in narrative:
                is_dup = True
                break
        if is_dup:
            continue
        
        seen.add(key)
        processed_sents.append(narrative)
    
    if not processed_sents:
        return text[:max_length]
    
    topics = analyze_topic(text, language=language)
    
    summary_parts = []
    
    if topics and topics[0]['score'] > 0:
        top_topics = [t['topic'] for t in topics[:2] if t['score'] > 0.3]
        if not top_topics:
            top_topics = [t['topic'] for t in topics[:1]]
        
        if top_topics:
            topic_str = '与'.join(top_topics)
            if keywords:
                kw_list = [kw['word'] for kw in keywords[:6] if kw['word'] != '还有' and kw['word'] not in top_topics]
                if kw_list:
                    kw_str = '、'.join(kw_list[:4])
                    opening = f"本次会议围绕{topic_str}展开讨论，涉及{kw_str}等方面"
                else:
                    opening = f"本次会议围绕{topic_str}展开讨论"
            else:
                opening = f"本次会议围绕{topic_str}展开讨论"
            summary_parts.append(opening)
    
    for sent in processed_sents[:4]:
        sent = re.sub(r'[这那这这个个的]$', '', sent)
        sent = re.sub(r'[，,、；;]$', '', sent)
        sent = re.sub(r'不能口头说一下就完了[这那]$', '', sent)
        sent = re.sub(r'不能口头说一下就完了$', '', sent)
        sent = re.sub(r'不能含糊其辞$', '', sent)
        sent = re.sub(r'任何人都不允许越过$', '', sent)
        sent = re.sub(r'不能使用任何可能误导客户的表述$', '', sent)
        if len(sent) > 8:
            summary_parts.append(sent)
    
    summary = '。'.join(summary_parts)
    if not summary.endswith('。'):
        summary += '。'
    
    from modules.whisper_utils import fix_traditional_chinese
    summary = fix_traditional_chinese(summary)
    
    summary = re.sub(r'[，,]{2,}', '，', summary)
    summary = re.sub(r'[。.]{2,}', '。', summary)
    summary = re.sub(r'([\u4e00-\u9fa5])\1{2,}', r'\1', summary)
    summary = re.sub(r'([\u4e00-\u9fa5]{2,})\1{1,}', r'\1', summary)
    
    summary = re.sub(r'讨论一下', '讨论', summary)
    summary = re.sub(r'对了[，,]*', '', summary)
    summary = re.sub(r'产品的产品', '产品', summary)
    summary = re.sub(r'风险等级是是', '风险等级是', summary)
    summary = re.sub(r'会议会议', '会议', summary)
    summary = re.sub(r'销售人员销售人员', '销售人员', summary)
    
    if len(summary) > max_length:
        summary = summary[:max_length - 3] + "..."
    
    print(f'=== generate_summary result ===: {summary[:200]}...')
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
        import threading
        result_container = []
        
        def load_and_analyze():
            try:
                from sentence_transformers import util
                model = get_sentence_transformer_model(language)
                if model is None:
                    raise Exception("sentence_transformer model not available")

                text_segment = text[:2000] if len(text) > 2000 else text
                text_embedding = model.encode(text_segment, convert_to_tensor=True)
                topic_embeddings = model.encode(topics, convert_to_tensor=True)

                cosine_scores = util.cos_sim(text_embedding, topic_embeddings)
                topic_scores = list(zip(topics, cosine_scores[0].tolist()))

                enhanced_scores = []
                for topic, sim_score in topic_scores:
                    kw_list = topic_keywords.get(topic, [])
                    kw_hits = sum(1 for kw in kw_list if kw in text)
                    kw_score = min(kw_hits / max(len(kw_list), 1), 1.0)
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
                result_container.append(result)
            except Exception as e:
                print(f"语义分析失败: {e}")

        thread = threading.Thread(target=load_and_analyze)
        thread.daemon = True
        thread.start()
        thread.join(timeout=10)
        
        if result_container:
            return result_container[0]
        else:
            print("语义分析超时，使用关键词匹配")
            return keyword_based_topic(text, topics, topic_keywords)

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
    使用snownlp开源库进行中文情感分析
    """
    if language == 'zh':
        try:
            from snownlp import SnowNLP
            
            s = SnowNLP(text)
            score = s.sentiments
            
            if score > 0.6:
                sentiment = 'positive'
            elif score < 0.4:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'

            return {
                'sentiment': sentiment,
                'score': round(score, 4),
                'positive': 0,
                'negative': 0,
                'positive_score': round(score, 2),
                'negative_score': round(1 - score, 2)
            }
        except Exception as e:
            print(f"snownlp sentiment analysis failed: {e}, using fallback")
            return _fallback_sentiment_analysis(text)
    
    return _fallback_sentiment_analysis(text)


def _fallback_sentiment_analysis(text):
    """
    备用情绪分析方案（基于词典）
    """
    positive_words = {
        '好': 1, '优秀': 2, '完成': 1, '成功': 2, '进步': 1, '提升': 1, '积极': 1,
        '满意': 2, '支持': 1, '同意': 1, '通过': 1, '批准': 1, '赞': 1, '赞成': 1,
        '顺利': 1, '突破': 2, '创新': 1, '高效': 1, '优化': 1, '改善': 1,
        '增长': 1, '盈利': 1, '超越': 1, '达成': 1, '实现': 1, '解决': 1,
        '加强': 1, '推进': 1, '落实': 1, '成效': 1, '成果': 1, '收获': 1,
        '确认': 1, '明确': 1, '完善': 1, '规范': 1, '保障': 1, '确保': 1,
    }
    
    negative_words = {
        '消极': 1, '反对': 1, '抵制': 2, '抱怨': 1, '不满': 1, '拒绝': 1,
        '不行': 1, '不可能': 1, '做不到': 1, '失败': 2, '延误': 1, '延迟': 1,
        '隐患': 1, '困难': 1, '障碍': 1, '阻碍': 1,
        '下降': 1, '减少': 1, '亏损': 2, '流失': 1, '违规': 2, '违纪': 2,
        '偏离': 1, '不当': 1, '不足': 1, '缺乏': 1, '缺失': 1, '遗漏': 1,
        '忽悠': 2, '骗': 2, '蒙': 2, '糊弄': 2, '坑': 2,
        '麻烦': 1, '混乱': 1, '糟': 1,
    }
    
    context_neutral_words = {
        '风险', '问题', '讨论', '分析', '评估', '检查', '审核',
        '要求', '规定', '制度', '政策', '规范', '标准',
        '禁止', '必须', '应该', '需要', '建议', '决定',
        '培训', '学习', '会议', '沟通', '交流', '意见',
    }
    
    negation_words = {'不', '没', '无', '非', '未', '别', '莫', '勿', '否'}
    degree_words = {'非常': 1.5, '十分': 1.5, '特别': 1.5, '极其': 2, '格外': 1.5,
                   '比较': 1.2, '相当': 1.3, '很': 1.2, '太': 1.5, '极为': 2}

    words = jieba.lcut(text)
    
    positive_score = 0
    negative_score = 0

    i = 0
    while i < len(words):
        w = words[i]
        
        if w in context_neutral_words:
            i += 1
            continue
        
        degree = 1.0
        if w in degree_words:
            degree = degree_words[w]
            i += 1
            if i >= len(words):
                break
            w = words[i]
        
        negated = False
        if w in negation_words:
            negated = True
            i += 1
            if i >= len(words):
                break
            w = words[i]
        
        if w in positive_words:
            score = positive_words[w] * degree
            if negated:
                negative_score += score
            else:
                positive_score += score
        elif w in negative_words:
            score = negative_words[w] * degree
            if negated:
                positive_score += score
            else:
                negative_score += score
        
        i += 1

    total = positive_score + negative_score
    if total == 0:
        return {'sentiment': 'neutral', 'score': 0.5, 'positive': 0, 'negative': 0, 'positive_score': 0, 'negative_score': 0}

    score = positive_score / total
    if score > 0.55:
        sentiment = 'positive'
    elif score < 0.45:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'

    return {
        'sentiment': sentiment,
        'score': round(score, 4),
        'positive': 0,
        'negative': 0,
        'positive_score': round(positive_score, 2),
        'negative_score': round(negative_score, 2)
    }
