import os
import re
import chardet
from docx import Document
import fitz


def extract_text_from_file(file_path):
    """
    从文件中提取文本内容
    支持：.txt、.pdf、.docx、.doc
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.txt':
        return _extract_txt(file_path)
    elif ext == '.pdf':
        return _extract_pdf(file_path)
    elif ext == '.docx':
        return _extract_docx(file_path)
    elif ext == '.doc':
        return _extract_doc(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {ext}")


def _extract_txt(file_path):
    """提取TXT文件内容，自动检测编码"""
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding'] or 'utf-8'
    
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read()
    except:
        with open(file_path, 'r', encoding='gbk') as f:
            return f.read()


def _extract_pdf(file_path):
    """提取PDF文件内容"""
    doc = fitz.open(file_path)
    text = ''
    for page in doc:
        text += page.get_text()
    return text


def _extract_docx(file_path):
    """提取DOCX文件内容"""
    doc = Document(file_path)
    text = '\n'.join([para.text for para in doc.paragraphs])
    return text


def _extract_doc(file_path):
    """提取DOC文件内容（使用antiword或textract）"""
    try:
        from textract import process
        return process(file_path).decode('utf-8')
    except:
        try:
            import subprocess
            result = subprocess.run(['antiword', file_path], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout
        except:
            pass
    return ''


def extract_title(content, filename):
    """从内容中提取标题（优先首行，其次文件名）"""
    first_line = content.strip().split('\n')[0] if content else ''
    
    if first_line and len(first_line) <= 50:
        clean_title = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9_-]', '', first_line)
        if len(clean_title) >= 2:
            return clean_title
    
    name_without_ext = os.path.splitext(filename)[0]
    return name_without_ext


def extract_keywords(content, top_n=5):
    """从内容中提取关键词（基于词频）"""
    import jieba
    from collections import Counter
    
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', content)
    words = jieba.lcut(text)
    
    stopwords = {
        '的', '是', '在', '和', '了', '有', '我', '他', '她', '它',
        '这', '那', '什么', '怎么', '为什么', '因为', '所以', '但是',
        '一个', '一些', '这个', '那个', '这些', '那些', '我们', '你们', '他们',
        '可以', '可能', '应该', '需要', '会', '不会', '能', '不能',
        '说', '话', '听', '看', '想', '做', '知道', '觉得', '认为',
        '今天', '明天', '昨天', '现在', '刚才', '然后', '最后', '首先',
        '会议', '内容', '方面', '问题', '事情', '东西', '地方', '时间', '人员',
        '情况', '状态', '方式', '方法', '过程', '结果', '原因', '目的',
        '大家', '各位', '领导', '同事', '进行', '通过', '关于', '对于', '根据',
        '按照', '作为', '为了', '一般', '有点', '一下', '一些', '一点', '一直',
        '一定', '一样', '没有', '没什么',
    }
    
    filtered = [w for w in words if len(w) >= 2 and w not in stopwords]
    counter = Counter(filtered)
    
    return [kw for kw, _ in counter.most_common(top_n)]


def extract_required_points(content, max_points=5):
    """从内容中提取必传要点（基于标题和关键句子）"""
    points = []
    
    lines = content.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if len(line) > 5 and len(line) <= 80:
            if re.match(r'^[一二三四五六七八九十\d]+[、.．)）]', line):
                clean = re.sub(r'^[一二三四五六七八九十\d]+[、.．)）]\s*', '', line)
                if clean and clean not in points:
                    points.append(clean)
            elif re.match(r'^[-*●•·]', line):
                clean = re.sub(r'^[-*●•·]\s*', '', line)
                if clean and clean not in points:
                    points.append(clean)
            elif line.endswith('。') or line.endswith('；'):
                if line not in points:
                    points.append(line[:-1])
        
        if len(points) >= max_points:
            break
    
    return points


def parse_policy_file(file_path):
    """
    解析政策文件，返回结构化数据
    返回：{'title', 'content', 'keywords', 'required_points'}
    """
    content = extract_text_from_file(file_path)
    
    if len(content) > 5000:
        content = content[:5000]
    
    title = extract_title(content, os.path.basename(file_path))
    keywords = extract_keywords(content, top_n=5)
    required_points = extract_required_points(content, max_points=5)
    
    return {
        'title': title,
        'content': content,
        'keywords': keywords,
        'required_points': required_points
    }
