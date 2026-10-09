import os
import secrets


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _csv_env(name):
    return [value.strip() for value in os.environ.get(name, '').split(',') if value.strip()]

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or secrets.token_urlsafe(32)
    
    TEMPLATES_AUTO_RELOAD = True
    SEND_FILE_MAX_AGE_DEFAULT = 0
    
    DATA_FOLDER = os.path.join(BASE_DIR, 'data')
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac', 'm4a', 'webm', 'mp4', 'aac'}
    
    WHISPER_MODEL = os.environ.get('WHISPER_MODEL') or 'medium'
    VOICE_API_URL = os.environ.get('VOICE_API_URL', '').rstrip('/')
    VOICE_API_TIMEOUT = int(os.environ.get('VOICE_API_TIMEOUT', '1800'))
    # 是否使用大模型生成摘要（关闭后摘要退回 TextRank 抽取式，速度更快、占用更低）
    ENABLE_TEXT_MODELS = os.environ.get('ENABLE_TEXT_MODELS', 'true').lower() in {'1', 'true', 'yes'}

    # ASR 热词（行业词汇增强）：空格分隔，会与知识库关键词合并后交给识别引擎，
    # 同时注册进 jieba 词典（避免「中国人寿」被切成「中国」+「人寿」）。
    # 对应需求「利用行业词汇库建立热词增强的语音识别解码网络」。
    # 可用环境变量覆盖或清空。
    ASR_HOTWORDS = os.environ.get(
        'ASR_HOTWORDS',
        # 合规要点类
        '风险测评 投资者适当性 风险告知书 风险揭示 录音录像 持证上岗 书面确认 '
        '风险等级 承受能力 合规销售 销售话术 整改报告 合规培训 '
        # 禁止表述类（识别出来才能被风险检测命中）
        '保本保收益 零风险 稳赚不赔 绝对安全 保证收益 高收益无风险 '
        # 保险业务类（本项目知识库即金融/保险销售场景）
        '中国人寿 中国人保 中国平安 太平洋保险 新华保险 泰康人寿 '
        '医疗险 重疾险 意外险 寿险 保单 保单体检 理赔 免赔额 保额 保费 '
        '保险责任 免责条款 投保人 被保险人 受益人 健康告知 核保 住院报销',
    )
    
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///' + os.path.join(DATA_FOLDER, 'meeting_analysis.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    MAX_CONTENT_LENGTH = 256 * 1024 * 1024
    MAX_PAGE_SIZE = 100
    
    HF_TOKEN = os.environ.get('HF_TOKEN') or ''

    HOST = os.environ.get('HOST', '0.0.0.0')
    PORT = int(os.environ.get('PORT', '5001'))
    SSL_CERT_FILE = os.environ.get('SSL_CERT_FILE')
    SSL_KEY_FILE = os.environ.get('SSL_KEY_FILE')
    CORS_ORIGINS = _csv_env('CORS_ORIGINS')
    TRUST_PROXY = os.environ.get('TRUST_PROXY', '').lower() in {'1', 'true', 'yes'}
    PRELOAD_MODELS = os.environ.get('PRELOAD_MODELS', 'true').lower() in {'1', 'true', 'yes'}
    
    SCORE_WEIGHTS = {
        'semantic_similarity': 40,
        'point_coverage': 30,
        'risk_detection': 20,
        'keyword_matching': 10
    }
    
    TOPICS = [
        "工作汇报", "项目讨论", "问题解决", "决策制定",
        "进度跟进", "计划安排", "意见交流", "培训学习",
        "风险讨论", "合规审查", "财务预算", "人事安排",
        "客户服务", "产品设计", "技术方案", "市场营销",
        # 业务沟通场景（销售通话、客户拜访等）
        "客户沟通", "产品介绍", "理赔服务", "销售推广",
    ]
    
    RISK_KEYWORDS = ["消极", "反对", "抵制", "抱怨", "不满", "拒绝", "不行", "不可能", "做不到"]
