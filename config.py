import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'meeting-analysis-secret-key'
    
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac', 'm4a'}
    
    WHISPER_MODEL = os.environ.get('WHISPER_MODEL') or 'medium'
    WHISPER_MODEL_REALTIME = os.environ.get('WHISPER_MODEL_REALTIME') or 'small'
    
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///' + os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'meeting_analysis.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    MAX_CONTENT_LENGTH = 256 * 1024 * 1024
    
    HF_TOKEN = os.environ.get('HF_TOKEN') or ''
    
    SCORE_WEIGHTS = {
        'semantic_similarity': 40,
        'point_coverage': 30,
        'risk_detection': 20,
        'keyword_matching': 10
    }
    
    TOPICS = ["工作汇报", "项目讨论", "问题解决", "决策制定", "进度跟进", "计划安排", "意见交流", "培训学习"]
    
    RISK_KEYWORDS = ["消极", "反对", "抵制", "抱怨", "不满", "拒绝", "不行", "不可能", "做不到"]