from datetime import datetime
from database import db

class Meeting(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    date = db.Column(db.DateTime, nullable=False, default=datetime.now)
    duration = db.Column(db.Integer)
    status = db.Column(db.String(20), default='processing')
    audio_path = db.Column(db.String(500))
    total_score = db.Column(db.Float)
    score_level = db.Column(db.String(20))
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, onupdate=datetime.now)
    
    transcriptions = db.relationship('Transcription', backref='meeting', lazy=True)
    compliance_report = db.relationship('ComplianceReport', backref='meeting', uselist=False)
    
    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'date': self.date.isoformat(),
            'duration': int(self.duration) if self.duration else 0,
            'status': self.status,
            'total_score': float(self.total_score) if self.total_score else None,
            'score_level': self.score_level,
            'created_at': self.created_at.isoformat()
        }

class Transcription(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    meeting_id = db.Column(db.Integer, db.ForeignKey('meeting.id'), nullable=False)
    speaker = db.Column(db.String(50))
    text = db.Column(db.Text, nullable=False)
    start_time = db.Column(db.Float)
    end_time = db.Column(db.Float)
    confidence = db.Column(db.Float)
    language = db.Column(db.String(10))
    
    def to_dict(self):
        return {
            'id': self.id,
            'speaker': self.speaker,
            'text': self.text,
            'start_time': float(self.start_time) if self.start_time else 0,
            'end_time': float(self.end_time) if self.end_time else 0,
            'confidence': float(self.confidence) if self.confidence else 0,
            'language': self.language
        }

class KnowledgeBase(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    content = db.Column(db.Text, nullable=False)
    item_type = db.Column(db.String(50), nullable=False)
    keywords = db.Column(db.Text)
    required_points = db.Column(db.Text)
    status = db.Column(db.String(20), default='active')
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, onupdate=datetime.now)
    
    def to_dict(self):
        import json
        return {
            'id': self.id,
            'title': self.title,
            'content': self.content,
            'item_type': self.item_type,
            'keywords': json.loads(self.keywords) if self.keywords else [],
            'required_points': json.loads(self.required_points) if self.required_points else [],
            'status': self.status,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class ComplianceReport(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    meeting_id = db.Column(db.Integer, db.ForeignKey('meeting.id'), nullable=False)
    total_score = db.Column(db.Float, nullable=False)
    score_level = db.Column(db.String(20))
    detailed_scores = db.Column(db.Text)
    missing_points = db.Column(db.Text)
    risk_keywords = db.Column(db.Text)
    risk_time_markers = db.Column(db.Text)  # 新增：风险内容时间节点
    point_time_markers = db.Column(db.Text)  # 新增：要点时间节点
    matched_keywords = db.Column(db.Text)
    suggestions = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.now)
    
    def to_dict(self):
        import json
        return {
            'id': self.id,
            'meeting_id': self.meeting_id,
            'total_score': float(self.total_score) if self.total_score else 0,
            'score_level': self.score_level,
            'detailed_scores': json.loads(self.detailed_scores) if self.detailed_scores else {},
            'missing_points': json.loads(self.missing_points) if self.missing_points else [],
            'risk_keywords': json.loads(self.risk_keywords) if self.risk_keywords else [],
            'risk_time_markers': json.loads(self.risk_time_markers) if self.risk_time_markers else [],
            'point_time_markers': json.loads(self.point_time_markers) if self.point_time_markers else [],
            'matched_keywords': json.loads(self.matched_keywords) if self.matched_keywords else [],
            'suggestions': json.loads(self.suggestions) if self.suggestions else [],
            'created_at': self.created_at.isoformat()
        }

class ScoreWeight(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    weight_name = db.Column(db.String(50), nullable=False)
    weight_value = db.Column(db.Float, nullable=False)
    description = db.Column(db.String(255))
    updated_at = db.Column(db.DateTime, onupdate=datetime.now)
    
    def to_dict(self):
        return {
            'id': self.id,
            'weight_name': self.weight_name,
            'weight_value': self.weight_value,
            'description': self.description
        }