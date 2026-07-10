import os
import json
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO
import whisper
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__)
app.config.from_object('config.Config')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

from database import db

db.init_app(app)

from models import Meeting, Transcription, KnowledgeBase, ComplianceReport, ScoreWeight
from modules.audio_preprocessor import preprocess_audio, format_time, detect_speech_segments, get_audio_quality_report
from modules.speaker_diarization import speaker_diarization_simple
from modules.text_analyzer import extract_keywords, analyze_topic, generate_summary, analyze_sentiment
from modules.compliance_checker import calculate_compliance_score, generate_compliance_report, get_score_level
from modules.realtime_transcriber import register_socketio_events
from modules.meeting_detector import MeetingDetector, count_participants, analyze_participation_distribution
from modules.report_generator import generate_meeting_summary_report, generate_compliance_trend_report, generate_report_html

whisper_model = None

def init_whisper_model():
    global whisper_model
    if whisper_model is None:
        print('Initializing Whisper model...')
        try:
            whisper_model = whisper.load_model(app.config['WHISPER_MODEL'])
            print(f'Whisper {app.config["WHISPER_MODEL"]} model loaded successfully')
        except Exception as e:
            print(f'Failed to load {app.config["WHISPER_MODEL"]}: {e}')
            print('Loading small model as fallback...')
            whisper_model = whisper.load_model('small')
            print('Whisper small model loaded')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'message': 'API service is running'})

@app.route('/api/meetings', methods=['GET'])
def get_meetings():
    page = request.args.get('page', 1, type=int)
    page_size = request.args.get('page_size', 10, type=int)
    status = request.args.get('status')
    
    query = Meeting.query
    if status:
        query = query.filter_by(status=status)
    
    meetings = query.order_by(Meeting.created_at.desc()).paginate(page=page, per_page=page_size)
    
    return jsonify({
        'code': 200,
        'data': [m.to_dict() for m in meetings.items],
        'total': meetings.total,
        'page': page,
        'page_size': page_size
    })

@app.route('/api/meetings/<int:meeting_id>')
def get_meeting(meeting_id):
    meeting = Meeting.query.get_or_404(meeting_id)
    transcriptions = Transcription.query.filter_by(meeting_id=meeting_id).all()
    compliance_report = ComplianceReport.query.filter_by(meeting_id=meeting_id).first()
    
    data = meeting.to_dict()
    data['transcriptions'] = [t.to_dict() for t in transcriptions]
    data['compliance_report'] = compliance_report.to_dict() if compliance_report else None
    
    return jsonify({'code': 200, 'data': data})

@app.route('/api/meetings', methods=['POST'])
def create_meeting():
    if 'audio_file' not in request.files:
        return jsonify({'code': 400, 'message': 'No audio file provided'}), 400
    
    audio_file = request.files['audio_file']
    if audio_file.filename == '':
        return jsonify({'code': 400, 'message': 'No audio file selected'}), 400
    
    if not allowed_file(audio_file.filename):
        return jsonify({'code': 400, 'message': 'File type not allowed'}), 400
    
    meeting_title = request.form.get('meeting_title', '未命名会议')
    enable_diarization = request.form.get('enable_diarization', 'true').lower() == 'true'
    enable_compliance = request.form.get('enable_compliance', 'true').lower() == 'true'
    
    file_ext = audio_file.filename.rsplit('.', 1)[1].lower()
    file_id = str(uuid.uuid4())
    original_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{file_id}_original.{file_ext}')
    processed_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{file_id}_processed.wav')
    
    audio_file.save(original_path)
    
    try:
        preprocess_audio(original_path, processed_path)
    except Exception as e:
        print(f'Audio preprocessing failed: {e}')
        processed_path = original_path
    
    meeting = Meeting(
        title=meeting_title,
        date=datetime.now(),
        status='processing',
        audio_path=processed_path
    )
    db.session.add(meeting)
    db.session.commit()
    
    init_whisper_model()
    
    try:
        result = whisper_model.transcribe(
            processed_path,
            language='zh',
            initial_prompt='以下是简体中文的语音转写内容，请使用简体中文输出。'
        )
        full_text = result['text']

        if enable_diarization:
            try:
                speaker_segments = speaker_diarization_simple(processed_path)
            except Exception as e:
                print(f'Speaker diarization failed: {e}')
                speaker_segments = []
        else:
            speaker_segments = []

        for segment in result['segments']:
            speaker = 'SPEAKER_00'
            if speaker_segments:
                for ss in speaker_segments:
                    if ss['start'] <= segment['start'] <= ss['end']:
                        speaker = ss['speaker']
                        break

            transcription = Transcription(
                meeting_id=meeting.id,
                speaker=speaker,
                text=segment['text'],
                start_time=segment['start'],
                end_time=segment['end'],
                confidence=segment.get('confidence', 0.0),
                language=result['language']
            )
            db.session.add(transcription)

        # 关键词提取
        keywords = extract_keywords(full_text, top_n=10)
        # 主题分析
        topics = analyze_topic(full_text)
        # 会议摘要生成
        summary = generate_summary(full_text, max_length=300)
        # 情绪分析
        sentiment = analyze_sentiment(full_text)

        # 音频质量报告
        audio_quality = None
        try:
            audio_quality = get_audio_quality_report(original_path, processed_path)
        except Exception as e:
            print(f'Audio quality report failed: {e}')

        meeting.duration = int(result['segments'][-1]['end']) if result['segments'] else 0

        compliance_result = None
        if enable_compliance:
            knowledge_items = KnowledgeBase.query.filter_by(status='active').all()
            # 获取转写段落用于时间节点标记
            transcription_segments = [t.to_dict() for t in Transcription.query.filter_by(meeting_id=meeting.id).all()]
            compliance_result = calculate_compliance_score(full_text, knowledge_items, transcription_segments=transcription_segments)

            report = ComplianceReport(
                meeting_id=meeting.id,
                total_score=compliance_result['total_score'],
                score_level=get_score_level(compliance_result['total_score']),
                detailed_scores=json.dumps(compliance_result['components']),
                missing_points=json.dumps(compliance_result['missing_points']),
                risk_keywords=json.dumps(compliance_result['risk_keywords_found']),
                risk_time_markers=json.dumps(compliance_result.get('risk_time_markers', [])),
                point_time_markers=json.dumps(compliance_result.get('point_time_markers', [])),
                matched_keywords=json.dumps(compliance_result['matched_keywords']),
                suggestions=json.dumps(compliance_result['suggestions'])
            )
            db.session.add(report)

            meeting.total_score = compliance_result['total_score']
            meeting.score_level = report.score_level

        meeting.status = 'finished'
        db.session.commit()

        response_data = {
            'meeting_id': meeting.id,
            'title': meeting.title,
            'transcriptions': [t.to_dict() for t in Transcription.query.filter_by(meeting_id=meeting.id).all()],
            'keywords': keywords,
            'topics': topics,
            'summary': summary,
            'sentiment': sentiment,
            'audio_quality': audio_quality,
            'duration': meeting.duration,
            'language': result['language']
        }

        if compliance_result:
            response_data['compliance'] = {
                'total_score': compliance_result['total_score'],
                'score_level': get_score_level(compliance_result['total_score']),
                'components': compliance_result['components'],
                'covered_points': compliance_result['covered_points'],
                'point_time_markers': compliance_result.get('point_time_markers', []),
                'missing_points': compliance_result['missing_points'],
                'risk_keywords': compliance_result['risk_keywords_found'],
                'risk_time_markers': compliance_result.get('risk_time_markers', []),
                'matched_keywords': compliance_result['matched_keywords'],
                'suggestions': compliance_result['suggestions']
            }

        return jsonify({'code': 200, 'message': '分析完成', 'data': response_data})

    except Exception as e:
        print(f'Analysis failed: {e}')
        import traceback
        traceback.print_exc()
        meeting.status = 'failed'
        db.session.commit()
        return jsonify({'code': 500, 'message': f'分析失败: {str(e)}'}), 500

@app.route('/api/meetings/<int:meeting_id>', methods=['DELETE'])
def delete_meeting(meeting_id):
    meeting = Meeting.query.get_or_404(meeting_id)
    
    Transcription.query.filter_by(meeting_id=meeting_id).delete()
    ComplianceReport.query.filter_by(meeting_id=meeting_id).delete()
    
    if meeting.audio_path and os.path.exists(meeting.audio_path):
        os.remove(meeting.audio_path)
    
    db.session.delete(meeting)
    db.session.commit()
    
    return jsonify({'code': 200, 'message': '删除成功'})

@app.route('/api/knowledge-base', methods=['GET'])
def get_knowledge_base():
    page = request.args.get('page', 1, type=int)
    page_size = request.args.get('page_size', 10, type=int)
    item_type = request.args.get('item_type')
    
    query = KnowledgeBase.query
    if item_type:
        query = query.filter_by(item_type=item_type)
    
    items = query.order_by(KnowledgeBase.created_at.desc()).paginate(page=page, per_page=page_size)
    
    return jsonify({
        'code': 200,
        'data': [item.to_dict() for item in items.items],
        'total': items.total
    })

@app.route('/api/knowledge-base/<int:item_id>')
def get_knowledge_item(item_id):
    item = KnowledgeBase.query.get_or_404(item_id)
    return jsonify({'code': 200, 'data': item.to_dict()})

@app.route('/api/knowledge-base', methods=['POST'])
def create_knowledge_item():
    data = request.get_json()
    
    item = KnowledgeBase(
        title=data.get('title'),
        content=data.get('content'),
        item_type=data.get('item_type', 'policy'),
        keywords=json.dumps(data.get('keywords', [])),
        required_points=json.dumps(data.get('required_points', []))
    )
    db.session.add(item)
    db.session.commit()
    
    return jsonify({'code': 200, 'message': '创建成功', 'data': item.to_dict()})

@app.route('/api/knowledge-base/<int:item_id>', methods=['PUT'])
def update_knowledge_item(item_id):
    item = KnowledgeBase.query.get_or_404(item_id)
    data = request.get_json()
    
    item.title = data.get('title', item.title)
    item.content = data.get('content', item.content)
    item.item_type = data.get('item_type', item.item_type)
    item.keywords = json.dumps(data.get('keywords', [])) if 'keywords' in data else item.keywords
    item.required_points = json.dumps(data.get('required_points', [])) if 'required_points' in data else item.required_points
    item.status = data.get('status', item.status)
    
    db.session.commit()
    
    return jsonify({'code': 200, 'message': '更新成功', 'data': item.to_dict()})

@app.route('/api/knowledge-base/<int:item_id>', methods=['DELETE'])
def delete_knowledge_item(item_id):
    item = KnowledgeBase.query.get_or_404(item_id)
    db.session.delete(item)
    db.session.commit()
    return jsonify({'code': 200, 'message': '删除成功'})

@app.route('/api/knowledge-base/search')
def search_knowledge():
    query = request.args.get('q', '')
    items = KnowledgeBase.query.filter(
        (KnowledgeBase.title.contains(query)) | 
        (KnowledgeBase.content.contains(query))
    ).all()
    return jsonify({'code': 200, 'data': [item.to_dict() for item in items]})

@app.route('/api/meetings/<int:meeting_id>/compliance')
def get_compliance_report(meeting_id):
    report = ComplianceReport.query.filter_by(meeting_id=meeting_id).first()
    
    if report:
        return jsonify({'code': 200, 'data': report.to_dict()})
    else:
        return jsonify({'code': 404, 'message': '合规报告不存在'}), 404

@app.route('/api/score-weights', methods=['GET'])
def get_score_weights():
    weights = ScoreWeight.query.all()
    if weights:
        return jsonify({'code': 200, 'data': [w.to_dict() for w in weights]})
    else:
        default_weights = app.config['SCORE_WEIGHTS']
        return jsonify({'code': 200, 'data': [
            {'weight_name': k, 'weight_value': v} for k, v in default_weights.items()
        ]})

@app.route('/api/score-weights', methods=['PUT'])
def update_score_weights():
    data = request.get_json()
    
    for name, value in data.items():
        weight = ScoreWeight.query.filter_by(weight_name=name).first()
        if weight:
            weight.weight_value = value
        else:
            weight = ScoreWeight(weight_name=name, weight_value=value)
            db.session.add(weight)
    
    db.session.commit()
    
    return jsonify({'code': 200, 'message': '权重更新成功'})

@app.route('/api/languages')
def get_languages():
    languages = {
        'zh': 'Chinese',
        'en': 'English',
        'ja': 'Japanese',
        'ko': 'Korean',
        'fr': 'French',
        'de': 'German',
        'es': 'Spanish',
        'ru': 'Russian',
        'ar': 'Arabic',
        'pt': 'Portuguese'
    }
    return jsonify({'code': 200, 'data': languages})

@app.route('/api/topics')
def get_topics():
    return jsonify({'code': 200, 'data': app.config['TOPICS']})

@app.route('/api/risk-keywords')
def get_risk_keywords():
    return jsonify({'code': 200, 'data': app.config['RISK_KEYWORDS']})

@app.route('/api/hardware/status')
def get_hardware_status():
    import sounddevice as sd
    import platform
    
    status = {
        'system': platform.system(),
        'platform': platform.platform(),
        'microphones': [],
        'speakers': [],
        'camera': None
    }
    
    try:
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                status['microphones'].append({
                    'id': i,
                    'name': dev['name'],
                    'channels': dev['max_input_channels'],
                    'sample_rate': int(dev['default_samplerate'])
                })
            if dev['max_output_channels'] > 0:
                status['speakers'].append({
                    'id': i,
                    'name': dev['name'],
                    'channels': dev['max_output_channels']
                })
    except Exception as e:
        status['microphones'] = ['无法检测']
        status['speakers'] = ['无法检测']
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            status['camera'] = {
                'available': True,
                'resolution': f'{int(width)}x{int(height)}',
                'fps': int(fps)
            }
            cap.release()
        else:
            status['camera'] = {'available': False}
    except ImportError:
        status['camera'] = {'available': False, 'error': 'OpenCV未安装'}
    except Exception as e:
        status['camera'] = {'available': False, 'error': str(e)}
    
    return jsonify({'code': 200, 'data': status})

@app.route('/api/meetings/<int:meeting_id>/participants')
def get_meeting_participants(meeting_id):
    transcriptions = Transcription.query.filter_by(meeting_id=meeting_id).all()
    speaker_segments = [t.to_dict() for t in transcriptions]
    
    participant_count = count_participants(speaker_segments)
    distribution = analyze_participation_distribution(speaker_segments)
    
    return jsonify({
        'code': 200,
        'data': {
            'participant_count': participant_count,
            'distribution': distribution
        }
    })

@app.route('/api/reports/meeting-summary/<int:meeting_id>')
def get_meeting_summary_report(meeting_id):
    meeting = Meeting.query.get_or_404(meeting_id)
    transcriptions = Transcription.query.filter_by(meeting_id=meeting_id).all()
    compliance_report = ComplianceReport.query.filter_by(meeting_id=meeting_id).first()
    
    meeting_data = {
        'meeting_id': meeting.id,
        'title': meeting.title,
        'date': meeting.date.isoformat(),
        'duration': meeting.duration,
        'participant_count': count_participants([t.to_dict() for t in transcriptions]),
        'transcriptions': [t.to_dict() for t in transcriptions]
    }
    
    full_text = ' '.join([t.text for t in transcriptions])
    meeting_data['keywords'] = extract_keywords(full_text, top_n=10)
    meeting_data['topics'] = analyze_topic(full_text)
    meeting_data['summary'] = generate_summary(full_text, max_length=300)
    meeting_data['sentiment'] = analyze_sentiment(full_text)
    
    if compliance_report:
        meeting_data['compliance'] = compliance_report.to_dict()
    
    html_report = generate_report_html('meeting_summary', meeting_data)
    
    return html_report, 200, {'Content-Type': 'text/html; charset=utf-8'}

@app.route('/api/reports/compliance-trend')
def get_compliance_trend_report():
    meetings = Meeting.query.filter_by(status='finished').order_by(Meeting.date).all()
    meetings_data = [m.to_dict() for m in meetings]
    
    for m in meetings:
        report = ComplianceReport.query.filter_by(meeting_id=m.id).first()
        if report:
            data = next(d for d in meetings_data if d['id'] == m.id)
            data['compliance_report'] = report
    
    report = generate_compliance_trend_report(meetings_data)
    
    if not report:
        return jsonify({'code': 404, 'message': '暂无会议数据'}), 404
    
    html_report = generate_report_html('compliance_trend', report)
    return html_report, 200, {'Content-Type': 'text/html; charset=utf-8'}

@app.route('/api/meeting-status')
def get_meeting_status_detection():
    import sounddevice as sd
    import numpy as np
    
    try:
        duration = 5
        fs = 16000
        
        audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        
        rms = np.sqrt(np.mean(audio_data ** 2))
        is_speech = rms > 0.03
        
        return jsonify({
            'code': 200,
            'data': {
                'audio_level': float(rms),
                'is_speech_detected': bool(is_speech),
                'suggested_action': 'start_recording' if is_speech else 'monitoring'
            }
        })
    except Exception as e:
        return jsonify({'code': 500, 'message': f'检测失败: {str(e)}'}), 500

with app.app_context():
    db.create_all()

    if not ScoreWeight.query.first():
        for name, value in app.config['SCORE_WEIGHTS'].items():
            weight = ScoreWeight(weight_name=name, weight_value=value)
            db.session.add(weight)
        db.session.commit()

if __name__ == '__main__':
    init_whisper_model()
    register_socketio_events(socketio, whisper_model)
    print(f'\nReady to accept requests on 0.0.0.0:5000')
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)