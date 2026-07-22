import os
import json
import uuid
import threading
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

# 分析进度存储
analysis_progress = {}

from database import db

db.init_app(app)

from models import Meeting, Transcription, KnowledgeBase, ComplianceReport, ScoreWeight
from modules.audio_preprocessor import preprocess_audio, format_time, detect_speech_segments, get_audio_quality_report
from modules.speaker_diarization import speaker_diarization_simple
from modules.text_analyzer import extract_keywords, analyze_topic, generate_summary, analyze_sentiment
from modules.compliance_checker import calculate_compliance_score, generate_compliance_report, get_score_level
from modules.funasr_transcriber import register_socketio_events
from modules.meeting_detector import MeetingDetector, count_participants, analyze_participation_distribution
from modules.report_generator import generate_meeting_summary_report, generate_compliance_trend_report, generate_report_html

whisper_model = None
whisper_model_realtime = None

def init_whisper_model():
    """加载上传分析用的模型（medium，准确度优先）"""
    global whisper_model
    if whisper_model is None:
        model_name = app.config['WHISPER_MODEL']
        print(f'Initializing Whisper model ({model_name}) for upload analysis...')
        try:
            whisper_model = whisper.load_model(model_name)
            print(f'Whisper {model_name} model loaded successfully')
        except Exception as e:
            print(f'Failed to load {model_name}: {e}')
            print('Loading small model as fallback...')
            whisper_model = whisper.load_model('small')
            print('Whisper small model loaded')

def init_whisper_model_realtime():
    """加载实时转写用的模型（small，速度优先）"""
    global whisper_model_realtime
    if whisper_model_realtime is None:
        model_name = app.config['WHISPER_MODEL_REALTIME']
        print(f'Initializing Whisper model ({model_name}) for realtime transcription...')
        try:
            whisper_model_realtime = whisper.load_model(model_name)
            print(f'Whisper {model_name} model loaded successfully')
        except Exception as e:
            print(f'Failed to load {model_name}: {e}')
            whisper_model_realtime = whisper.load_model('tiny')
            print('Whisper tiny model loaded')

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
    print('create_meeting called')
    if 'audio_file' not in request.files:
        return jsonify({'code': 400, 'message': 'No audio file provided'}), 400
    
    audio_file = request.files['audio_file']
    print(f'Got audio file: {audio_file.filename}')
    if audio_file.filename == '':
        return jsonify({'code': 400, 'message': 'No audio file selected'}), 400
    
    if not allowed_file(audio_file.filename):
        return jsonify({'code': 400, 'message': 'File type not allowed'}), 400
    
    meeting_title = request.form.get('meeting_title', '未命名会议')
    enable_diarization = request.form.get('enable_diarization', 'false').lower() == 'true'
    enable_compliance = request.form.get('enable_compliance', 'true').lower() == 'true'

    file_ext = audio_file.filename.rsplit('.', 1)[1].lower()
    file_id = str(uuid.uuid4())
    original_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{file_id}_original.{file_ext}')
    processed_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{file_id}_processed.wav')

    print(f'Saving file to: {original_path}')
    audio_file.save(original_path)
    print('File saved')

    file_size = os.path.getsize(original_path)
    print(f'File size: {file_size} bytes')
    if file_size < 1024:
        return jsonify({'code': 400, 'message': '音频文件太小或为空，请上传有效的音频文件'}), 400

    processed_path = original_path

    meeting = Meeting(
        title=meeting_title,
        date=datetime.now(),
        status='processing',
        audio_path=processed_path
    )
    db.session.add(meeting)
    db.session.commit()
    print(f'Meeting created with ID: {meeting.id}', flush=True)

    thread = threading.Thread(
        target=_process_meeting_async,
        args=(meeting.id, processed_path, enable_diarization, enable_compliance)
    )
    thread.daemon = True
    thread.start()
    print('Async thread started', flush=True)

    return jsonify({'code': 200, 'message': '分析已开始', 'meeting_id': meeting.id})


def _process_meeting_async(meeting_id, audio_path, enable_diarization, enable_compliance):
    """异步处理会议分析"""
    import time as _time
    import os

    processed_path = audio_path.rsplit('.', 1)[0] + '_processed.wav'

    def update_progress(percent, message):
        analysis_progress[meeting_id] = {'progress': percent, 'message': message}
        try:
            socketio.emit('analysis_progress', {'meeting_id': meeting_id, 'progress': percent, 'message': message})
        except Exception:
            pass

    try:
        update_progress(3, '正在预处理音频...')
        _t_pre = _time.time()
        if enable_diarization:
            try:
                preprocess_audio(audio_path, processed_path)
                if os.path.exists(processed_path):
                    audio_path = processed_path
                print(f'Meeting {meeting_id}: preprocessing done in {_time.time()-_t_pre:.1f}s')
            except Exception as e:
                print(f'Audio preprocessing failed: {e}')

        update_progress(5, '正在初始化模型...')
        init_whisper_model()

        update_progress(10, '正在转写音频...')
        print(f'Meeting {meeting_id}: starting transcription...')
        _t0 = _time.time()
        from modules.whisper_utils import transcribe_with_fix
        result = transcribe_with_fix(whisper_model, audio_path, language='zh')
        full_text = result['text']
        print(f'Meeting {meeting_id}: transcription done in {_time.time()-_t0:.1f}s, text length: {len(full_text)}')

        update_progress(40, '正在进行说话人分离...')
        speaker_segments = []
        if enable_diarization:
            try:
                _t1 = _time.time()
                speaker_segments = speaker_diarization_simple(audio_path)
                print(f'Meeting {meeting_id}: diarization done in {_time.time()-_t1:.1f}s')
            except Exception as e:
                print(f'Speaker diarization failed: {e}')

        with app.app_context():
            meeting = Meeting.query.get(meeting_id)
            if not meeting:
                print(f'Meeting {meeting_id}: not found, aborting')
                return

            # 保存转写段落
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
            db.session.commit()

            _t2 = _time.time()
            update_progress(55, '正在提取关键词...')
            keywords = extract_keywords(full_text, top_n=10)
            print(f'Meeting {meeting_id}: keywords done in {_time.time()-_t2:.1f}s')
            
            _t3 = _time.time()
            update_progress(65, '正在分析主题...')
            topics = analyze_topic(full_text)
            print(f'Meeting {meeting_id}: topics done in {_time.time()-_t3:.1f}s')
            
            _t4 = _time.time()
            update_progress(75, '正在生成会议摘要...')
            print(f'=== About to call generate_summary, text length: {len(full_text)}')
            summary = generate_summary(full_text, max_length=300)
            print(f'=== generate_summary returned: {summary[:100]}')
            print(f'Meeting {meeting_id}: summary done in {_time.time()-_t4:.1f}s')
            
            _t5 = _time.time()
            update_progress(80, '正在分析情绪...')
            sentiment = analyze_sentiment(full_text)
            print(f'Meeting {meeting_id}: sentiment done in {_time.time()-_t5:.1f}s')

            audio_quality = None
            if enable_diarization:
                try:
                    audio_quality = get_audio_quality_report(audio_path, audio_path)
                except Exception as e:
                    print(f'Audio quality report failed: {e}')

            meeting.duration = int(result['segments'][-1]['end']) if result['segments'] else 0
            meeting.summary = summary
            meeting.keywords = json.dumps(keywords)
            meeting.topics = json.dumps(topics)
            meeting.sentiment = json.dumps(sentiment)

            compliance_result = None
            if enable_compliance:
                update_progress(85, '正在进行合规检查...')
                knowledge_items = KnowledgeBase.query.filter_by(status='active').all()
                transcription_segments = [t.to_dict() for t in Transcription.query.filter_by(meeting_id=meeting.id).all()]
                
                score_weights = {}
                db_weights = ScoreWeight.query.all()
                if db_weights:
                    for w in db_weights:
                        score_weights[w.weight_name] = w.weight_value
                else:
                    score_weights = app.config['SCORE_WEIGHTS']
                
                compliance_result = calculate_compliance_score(full_text, knowledge_items, score_weights=score_weights, transcription_segments=transcription_segments)

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

            update_progress(100, '分析完成')
            print(f'Meeting {meeting_id} analysis completed successfully')

    except Exception as e:
        print(f'Meeting {meeting_id} analysis failed: {e}')
        import traceback
        traceback.print_exc()
        update_progress(-1, f'分析失败: {str(e)}')
        with app.app_context():
            meeting = Meeting.query.get(meeting_id)
            if meeting:
                meeting.status = 'failed'
                db.session.commit()


@app.route('/api/meetings/<int:meeting_id>/progress')
def get_analysis_progress(meeting_id):
    """获取分析进度"""
    progress = analysis_progress.get(meeting_id, {'progress': 0, 'message': '等待中...'})
    return jsonify({'code': 200, 'data': progress})

@app.route('/api/meetings/<int:meeting_id>', methods=['PUT'])
def update_meeting(meeting_id):
    meeting = Meeting.query.get_or_404(meeting_id)
    
    data = request.get_json()
    
    if 'title' in data:
        meeting.title = data['title']
    if 'summary' in data:
        meeting.summary = data['summary']
    if 'total_score' in data:
        meeting.total_score = data['total_score']
    
    db.session.commit()
    
    return jsonify({'code': 200, 'message': '更新成功'})

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
    content_type = request.headers.get('Content-Type', '')
    
    if 'multipart/form-data' in content_type and 'file' in request.files:
        return upload_knowledge_file()
    
    try:
        data = request.get_json()
    except:
        data = None
    
    if data is None:
        return jsonify({'code': 400, 'message': '请求格式错误'}), 400
    
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


def upload_knowledge_file():
    """上传政策文件并自动解析"""
    file = request.files['file']
    
    if not file or file.filename == '':
        return jsonify({'code': 400, 'message': '请选择文件'}), 400
    
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ['.txt', '.pdf', '.docx', '.doc']:
        return jsonify({'code': 400, 'message': '仅支持 .txt、.pdf、.docx、.doc 格式'}), 400
    
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
        file.save(f.name)
        temp_path = f.name
    
    try:
        from modules.file_parser import parse_policy_file
        parsed = parse_policy_file(temp_path)
        
        item_type = request.form.get('item_type', 'policy')
        
        item = KnowledgeBase(
            title=parsed['title'],
            content=parsed['content'],
            item_type=item_type,
            keywords=json.dumps(parsed['keywords']),
            required_points=json.dumps(parsed['required_points']),
            status='active'
        )
        db.session.add(item)
        db.session.commit()
        
        return jsonify({
            'code': 200,
            'message': '上传成功',
            'data': item.to_dict(),
            'parsed': {
                'title': parsed['title'],
                'keywords': parsed['keywords'],
                'required_points': parsed['required_points'],
                'content_length': len(parsed['content'])
            }
        })
    except Exception as e:
        print(f'文件解析失败: {e}')
        return jsonify({'code': 500, 'message': f'文件解析失败: {str(e)}'}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

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

@app.route('/api/meetings/test-analyze', methods=['POST'])
def test_analyze():
    data = request.get_json()
    text = data.get('text', '')
    
    from modules.compliance_checker import check_compliance, calculate_compliance_score
    
    result = check_compliance(text)
    weights = get_default_weights()
    score_result = calculate_compliance_score(result, weights)
    
    return jsonify({
        'code': 200,
        'compliance_score': score_result.get('total_score'),
        'score_level': score_result.get('score_level'),
        'missing_points': result.get('missing_points'),
        'risk_keywords': result.get('risk_keywords'),
        'suggestions': result.get('suggestions'),
        'matched_keywords': result.get('matched_keywords')
    })


@app.route('/api/meetings/test-summary', methods=['POST'])
def test_summary():
    data = request.get_json()
    text = data.get('text', '')
    
    from modules.text_analyzer import generate_summary, extract_keywords, analyze_topic, analyze_sentiment
    
    summary = generate_summary(text)
    keywords = extract_keywords(text)
    topics = analyze_topic(text)
    sentiment = analyze_sentiment(text)
    
    return jsonify({
        'code': 200,
        'summary': summary,
        'keywords': keywords,
        'topics': topics,
        'sentiment': sentiment
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

        # 安全检查：确保数据有效
        if audio_data is None or len(audio_data) == 0:
            return jsonify({
                'code': 200,
                'data': {
                    'audio_level': 0.0,
                    'audio_level_db': -100.0,
                    'is_speech_detected': False,
                    'suggested_action': 'monitoring'
                }
            })

        # 计算 RMS，处理异常值
        squared = np.square(audio_data.astype(np.float64))
        mean_sq = np.mean(squared)

        # 防止除零和无效值
        if not np.isfinite(mean_sq) or mean_sq <= 0:
            rms = 0.0
            db = -100.0
        else:
            rms = float(np.sqrt(mean_sq))
            # 转换为分贝
            db = 20.0 * np.log10(rms + 1e-10)

        # 再次检查是否为有效数值
        if not np.isfinite(rms):
            rms = 0.0
        if not np.isfinite(db):
            db = -100.0

        is_speech = rms > 0.03

        return jsonify({
            'code': 200,
            'data': {
                'audio_level': round(rms, 6),
                'audio_level_db': round(float(db), 2),
                'is_speech_detected': bool(is_speech),
                'suggested_action': 'start_recording' if is_speech else 'monitoring'
            }
        })
    except Exception as e:
        return jsonify({'code': 500, 'message': f'检测失败: {str(e)}'}), 500

with app.app_context():
    db.create_all()
    
    # 确保uploads目录存在
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    if not ScoreWeight.query.first():
        for name, value in app.config['SCORE_WEIGHTS'].items():
            weight = ScoreWeight(weight_name=name, weight_value=value)
            db.session.add(weight)
        db.session.commit()
    
    # 初始化一些示例数据（如果知识库为空）
    if not KnowledgeBase.query.first():
        # 添加一些示例的政策文件
        policy1 = KnowledgeBase(
            title='公司会议规范',
            content='所有公司会议必须遵循公司的价值观，尊重每一位参会者的意见，保持积极的工作态度。',
            item_type='policy',
            keywords=json.dumps(['规范', '价值观', '尊重']),
            required_points=json.dumps([])
        )
        
        # 添加风险关键词
        risk_keywords = KnowledgeBase(
            title='风险词汇表',
            content='会议中应避免使用的负面或消极词汇。',
            item_type='risk_keywords',
            keywords=json.dumps(['消极', '反对', '抵制', '抱怨', '不满', '拒绝', '不行', '不可能', '做不到']),
            required_points=json.dumps([])
        )
        
        # 添加必传要点
        key_points = KnowledgeBase(
            title='项目例会要点',
            content='项目例会必须包含的要点内容。',
            item_type='key_points',
            keywords=json.dumps(['进度', '问题', '计划', '目标']),
            required_points=json.dumps(['进度汇报', '问题讨论', '下周计划', '风险说明'])
        )
        
        db.session.add(policy1)
        db.session.add(risk_keywords)
        db.session.add(key_points)
        db.session.commit()
    
    # 补充金融合规知识库模板（如果不存在）
    if not KnowledgeBase.query.filter_by(title='理财产品销售合规管理办法').first():
        # 金融合规模板
        fin_policy = KnowledgeBase(
            title='理财产品销售合规管理办法',
            content='理财产品销售必须遵守合规要求，包括投资者适当性管理、风险测评、风险告知、禁止误导性宣传等。销售人员必须持证上岗，销售过程需录音录像。',
            item_type='policy',
            keywords=json.dumps(['风险', '销售', '投资者', '必须', '理财', '产品', '合规', '客户', '告知', '测评', '适当性']),
            required_points=json.dumps([])
        )
        
        fin_risk = KnowledgeBase(
            title='金融销售风险关键词',
            content='金融销售中禁止使用的风险词汇。',
            item_type='risk_keywords',
            keywords=json.dumps(['保本保收益', '零风险', '稳赚不赔', '绝对安全', '保证收益', '高收益无风险', '只赚不赔', '无风险']),
            required_points=json.dumps([])
        )
        
        fin_points = KnowledgeBase(
            title='理财销售必传要点',
            content='理财产品销售必须覆盖的合规要点。',
            item_type='key_points',
            keywords=json.dumps(['风险测评', '适当性', '风险告知', '录音录像', '持证上岗', '风险等级', '承受能力', '书面确认', '风险揭示', '合规销售']),
            required_points=json.dumps(['投资者风险测评', '风险等级匹配', '风险告知义务', '销售过程录音录像', '销售人员持证上岗', '书面确认风险揭示书'])
        )
        
        db.session.add(fin_policy)
        db.session.add(fin_risk)
        db.session.add(fin_points)
        db.session.commit()

if __name__ == '__main__':
    init_whisper_model()          # 加载 medium 用于上传分析
    
    from modules.funasr_transcriber import init_funasr_model
    init_funasr_model()           # 预加载FunASR用于实时转写
    
    register_socketio_events(socketio)
    print(f'\nReady to accept requests on 0.0.0.0:5000')
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)