import os
import json
import uuid
import threading
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO

app = Flask(__name__)
app.config.from_object('config.Config')
CORS(app, origins=app.config.get('CORS_ORIGINS') or '*')
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

if app.config.get('TRUST_PROXY'):
    from werkzeug.middleware.proxy_fix import ProxyFix
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

# 分析进度存储
analysis_progress = {}

from database import db

db.init_app(app)

from models import Meeting, Transcription, KnowledgeBase, ComplianceReport, ScoreWeight
from modules.meeting_detector import count_participants, analyze_participation_distribution
from modules import meeting_store

# 实时转写的 Socket.IO 事件必须在模块导入时就注册。
# 旧实现把注册放在 __main__ 分支里，导致用 flask run / gunicorn / 其它启动器
# 启动时实时转写完全不可用（连接成功但没有任何事件处理器）。
from modules.funasr_transcriber import register_socketio_events
register_socketio_events(socketio)

# 语音识别与模型加载：统一走 asr_engine（FunASR 优先，Whisper 自动兜底）
_engine_lock = threading.Lock()
_engine_ready = {'asr': False, 'vad': False}


def init_asr_models():
    """预加载语音识别相关模型（幂等）。"""
    from modules import asr_engine

    with _engine_lock:
        if not _engine_ready['asr']:
            engine = asr_engine.get_engine('auto')
            _engine_ready['asr'] = True
            print(f'[启动] 语音识别引擎就绪: {engine.name}')
        if not _engine_ready['vad']:
            from modules import model_registry

            model_registry.get_vad_model()
            _engine_ready['vad'] = True
    return True


def init_whisper_model():
    """兼容旧调用点：返回当前识别引擎。"""
    from modules import asr_engine

    return asr_engine.get_engine('auto')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def parse_bool(value, default=False):
    """解析布尔参数（表单里可能是 'true'/'false'/'1'/'0' 字符串）。"""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {'1', 'true', 'yes', 'on'}


def clamp_page_size(value):
    """限制分页大小，避免一次拉取过多数据。"""
    try:
        size = int(value)
    except (TypeError, ValueError):
        size = 10
    return max(1, min(size, app.config.get('MAX_PAGE_SIZE', 100)))


def resolve_score_weights():
    """读取评分权重：数据库优先，其次配置默认值。"""
    weights = {}
    for weight in ScoreWeight.query.all():
        weights[weight.weight_name] = weight.weight_value
    return weights or dict(app.config['SCORE_WEIGHTS'])


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/health')
def health():
    """健康检查：附带本地模型可用性，便于启动脚本判断服务是否真正就绪。"""
    from modules import model_registry

    return jsonify({
        'status': 'ok',
        'message': 'API service is running',
        'models': model_registry.status(),
    })

@app.route('/api/meetings', methods=['GET'])
def get_meetings():
    page = request.args.get('page', 1, type=int)
    page_size = clamp_page_size(request.args.get('page_size', 10))
    status = request.args.get('status')
    min_score = request.args.get('min_score', type=float)
    max_score = request.args.get('max_score', type=float)
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    include_compliance = parse_bool(request.args.get('include_compliance'), default=False)

    query = Meeting.query
    if status:
        query = query.filter_by(status=status)
    if min_score is not None:
        query = query.filter(Meeting.total_score >= min_score)
    if max_score is not None:
        query = query.filter(Meeting.total_score <= max_score)
    if start_date:
        try:
            query = query.filter(Meeting.date >= datetime.fromisoformat(start_date))
        except ValueError:
            return jsonify({'code': 400, 'message': 'start_date 格式应为 YYYY-MM-DD'}), 400
    if end_date:
        try:
            # 含当天：取结束日期次日零点为上界
            query = query.filter(Meeting.date < datetime.fromisoformat(end_date) + timedelta(days=1))
        except ValueError:
            return jsonify({'code': 400, 'message': 'end_date 格式应为 YYYY-MM-DD'}), 400

    meetings = query.order_by(Meeting.created_at.desc()).paginate(
        page=page, per_page=page_size, error_out=False
    )

    items = [m.to_dict() for m in meetings.items]

    if include_compliance and items:
        # 一次性把本页会议的合规摘要带出来，避免前端逐条再请求（N+1）。
        # 列表页需要「遗漏要点数 / 风险内容数 / 建议」，这些只在合规报告里。
        ids = [item['id'] for item in items]
        reports = {
            report.meeting_id: report
            for report in ComplianceReport.query.filter(
                ComplianceReport.meeting_id.in_(ids)
            ).all()
        }
        for item in items:
            report = reports.get(item['id'])
            if report is None:
                item['compliance_summary'] = None
                continue
            missing = json.loads(report.missing_points) if report.missing_points else []
            risks = json.loads(report.risk_keywords) if report.risk_keywords else []
            suggestions = json.loads(report.suggestions) if report.suggestions else []
            item['compliance_summary'] = {
                'missing_points_count': len(missing),
                'risk_keywords_count': len(risks),
                'suggestions_count': len(suggestions),
                'first_suggestion': suggestions[0] if suggestions else None,
            }

    return jsonify({
        'code': 200,
        'data': items,
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
        return jsonify({'code': 400, 'message': '未选择音频文件'}), 400

    if not allowed_file(audio_file.filename):
        return jsonify({
            'code': 400,
            'message': f'不支持的文件类型，支持：{", ".join(sorted(app.config["ALLOWED_EXTENSIONS"]))}',
        }), 400

    meeting_title = (request.form.get('meeting_title') or '未命名会议').strip()
    enable_diarization = parse_bool(request.form.get('enable_diarization'), default=True)
    enable_compliance = parse_bool(request.form.get('enable_compliance'), default=True)

    file_ext = audio_file.filename.rsplit('.', 1)[1].lower()
    file_id = str(uuid.uuid4())
    original_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{file_id}_original.{file_ext}')

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    audio_file.save(original_path)

    # 文件过小直接拒绝，并清理已写入的临时文件，避免残留垃圾
    if os.path.getsize(original_path) < 1024:
        try:
            os.remove(original_path)
        except OSError:
            pass
        return jsonify({'code': 400, 'message': '音频文件太小或为空，请上传有效的音频文件'}), 400

    meeting = Meeting(
        title=meeting_title,
        date=datetime.now(),
        status='processing',
        audio_path=original_path,
    )
    db.session.add(meeting)
    db.session.commit()

    thread = threading.Thread(
        target=_process_meeting_async,
        args=(meeting.id, original_path, enable_diarization, enable_compliance),
        daemon=True,
    )
    thread.start()

    return jsonify({'code': 200, 'message': '分析已开始', 'meeting_id': meeting.id})


def _process_meeting_async(meeting_id, audio_path, enable_diarization, enable_compliance):
    """
    异步处理会议分析（上传音频路径）。

    注意：本函数运行在独立线程里，线程内没有 Flask 应用上下文，
    因此所有数据库访问（查知识库、查评分权重、读写会议记录）都必须放在
    `with app.app_context():` 之内，否则会抛
    "Working outside of application context."。
    """
    import time as _time

    def update_progress(percent, message):
        analysis_progress[meeting_id] = {'progress': percent, 'message': message}
        try:
            socketio.emit('analysis_progress', {
                'meeting_id': meeting_id,
                'progress': percent,
                'message': message,
            })
        except Exception:
            pass

    try:
        from modules.audio_preprocessor import preprocess_audio, get_audio_quality_report
        from modules.text_analyzer import (
            analyze_sentiment, analyze_topic, extract_action_items,
            extract_decisions, extract_keywords, generate_summary,
        )
        from modules.compliance_checker import calculate_compliance_score, get_score_level
        from modules import asr_engine, speaker_diarization

        started = _time.time()

        # 知识库条目（热词增强与合规比对共用）。这里单独开一次应用上下文：
        # 线程内没有上下文，识别之前就要拿到关键词做热词。
        with app.app_context():
            knowledge_items = KnowledgeBase.query.filter_by(status='active').all()
            hotwords = asr_engine.build_hotwords(knowledge_items)
        print(f'Meeting {meeting_id}: 热词 {len(hotwords or [])} 条')

        # 预处理：按信噪比自适应（干净音频走轻度处理，含噪音频才降噪，见 audio_preprocessor）
        update_progress(5, '正在预处理音频...')
        processed_path = audio_path.rsplit('.', 1)[0] + '_processed.wav'
        analysis_audio = audio_path
        preprocessing_ok = False
        try:
            preprocess_audio(audio_path, processed_path)
            if os.path.exists(processed_path):
                analysis_audio = processed_path
                preprocessing_ok = True
        except Exception as exc:
            print(f'Meeting {meeting_id}: 音频预处理失败: {exc}')

        update_progress(15, '正在识别语音...')
        result = asr_engine.transcribe(analysis_audio, language='zh', hotwords=hotwords)
        full_text = result['text']
        print(f'Meeting {meeting_id}: 识别完成，引擎={result.get("engine")}，'
              f'{len(result["segments"])} 句，耗时 {_time.time()-started:.1f}s')

        if not full_text.strip():
            raise RuntimeError('未识别到有效语音内容')

        update_progress(40, '正在进行说话人分离...')
        speaker_segments = []
        if enable_diarization:
            try:
                t0 = _time.time()
                # 说话人分离必须用原始音频：预处理（降噪/语音增强）会改变频谱包络，
                # 而声纹特征恰恰依赖频谱包络，实测在预处理后音频上 2 人会并成 1 人
                # （混淆率 3.55% -> 38.53%）。预处理只用于提升识别质量。
                speaker_segments = speaker_diarization.speaker_diarization_simple(audio_path)
                print(f'Meeting {meeting_id}: 说话人分离耗时 {_time.time()-t0:.1f}s')
            except Exception as exc:
                print(f'Meeting {meeting_id}: 说话人分离失败: {exc}')

        # 识别分句 → 会议转写记录（按时间重叠最大者判定说话人）
        raw_segments = [{
            'speaker': 'SPEAKER_00',
            'text': seg['text'],
            'start_time': seg['start'],
            'end_time': seg['end'],
            'confidence': seg.get('confidence', 1.0),
            'language': result.get('language', 'zh'),
        } for seg in result['segments']]

        if speaker_segments:
            raw_segments = speaker_diarization.assign_speakers_to_segments(
                raw_segments, speaker_segments
            )

        audio_quality = None
        if preprocessing_ok:
            try:
                # 对比原始音频与预处理后音频，评估降噪/增强效果
                audio_quality = get_audio_quality_report(audio_path, processed_path)
            except Exception as exc:
                print(f'Meeting {meeting_id}: 音频质量报告失败: {exc}')

        update_progress(55, '正在提取关键词...')
        keywords = extract_keywords(full_text, top_n=10)

        update_progress(65, '正在分析主题...')
        topics = analyze_topic(full_text)

        update_progress(75, '正在生成会议摘要...')
        summary = generate_summary(full_text, max_length=300)

        update_progress(80, '正在提取待办事项与决议...')
        action_items = extract_action_items(full_text)
        decisions = extract_decisions(full_text)

        sentiment = analyze_sentiment(full_text)

        duration = int(max((s.get('end_time') or 0) for s in raw_segments)) if raw_segments else 0

        # 数据库相关操作统一放在应用上下文中（线程内默认没有上下文）
        with app.app_context():
            meeting = Meeting.query.get(meeting_id)
            if not meeting:
                print(f'Meeting {meeting_id}: 记录不存在，终止')
                return

            compliance_result = None
            if enable_compliance:
                update_progress(88, '正在进行合规检查...')
                # knowledge_items 已在识别前取好（同时用于热词增强）
                if knowledge_items:
                    compliance_result = calculate_compliance_score(
                        full_text,
                        knowledge_items,
                        score_weights=resolve_score_weights(),
                        transcription_segments=raw_segments,
                    )
                    compliance_result['score_level'] = get_score_level(
                        compliance_result['total_score']
                    )

            for segment in raw_segments:
                db.session.add(Transcription(
                    meeting_id=meeting.id,
                    speaker=segment.get('speaker') or 'SPEAKER_00',
                    text=segment['text'],
                    start_time=segment['start_time'],
                    end_time=segment['end_time'],
                    confidence=segment.get('confidence', 0.0),
                    language=segment.get('language', 'zh'),
                ))

            meeting.duration = duration
            meeting.summary = summary
            meeting.keywords = json.dumps(keywords, ensure_ascii=False)
            meeting.topics = json.dumps(topics, ensure_ascii=False)
            meeting.sentiment = json.dumps(sentiment, ensure_ascii=False)
            meeting.action_items = json.dumps(action_items, ensure_ascii=False)
            meeting.decisions = json.dumps(decisions, ensure_ascii=False)
            if audio_quality:
                meeting.audio_quality = json.dumps(audio_quality, ensure_ascii=False)

            if compliance_result:
                db.session.add(ComplianceReport(
                    meeting_id=meeting.id,
                    total_score=compliance_result['total_score'],
                    score_level=compliance_result.get('score_level', ''),
                    detailed_scores=json.dumps(compliance_result['components'], ensure_ascii=False),
                    missing_points=json.dumps(compliance_result['missing_points'], ensure_ascii=False),
                    risk_keywords=json.dumps(compliance_result['risk_keywords_found'], ensure_ascii=False),
                    risk_time_markers=json.dumps(compliance_result.get('risk_time_markers', []), ensure_ascii=False),
                    point_time_markers=json.dumps(compliance_result.get('point_time_markers', []), ensure_ascii=False),
                    matched_keywords=json.dumps(compliance_result['matched_keywords'], ensure_ascii=False),
                    suggestions=json.dumps(compliance_result['suggestions'], ensure_ascii=False),
                ))
                meeting.total_score = compliance_result['total_score']
                meeting.score_level = compliance_result.get('score_level', '')

            meeting.status = 'finished'
            db.session.commit()

        update_progress(100, '分析完成')
        print(f'Meeting {meeting_id} 分析完成，总耗时 {_time.time()-started:.1f}s')

    except Exception as exc:
        print(f'Meeting {meeting_id} 分析失败: {exc}')
        import traceback
        traceback.print_exc()
        update_progress(-1, f'分析失败: {exc}')
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


# ---------------------------------------------------------------- 实时转写（HTTP）
# 内置页面用 HTTP 分块上传 + 轮询，避免依赖 CDN 上的 socket.io 客户端
# （离线环境下加载不到脚本会让实时功能点了没反应）。详见 modules/realtime_api.py。

@app.route('/api/realtime/start', methods=['POST'])
def realtime_start():
    """开始一次实时转写会话。"""
    from modules import realtime_api

    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({'code': 400, 'message': '请求体必须是 JSON 对象'}), 400

    sample_rate = data.get('sample_rate') or 16000
    try:
        sample_rate = int(sample_rate)
    except (TypeError, ValueError):
        return jsonify({'code': 400, 'message': 'sample_rate 必须是整数'}), 400
    if not 4000 <= sample_rate <= 192000:
        return jsonify({'code': 400, 'message': 'sample_rate 超出合理范围'}), 400

    enable_compliance = parse_bool(data.get('enable_compliance'), default=True)
    knowledge_items = []
    score_weights = None
    if enable_compliance:
        knowledge_items = KnowledgeBase.query.filter_by(status='active').all()
        score_weights = resolve_score_weights()

    session = realtime_api.start_session(
        upload_folder=app.config['UPLOAD_FOLDER'],
        language=data.get('language') or 'zh',
        sample_rate=sample_rate,
        meeting_title=data.get('meeting_title'),
        hotwords=data.get('hotwords'),
        enable_compliance=enable_compliance,
        knowledge_items=knowledge_items,
        score_weights=score_weights,
    )
    return jsonify({'code': 200, 'message': '会话已开始', 'data': session})


@app.route('/api/realtime/chunk', methods=['POST'])
def realtime_chunk():
    """接收一块 Base64 编码的 PCM 音频，返回识别结果（可能暂无输出）。"""
    from modules import realtime_api

    data = request.get_json(silent=True) or {}
    session_id = data.get('session_id')
    if not session_id:
        return jsonify({'code': 400, 'message': '缺少 session_id'}), 400

    try:
        result = realtime_api.push_chunk(session_id, data.get('audio'))
    except KeyError:
        return jsonify({'code': 404, 'message': '会话不存在或已结束'}), 404
    except Exception as exc:
        print(f'[实时转写] 音频块处理失败: {exc}')
        return jsonify({'code': 400, 'message': f'音频数据无效: {exc}'}), 400

    return jsonify({'code': 200, 'data': result})


@app.route('/api/realtime/stop', methods=['POST'])
def realtime_stop():
    """结束会话：保存音频并转入后台完整分析。"""
    from modules import realtime_api

    data = request.get_json(silent=True) or {}
    session_id = data.get('session_id')
    if not session_id:
        return jsonify({'code': 400, 'message': '缺少 session_id'}), 400

    try:
        info = realtime_api.stop_session(session_id, app=app)
    except KeyError:
        return jsonify({'code': 404, 'message': '会话不存在或已结束'}), 404

    return jsonify({'code': 200, 'message': '已停止，正在分析', 'data': info})


@app.route('/api/realtime/result/<session_id>')
def realtime_result(session_id):
    """轮询实时会话的分析进度与最终结果。"""
    from modules import realtime_api

    try:
        return jsonify({'code': 200, 'data': realtime_api.get_result(session_id)})
    except KeyError:
        return jsonify({'code': 404, 'message': '会话不存在或结果已过期'}), 404


@app.route('/api/realtime/discard', methods=['POST'])
def realtime_discard():
    """放弃会话（用户中途关闭页面），清理临时音频。"""
    from modules import realtime_api

    data = request.get_json(silent=True) or {}
    session_id = data.get('session_id')
    if session_id:
        realtime_api.discard(session_id)
    return jsonify({'code': 200, 'message': '已清理'})

@app.route('/api/meetings/<int:meeting_id>', methods=['PUT'])
def update_meeting(meeting_id):
    meeting = Meeting.query.get_or_404(meeting_id)

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({'code': 400, 'message': '请求体必须是 JSON 对象'}), 400

    if 'title' in data:
        meeting.title = data['title']
    if 'summary' in data:
        meeting.summary = data['summary']
    if 'total_score' in data:
        meeting.total_score = data['total_score']

    db.session.commit()

    return jsonify({'code': 200, 'message': '更新成功', 'data': meeting.to_dict()})

@app.route('/api/meetings/<int:meeting_id>', methods=['DELETE'])
def delete_meeting(meeting_id):
    meeting = Meeting.query.get_or_404(meeting_id)

    audio_path = meeting.audio_path
    Transcription.query.filter_by(meeting_id=meeting_id).delete()
    ComplianceReport.query.filter_by(meeting_id=meeting_id).delete()
    db.session.delete(meeting)
    db.session.commit()

    # 数据库记录删除后再清理磁盘文件（原始音频 + 预处理产物）
    meeting_store.delete_meeting_files(type('M', (), {'audio_path': audio_path})())

    analysis_progress.pop(meeting_id, None)
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
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({'code': 400, 'message': '请求体必须是 JSON 对象'}), 400

    for name, value in data.items():
        weight = ScoreWeight.query.filter_by(weight_name=name).first()
        if weight:
            weight.weight_value = value
        else:
            db.session.add(ScoreWeight(weight_name=name, weight_value=value))

    db.session.commit()

    return jsonify({'code': 200, 'message': '权重更新成功', 'data': resolve_score_weights()})

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
    """
    文本合规自测接口：直接把一段文本丢进合规检查，便于调试知识库配置。

    旧实现引用了不存在的 check_compliance、未定义的 get_default_weights，
    并且把权重字典当成知识库列表传入，必然报错，这里按真实签名重写。
    """
    from modules.compliance_checker import calculate_compliance_score, get_score_level

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({'code': 400, 'message': '请求体必须是 JSON 对象'}), 400

    text = (data.get('text') or '').strip()
    if not text:
        return jsonify({'code': 400, 'message': '请提供待检查的文本'}), 400

    knowledge_items = KnowledgeBase.query.filter_by(status='active').all()
    if not knowledge_items:
        return jsonify({'code': 200, 'message': '知识库为空，请先添加合规规则', 'data': None})

    weights = resolve_score_weights()
    if isinstance(data.get('score_weights'), dict):
        weights = {**weights, **data['score_weights']}

    # 没有时间轴信息，仅校验文本层面（语义相似度/要点覆盖/风险/关键词）
    segments = [{'speaker': 'SPEAKER_00', 'text': text, 'start_time': 0, 'end_time': 0}]
    result = calculate_compliance_score(
        text, knowledge_items, score_weights=weights, transcription_segments=segments
    )

    return jsonify({
        'code': 200,
        'data': {
            'total_score': result['total_score'],
            'score_level': get_score_level(result['total_score']),
            'components': result['components'],
            'covered_points': result.get('covered_points', []),
            'missing_points': result['missing_points'],
            'risk_keywords_found': result['risk_keywords_found'],
            'risk_time_markers': result.get('risk_time_markers', []),
            'point_time_markers': result.get('point_time_markers', []),
            'matched_keywords': result['matched_keywords'],
            'suggestions': result['suggestions'],
        }
    })


@app.route('/api/meetings/test-summary', methods=['POST'])
def test_summary():
    """文本摘要自测接口。"""
    from modules.text_analyzer import (
        analyze_sentiment, analyze_topic, extract_action_items,
        extract_decisions, extract_keywords, generate_summary,
    )

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({'code': 400, 'message': '请求体必须是 JSON 对象'}), 400

    text = (data.get('text') or '').strip()
    if not text:
        return jsonify({'code': 400, 'message': '请提供待分析文本'}), 400

    return jsonify({
        'code': 200,
        'data': {
            'summary': generate_summary(text, max_length=300),
            'keywords': extract_keywords(text),
            'topics': analyze_topic(text),
            'sentiment': analyze_sentiment(text),
            'action_items': extract_action_items(text),
            'decisions': extract_decisions(text),
        }
    })


@app.route('/api/reports/meeting-summary/<int:meeting_id>')
def get_meeting_summary_report(meeting_id):
    from modules.report_generator import generate_report_html
    from modules.text_analyzer import (
        analyze_sentiment, analyze_topic, extract_action_items,
        extract_decisions, extract_keywords, generate_summary,
    )

    meeting = Meeting.query.get_or_404(meeting_id)
    transcriptions = Transcription.query.filter_by(meeting_id=meeting_id).all()
    compliance_report = ComplianceReport.query.filter_by(meeting_id=meeting_id).first()

    segments = [t.to_dict() for t in transcriptions]
    full_text = ' '.join(t.text for t in transcriptions)

    meeting_data = {
        'meeting_id': meeting.id,
        'title': meeting.title,
        'date': meeting.date.isoformat(),
        'duration': meeting.duration,
        'participant_count': count_participants(segments),
        'distribution': analyze_participation_distribution(segments),
        'transcriptions': segments,
        # 优先使用分析时已落库的结果；老记录没有这些字段时再现场计算
        'keywords': extract_keywords(full_text, top_n=10) if full_text and not meeting.keywords else (
            json.loads(meeting.keywords) if meeting.keywords else []),
        'topics': json.loads(meeting.topics) if meeting.topics else (
            analyze_topic(full_text) if full_text else []),
        'summary': meeting.summary or (generate_summary(full_text, max_length=300) if full_text else ''),
        'sentiment': json.loads(meeting.sentiment) if meeting.sentiment else (
            analyze_sentiment(full_text) if full_text else {}),
        'action_items': json.loads(meeting.action_items) if meeting.action_items else (
            extract_action_items(full_text) if full_text else []),
        'decisions': json.loads(meeting.decisions) if meeting.decisions else (
            extract_decisions(full_text) if full_text else []),
    }

    if compliance_report:
        meeting_data['compliance'] = compliance_report.to_dict()

    html_report = generate_report_html('meeting_summary', meeting_data)
    return html_report, 200, {'Content-Type': 'text/html; charset=utf-8'}

@app.route('/api/reports/compliance-trend')
def get_compliance_trend_report():
    from modules.report_generator import generate_compliance_trend_report, generate_report_html

    meetings = Meeting.query.filter_by(status='finished').order_by(Meeting.date).all()
    if not meetings:
        return jsonify({'code': 404, 'message': '暂无会议数据'}), 404

    meetings_data = []
    for meeting in meetings:
        item = meeting.to_dict()
        report = ComplianceReport.query.filter_by(meeting_id=meeting.id).first()
        # 报告生成器按字段名读取，这里统一传字典，避免混用 ORM 对象
        item['compliance_report'] = report.to_dict() if report else None
        meetings_data.append(item)

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

@app.route('/api/v1/transcribe', methods=['POST'])
def api_transcribe():
    """
    语音转写API接口
    供第三方软件调用，上传音频文件进行转写和分析
    
    请求方式: POST
    Content-Type: multipart/form-data 或 application/json
    
    参数:
        audio (file): 音频文件，支持 mp3, wav, m4a 格式
        audio_base64 (string): Base64编码的音频数据（二选一）
        language (string): 语言，默认 'zh'（中文）
        enable_compliance (boolean): 是否进行合规检查，默认 true
        enable_diarization (boolean): 是否进行说话人分离，默认 false
    
    返回:
        {
            "code": 200,
            "message": "成功",
            "data": {
                "text": "转写文本内容",
                "segments": [...],
                "keywords": [...],
                "topics": [...],
                "summary": "会议摘要",
                "sentiment": {...},
                "compliance_report": {...},
                "speaker_segments": [...]
            }
        }
    """
    audio_path = None
    processed_path = None
    try:
        import base64
        import binascii
        from modules.analysis_pipeline import analyze_audio

        json_body = request.get_json(silent=True) if request.is_json else {}
        if not isinstance(json_body, dict):
            json_body = {}
        audio_file = request.files.get('audio')
        audio_base64 = request.form.get('audio_base64') or json_body.get('audio_base64')
        
        if not audio_file and not audio_base64:
            return jsonify({
                'code': 400,
                'message': '请提供音频文件或Base64编码的音频数据'
            }), 400
        
        upload_folder = app.config.get('UPLOAD_FOLDER', 'uploads')
        os.makedirs(upload_folder, exist_ok=True)
        
        file_id = str(uuid.uuid4())
        
        if audio_file:
            filename = audio_file.filename or ''
            if not allowed_file(filename):
                return jsonify({
                    'code': 400,
                    'message': f'不支持的音频格式，支持：{", ".join(sorted(app.config["ALLOWED_EXTENSIONS"]))}',
                }), 400
            ext = filename.rsplit('.', 1)[1].lower()
            audio_path = os.path.join(upload_folder, f'{file_id}.{ext}')
            audio_file.save(audio_path)
        else:
            if not isinstance(audio_base64, str):
                return jsonify({'code': 400, 'message': 'audio_base64 必须是字符串'}), 400
            encoded_audio = audio_base64.split(',', 1)[-1] if audio_base64.startswith('data:') else audio_base64
            audio_bytes = base64.b64decode(encoded_audio, validate=True)
            audio_path = os.path.join(upload_folder, f'{file_id}.wav')
            with open(audio_path, 'wb') as f:
                f.write(audio_bytes)

        if os.path.getsize(audio_path) < 1024:
            return jsonify({'code': 400, 'message': '音频文件太小或为空'}), 400
        
        language = request.form.get('language') or json_body.get('language') or 'zh'
        supported_languages = {'zh', 'en', 'ja', 'ko', 'fr', 'de', 'es', 'ru', 'ar', 'pt'}
        if language not in supported_languages:
            return jsonify({'code': 400, 'message': f'不支持的语言代码：{language}'}), 400
        enable_compliance = parse_bool(
            request.form.get('enable_compliance') if not request.is_json else json_body.get('enable_compliance'),
            default=True,
        )
        enable_diarization = parse_bool(
            request.form.get('enable_diarization') if not request.is_json else json_body.get('enable_diarization'),
            default=False,
        )
        
        knowledge_items = []
        score_weights = None
        if enable_compliance:
            knowledge_items = KnowledgeBase.query.filter_by(status='active').all()
            db_weights = ScoreWeight.query.all()
            if db_weights:
                score_weights = {w.weight_name: w.weight_value for w in db_weights}
            else:
                score_weights = app.config.get('SCORE_WEIGHTS')
        
        def progress_callback(percent, message):
            print(f'[API转写] {percent}% - {message}')
        
        result = analyze_audio(
            audio_path,
            language=language,
            knowledge_items=knowledge_items,
            score_weights=score_weights,
            progress_callback=progress_callback,
            enable_diarization=enable_diarization,
        )
        
        if result:
            return jsonify({
                'code': 200,
                'message': '成功',
                'data': result
            })
        else:
            return jsonify({
                'code': 500,
                'message': '转写分析失败'
            }), 500
            
    except (binascii.Error, ValueError):
        return jsonify({'code': 400, 'message': 'audio_base64 不是有效的 Base64 数据'}), 400
    except Exception as e:
        print(f'API转写错误: {e}')
        import traceback
        traceback.print_exc()
        return jsonify({
            'code': 500,
            'message': '转写失败，请查看服务器日志'
        }), 500
    finally:
        cleanup_paths = {audio_path, processed_path}
        if audio_path:
            cleanup_paths.add(audio_path.rsplit('.', 1)[0] + '_processed.wav')
        for path in cleanup_paths:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError as cleanup_error:
                    app.logger.warning('无法清理临时音频 %s: %s', path, cleanup_error)


@app.route('/api/v1/health', methods=['GET'])
def api_health():
    """
    健康检查接口
    """
    return jsonify({
        'code': 200,
        'message': '服务正常运行',
        'data': {
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0'
        }
    })


with app.app_context():
    # 确保数据目录与上传目录存在
    for folder in (app.config.get('DATA_FOLDER'), app.config.get('UPLOAD_FOLDER')):
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

    db.create_all()
    # 老数据库不会因 create_all 获得新列，这里补齐（幂等）
    meeting_store.ensure_schema()

    if not ScoreWeight.query.first():
        for name, value in app.config['SCORE_WEIGHTS'].items():
            db.session.add(ScoreWeight(weight_name=name, weight_value=value))
        db.session.commit()

    # 初始化示例知识库（仅在知识库为空时）
    if not KnowledgeBase.query.first():
        db.session.add(KnowledgeBase(
            title='公司会议规范',
            content='所有公司会议必须遵循公司的价值观，尊重每一位参会者的意见，保持积极的工作态度。',
            item_type='policy',
            keywords=json.dumps(['规范', '价值观', '尊重'], ensure_ascii=False),
            required_points=json.dumps([], ensure_ascii=False),
        ))
        db.session.add(KnowledgeBase(
            title='风险词汇表',
            content='会议中应避免使用的负面或消极词汇。',
            item_type='risk_keywords',
            keywords=json.dumps(list(app.config['RISK_KEYWORDS']), ensure_ascii=False),
            required_points=json.dumps([], ensure_ascii=False),
        ))
        db.session.add(KnowledgeBase(
            title='项目例会要点',
            content='项目例会必须包含的要点内容。',
            item_type='key_points',
            keywords=json.dumps(['进度', '问题', '计划', '目标'], ensure_ascii=False),
            required_points=json.dumps(
                ['进度汇报', '问题讨论', '下周计划', '风险说明'], ensure_ascii=False),
        ))
        db.session.commit()

    # 补充金融合规知识库模板（如果不存在）
    if not KnowledgeBase.query.filter_by(title='理财产品销售合规管理办法').first():
        db.session.add(KnowledgeBase(
            title='理财产品销售合规管理办法',
            content=('理财产品销售必须遵守合规要求，包括投资者适当性管理、风险测评、风险告知、'
                     '禁止误导性宣传等。销售人员必须持证上岗，销售过程需录音录像。'),
            item_type='policy',
            keywords=json.dumps(
                ['风险', '销售', '投资者', '必须', '理财', '产品', '合规',
                 '客户', '告知', '测评', '适当性'], ensure_ascii=False),
            required_points=json.dumps([], ensure_ascii=False),
        ))
        db.session.add(KnowledgeBase(
            title='金融销售风险关键词',
            content='金融销售中禁止使用的风险词汇。',
            item_type='risk_keywords',
            keywords=json.dumps(
                ['保本保收益', '零风险', '稳赚不赔', '绝对安全', '保证收益',
                 '高收益无风险', '只赚不赔', '无风险'], ensure_ascii=False),
            required_points=json.dumps([], ensure_ascii=False),
        ))
        db.session.add(KnowledgeBase(
            title='理财销售必传要点',
            content='理财产品销售必须覆盖的合规要点。',
            item_type='key_points',
            keywords=json.dumps(
                ['风险测评', '适当性', '风险告知', '录音录像', '持证上岗',
                 '风险等级', '承受能力', '书面确认', '风险揭示', '合规销售'], ensure_ascii=False),
            required_points=json.dumps(
                ['投资者风险测评', '风险等级匹配', '风险告知义务', '销售过程录音录像',
                 '销售人员持证上岗', '书面确认风险揭示书'], ensure_ascii=False),
        ))
        db.session.commit()

if app.config.get('MAX_CONTENT_LENGTH') is None:
    app.config['MAX_CONTENT_LENGTH'] = 256 * 1024 * 1024

@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['Referrer-Policy'] = 'no-referrer'
    response.headers['Access-Control-Expose-Headers'] = 'Content-Disposition'
    return response

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        'code': 413,
        'message': f'上传文件过大，最大允许 {int(app.config.get("MAX_CONTENT_LENGTH", 0) / 1024 / 1024)}MB'
    }), 413


def preload_models_async():
    """
    在后台线程预加载模型。

    模型加载需要十几秒到几十秒，旧实现放在 socketio.run 之前同步执行，
    期间端口还没监听、页面打不开，看起来像启动失败。改为后台预热，
    服务立刻可用，健康检查接口会实时反映模型就绪状态。
    """
    if not app.config.get('PRELOAD_MODELS', True):
        print('[信息] 跳过模型预加载（PRELOAD_MODELS=false）')
        return

    def _worker():
        try:
            init_asr_models()
        except Exception as exc:
            print(f'[警告] 预加载语音识别模型失败: {exc}，将在首次请求时重试')

        try:
            from modules import model_registry

            model_registry.get_sentence_transformer('zh')
        except Exception as exc:
            print(f'[警告] 预加载句向量模型失败: {exc}，将在首次请求时重试')

        print('[启动] 模型预加载完成')

    threading.Thread(target=_worker, daemon=True).start()


def main():
    """启动服务：后台预热模型 → 立即监听配置的地址与端口。"""
    host = app.config.get('HOST', '0.0.0.0')
    port = int(app.config.get('PORT', 5001))
    ssl_context = None
    if app.config.get('SSL_CERT_FILE') and app.config.get('SSL_KEY_FILE'):
        ssl_context = (app.config['SSL_CERT_FILE'], app.config['SSL_KEY_FILE'])

    display_host = '127.0.0.1' if host in ('0.0.0.0', '::') else host
    scheme = 'https' if ssl_context else 'http'
    print(f'\n服务已启动，请在浏览器打开: {scheme}://{display_host}:{port}\n')

    preload_models_async()

    socketio.run(
        app,
        host=host,
        port=port,
        debug=False,
        allow_unsafe_werkzeug=True,
        ssl_context=ssl_context,
    )


if __name__ == '__main__':
    main()