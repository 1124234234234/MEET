"""
实时语音转写模块
支持 WebSocket 流式音频传输，边录边转写
支持实时合规比对
"""
import os
import json
import uuid
import tempfile
import numpy as np
import librosa
import soundfile as sf
from flask import current_app


class RealtimeTranscriber:
    """实时转写器：接收音频流，分段转写，实时合规检查，同时保存音频文件"""

    def __init__(self, whisper_model, language='zh', knowledge_items=None, audio_file_path=None):
        self.whisper_model = whisper_model
        self.language = language
        self.knowledge_items = knowledge_items or []
        self.buffer = []
        self.sr = 16000
        self.chunk_duration = 5.0  # 每5秒处理一次
        self.chunk_samples = int(self.sr * self.chunk_duration)
        self.min_audio_length = 1.0  # 最小处理长度1秒
        self.transcription_history = []
        self.compliance_history = []  # 合规检查历史
        self.total_start_time = 0  # 累计开始时间
        
        # 音频文件保存相关
        self.audio_file_path = audio_file_path
        self.audio_writer = None
        self.all_audio_data = []  # 保存所有原始音频数据用于最终保存

    def set_knowledge_items(self, knowledge_items):
        """设置知识库条目（用于实时合规检查）"""
        self.knowledge_items = knowledge_items

    def add_audio_chunk(self, audio_data):
        """
        接收音频块（numpy数组或bytes）
        返回转写结果和合规检查结果（如果有）
        同时保存音频数据用于最终生成音频文件
        """
        if isinstance(audio_data, bytes):
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            audio_np = audio_data

        self.buffer.append(audio_np)
        
        # 保存音频数据用于最终生成文件
        self.all_audio_data.append(audio_np.copy())

        # 检查是否积累足够的音频
        total_samples = sum(len(chunk) for chunk in self.buffer)
        if total_samples >= self.chunk_samples:
            return self._process_buffer()
        return None
    
    def save_audio_file(self):
        """保存录制的音频为WAV文件"""
        if not self.all_audio_data or not self.audio_file_path:
            return None
        
        try:
            # 合并所有音频数据
            full_audio = np.concatenate(self.all_audio_data)
            
            # 确保目录存在
            os.makedirs(os.path.dirname(self.audio_file_path), exist_ok=True)
            
            # 保存为WAV文件
            sf.write(self.audio_file_path, full_audio, self.sr)
            print(f"音频文件已保存: {self.audio_file_path}")
            return self.audio_file_path
        except Exception as e:
            print(f"保存音频文件失败: {e}")
            return None

    def _process_buffer(self):
        """处理缓冲区中的音频"""
        if not self.buffer:
            return None

        # 合并音频
        audio = np.concatenate(self.buffer)
        self.buffer = []

        # 检查长度
        if len(audio) < int(self.sr * self.min_audio_length):
            return None

        # 写入临时文件
        temp_path = tempfile.mktemp(suffix='.wav')
        try:
            sf.write(temp_path, audio, self.sr)

            # Whisper转写（使用繁简修复）
            from modules.whisper_utils import transcribe_with_fix
            result = transcribe_with_fix(
                self.whisper_model,
                temp_path,
                language=self.language,
                task='transcribe'
            )

            text = result['text'].strip()
            if text:
                segments = []
                for seg in result['segments']:
                    # 调整时间戳（加上累计开始时间）
                    adjusted_start = self.total_start_time + seg['start']
                    adjusted_end = self.total_start_time + seg['end']
                    
                    segments.append({
                        'text': seg['text'],
                        'start': round(adjusted_start, 2),
                        'end': round(adjusted_end, 2),
                        'confidence': round(seg.get('confidence', 0), 4)
                    })
                    self.transcription_history.append({
                        'text': seg['text'],
                        'start_time': adjusted_start,
                        'end_time': adjusted_end,
                        'confidence': seg.get('confidence', 0)
                    })

                # 更新累计开始时间
                self.total_start_time += len(audio) / self.sr

                # 实时合规检查
                compliance_result = None
                if self.knowledge_items:
                    from modules.compliance_checker import realtime_compliance_check
                    
                    for seg in segments:
                        check_result = realtime_compliance_check(
                            seg['text'],
                            self.knowledge_items,
                            seg['start'],
                            seg['end']
                        )
                        if check_result['has_risk'] or check_result['covered_points']:
                            compliance_result = check_result
                            self.compliance_history.append(check_result)

                return {
                    'text': text,
                    'segments': segments,
                    'compliance': compliance_result,
                    'is_final': False
                }

            return None

        except Exception as e:
            print(f"实时转写错误: {e}")
            return None
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def get_final_result(self):
        """获取最终完整转写结果和合规报告"""
        # 处理剩余缓冲区
        result = self._process_buffer() if self.buffer else None

        full_text = ' '.join([seg['text'] for seg in self.transcription_history])
        if result:
            full_text += ' ' + result['text']

        # 计算最终合规评分
        final_compliance = None
        if self.knowledge_items and self.transcription_history:
            from modules.compliance_checker import calculate_compliance_score, get_score_level
            
            compliance_result = calculate_compliance_score(
                full_text,
                self.knowledge_items,
                transcription_segments=self.transcription_history
            )
            
            final_compliance = {
                'total_score': compliance_result['total_score'],
                'score_level': get_score_level(compliance_result['total_score']),
                'components': compliance_result['components'],
                'covered_points': compliance_result['covered_points'],
                'point_time_markers': compliance_result.get('point_time_markers', []),
                'missing_points': compliance_result['missing_points'],
                'risk_keywords_found': compliance_result['risk_keywords_found'],
                'risk_time_markers': compliance_result.get('risk_time_markers', []),
                'matched_keywords': compliance_result['matched_keywords'],
                'suggestions': compliance_result['suggestions']
            }

        return {
            'text': full_text.strip(),
            'segments': self.transcription_history.copy(),
            'compliance': final_compliance,
            'compliance_history': self.compliance_history.copy(),
            'is_final': True
        }

    def reset(self):
        """重置转写器"""
        self.buffer = []
        self.transcription_history = []
        self.compliance_history = []
        self.total_start_time = 0


# WebSocket事件处理
def register_socketio_events(socketio, whisper_model):
    """注册WebSocket事件"""

    transcribers = {}

    @socketio.on('connect')
    def handle_connect():
        print('Client connected for realtime transcription')
        socketio.emit('connected', {'status': 'connected'})

    @socketio.on('disconnect')
    def handle_disconnect():
        print('Client disconnected')
        # 清理转写器
        for sid in list(transcribers.keys()):
            if sid not in transcribers:
                del transcribers[sid]

    @socketio.on('start_transcription')
    def handle_start(data):
        """开始实时转写会话"""
        from flask import request
        from models import KnowledgeBase
        
        sid = request.sid

        language = data.get('language', 'zh')
        enable_compliance = data.get('enable_compliance', True)
        meeting_title = data.get('meeting_title', '实时会议')
        
        # 生成音频文件路径
        upload_folder = current_app.config.get('UPLOAD_FOLDER', 'uploads')
        file_id = str(uuid.uuid4())
        audio_file_path = os.path.join(upload_folder, f'{file_id}_realtime.wav')
        
        # 获取知识库条目
        knowledge_items = []
        if enable_compliance:
            from database import db
            with current_app.app_context():
                knowledge_items = KnowledgeBase.query.filter_by(status='active').all()
        
        transcribers[sid] = RealtimeTranscriber(
            whisper_model, 
            language, 
            knowledge_items,
            audio_file_path=audio_file_path
        )

        socketio.emit('transcription_started', {
            'status': 'started',
            'language': language,
            'compliance_enabled': enable_compliance,
            'meeting_title': meeting_title
        }, to=sid)
        print(f'Transcription started for session {sid}, audio file: {audio_file_path}')

    @socketio.on('audio_chunk')
    def handle_audio_chunk(data):
        """接收音频块并实时转写"""
        from flask import request
        sid = request.sid

        if sid not in transcribers:
            socketio.emit('error', {'message': '请先开始转写会话'}, to=sid)
            return

        import base64
        try:
            # 解码base64音频数据
            if isinstance(data, dict) and 'audio' in data:
                audio_bytes = base64.b64decode(data['audio'])
            elif isinstance(data, str):
                audio_bytes = base64.b64decode(data)
            else:
                audio_bytes = data

            transcriber = transcribers[sid]
            result = transcriber.add_audio_chunk(audio_bytes)

            if result:
                # 发送转写结果
                socketio.emit('transcription_result', result, to=sid)
                
                # 如果有合规告警，发送告警
                if result.get('compliance') and result['compliance'].get('alerts'):
                    socketio.emit('compliance_alert', result['compliance'], to=sid)

        except Exception as e:
            print(f'Audio chunk error: {e}')
            socketio.emit('error', {'message': f'音频处理错误: {str(e)}'}, to=sid)

    @socketio.on('stop_transcription')
    def handle_stop():
        """停止转写，保存音频文件，进行分析，并保存会议记录到数据库"""
        from flask import request
        sid = request.sid

        if sid in transcribers:
            transcriber = transcribers[sid]
            
            # 1. 保存音频文件
            audio_file_path = transcriber.save_audio_file()
            
            # 2. 获取最终转写结果
            result = transcriber.get_final_result()
            
            # 3. 快速分析（跳过耗时步骤）
            if audio_file_path:
                full_analysis = _perform_fast_analysis(
                    audio_file_path,
                    transcriber.language,
                    transcriber.knowledge_items
                )
                if full_analysis:
                    result.update(full_analysis)
            
            # 4. 添加音频文件信息
            result['audio_file'] = audio_file_path
            result['audio_duration'] = len(transcriber.all_audio_data) * 0.1  # 估算时长
            
            # 5. 保存会议记录到数据库
            meeting_id = _save_meeting_to_db(result, audio_file_path, transcriber.knowledge_items)
            if meeting_id:
                result['meeting_id'] = meeting_id
            
            socketio.emit('transcription_final', result, to=sid)
            del transcribers[sid]
            print(f'Transcription finished for session {sid}, audio saved to {audio_file_path}, meeting_id={meeting_id}')
        else:
            socketio.emit('error', {'message': '没有活跃的转写会话'}, to=sid)


def _perform_full_analysis(audio_path, language, knowledge_items):
    """
    对录制的音频进行完整分析
    包括：音频预处理、说话人分离、关键词提取、主题分析、摘要生成等
    """
    try:
        from modules.audio_preprocessor import preprocess_audio, get_audio_quality_report
        from modules.speaker_diarization import speaker_diarization_simple
        from modules.text_analyzer import extract_keywords, analyze_topic, generate_summary, analyze_sentiment
        from modules.compliance_checker import calculate_compliance_score, get_score_level
        
        # 音频预处理
        processed_path = audio_path.replace('.wav', '_processed.wav')
        try:
            preprocess_audio(audio_path, processed_path)
        except Exception:
            processed_path = audio_path
        
        # 重新转写（使用完整音频，提高准确性）
        from app import whisper_model
        full_result = whisper_model.transcribe(
            processed_path,
            language=language,
            initial_prompt='以下是简体中文的语音转写内容，请使用简体中文输出。'
        )
        full_text = full_result['text']
        
        # 说话人分离
        speaker_segments = []
        try:
            speaker_segments = speaker_diarization_simple(processed_path)
        except Exception as e:
            print(f'Speaker diarization skipped: {e}')
        
        # 生成带说话人的转写段落
        transcriptions = []
        for seg in full_result['segments']:
            speaker = 'SPEAKER_00'
            if speaker_segments:
                for ss in speaker_segments:
                    if ss['start'] <= seg['start'] <= ss['end']:
                        speaker = ss['speaker']
                        break
            transcriptions.append({
                'speaker': speaker,
                'text': seg['text'],
                'start_time': seg['start'],
                'end_time': seg['end'],
                'confidence': seg.get('confidence', 0)
            })
        
        # 文本分析
        keywords = extract_keywords(full_text, top_n=10)
        topics = analyze_topic(full_text)
        summary = generate_summary(full_text, max_length=300)
        sentiment = analyze_sentiment(full_text)
        
        # 音频质量报告
        audio_quality = None
        try:
            audio_quality = get_audio_quality_report(audio_path, processed_path)
        except Exception as e:
            print(f'Audio quality report skipped: {e}')
        
        # 合规分析
        compliance_result = None
        if knowledge_items:
            compliance_result = calculate_compliance_score(
                full_text,
                knowledge_items,
                transcription_segments=transcriptions
            )
            compliance_result['score_level'] = get_score_level(compliance_result['total_score'])
        
        return {
            'transcriptions': transcriptions,
            'keywords': keywords,
            'topics': topics,
            'summary': summary,
            'sentiment': sentiment,
            'audio_quality': audio_quality,
            'compliance': compliance_result
        }
    
    except Exception as e:
        print(f'Full analysis failed: {e}')
        import traceback
        traceback.print_exc()
        return None


def _perform_fast_analysis(audio_path, language, knowledge_items):
    """
    快速分析（跳过耗时步骤）
    不做音频预处理和说话人分离，直接使用原始音频转写
    """
    try:
        from modules.text_analyzer import extract_keywords, analyze_topic, generate_summary, analyze_sentiment
        from modules.compliance_checker import calculate_compliance_score, get_score_level

        # 直接转写（不做音频预处理，使用繁简修复）
        from app import whisper_model
        from modules.whisper_utils import transcribe_with_fix
        full_result = transcribe_with_fix(whisper_model, audio_path, language=language)
        full_text = full_result['text']

        # 生成转写段落（不做说话人分离）
        transcriptions = []
        for seg in full_result['segments']:
            transcriptions.append({
                'speaker': 'SPEAKER_00',
                'text': seg['text'],
                'start_time': seg['start'],
                'end_time': seg['end'],
                'confidence': seg.get('confidence', 0)
            })

        # 文本分析（关键词、主题、摘要、情绪）
        keywords = extract_keywords(full_text, top_n=10)
        topics = analyze_topic(full_text)
        summary = generate_summary(full_text, max_length=300)
        sentiment = analyze_sentiment(full_text)

        # 合规分析（如果有知识库）
        compliance_result = None
        if knowledge_items:
            compliance_result = calculate_compliance_score(
                full_text,
                knowledge_items,
                transcription_segments=transcriptions
            )
            compliance_result['score_level'] = get_score_level(compliance_result['total_score'])

        return {
            'transcriptions': transcriptions,
            'keywords': keywords,
            'topics': topics,
            'summary': summary,
            'sentiment': sentiment,
            'compliance': compliance_result,
            'full_text': full_text
        }

    except Exception as e:
        print(f'Fast analysis failed: {e}')
        import traceback
        traceback.print_exc()
        return None


def _save_meeting_to_db(result, audio_file_path, knowledge_items):
    """
    将会议记录保存到数据库
    """
    try:
        from app import db
        from models import Meeting, Transcription, ComplianceReport
        from modules.compliance_checker import get_score_level
        import json

        # 创建会议记录
        meeting = Meeting(
            title=f'实时转写会议 {result.get("meeting_id", "")}',
            audio_path=audio_file_path,
            duration=int(result.get('audio_duration', 0)),
            status='finished',
            summary=result.get('summary', ''),
            keywords=json.dumps(result.get('keywords', [])),
            topics=json.dumps(result.get('topics', [])),
            sentiment=json.dumps(result.get('sentiment', {}))
        )
        db.session.add(meeting)
        db.session.commit()

        # 保存转写段落
        transcriptions = result.get('transcriptions', [])
        if transcriptions:
            for seg in transcriptions:
                transcription = Transcription(
                    meeting_id=meeting.id,
                    speaker=seg.get('speaker', 'SPEAKER_00'),
                    text=seg.get('text', ''),
                    start_time=seg.get('start_time', 0),
                    end_time=seg.get('end_time', 0),
                    confidence=seg.get('confidence', 0.0),
                    language='zh'
                )
                db.session.add(transcription)
            db.session.commit()

        # 保存合规报告
        compliance = result.get('compliance')
        if compliance:
            report = ComplianceReport(
                meeting_id=meeting.id,
                total_score=compliance.get('total_score', 0),
                score_level=compliance.get('score_level', ''),
                detailed_scores=json.dumps(compliance.get('components', {})),
                missing_points=json.dumps(compliance.get('missing_points', [])),
                risk_keywords=json.dumps(compliance.get('risk_keywords_found', [])),
                risk_time_markers=json.dumps(compliance.get('risk_time_markers', [])),
                point_time_markers=json.dumps(compliance.get('point_time_markers', [])),
                matched_keywords=json.dumps(compliance.get('matched_keywords', [])),
                suggestions=json.dumps(compliance.get('suggestions', []))
            )
            db.session.add(report)
            meeting.total_score = compliance.get('total_score', 0)
            meeting.score_level = compliance.get('score_level', '')
            db.session.commit()

        return meeting.id

    except Exception as e:
        print(f'Save meeting to DB failed: {e}')
        import traceback
        traceback.print_exc()
        return None