"""
实时语音转写模块（基于FunASR）
阿里达摩院开源语音识别，中文效果优于Whisper
支持 WebSocket 流式音频传输，边录边转写
流程：实时转写先生成音频文件，再上传文件进行分析
"""
import os
import json
import uuid
import tempfile
import numpy as np
import soundfile as sf
import re
from flask import current_app

def _clean_transcription_text(text):
    """清理转写文本：去除空格、添加标点符号、优化分句"""
    if not text:
        return text
    
    text = text.strip()
    
    text = text.replace(' ', '').replace('\u3000', '').replace('\xa0', '')
    
    comma_before_words = [
        '并且', '而且', '但是', '不过', '然而', '可是', '却', '反而', '倒是',
        '因此', '所以', '因而', '于是', '从而', '以致',
        '因为', '由于', '既然', '要是', '如果', '假如', '倘若', '万一', '除非',
        '虽然', '尽管', '即使', '哪怕', '纵然',
        '只要', '只有', '无论', '不管', '不论',
        '不仅', '不但', '不光', '不只',
        '还是', '或是', '或者',
        '以及', '及', '与', '和', '跟',
        '另外', '此外', '再者', '还有', '同时', '另外还有',
        '第一', '第二', '第三', '第四', '第五', '首先', '其次', '最后', '再次',
        '那么', '这么', '那样', '这样',
        '其实', '事实上', '实际上', '本质上',
        '简单来说', '总而言之', '总的来说', '概括地说',
        '例如', '比如', '比如说', '举例来说',
        '特别是', '尤其是', '值得注意的是',
        '也就是说', '换句话说', '具体而言',
        '一方面', '另一方面',
        '综上所述', '由此可见', '据此',
        '我们认为', '我们觉得', '我们建议', '我们决定', '我们需要',
        '会议认为', '会议决定', '会议建议',
        '有观点认为', '有观点提出',
    ]
    
    for word in comma_before_words:
        text = text.replace(word, '，' + word)
    
    comma_after_words = [
        '的话', '之后', '之前', '以来', '以来的',
        '今天', '明天', '昨天', '前天', '后天',
        '现在', '刚才', '刚刚', '之前', '以后',
        '目前', '当前', '近来', '最近',
        '同时', '与此同时', '一起', '一同',
        '总之', '总而言之', '总之来说',
        '最后', '最终', '终于',
        '结果', '结论', '因此',
        '首先', '其次', '再次', '最后',
        '第一', '第二', '第三',
        '例如', '比如',
        '事实上', '实际上', '其实',
        '另外', '此外',
        '不过', '但是', '然而',
        '因此', '所以',
    ]
    
    for word in comma_after_words:
        text = text.replace(word, word + '，')
    
    text = re.sub(r'([。！？])\s*，', r'\1', text)
    text = re.sub(r'，，+', '，', text)
    
    sentence_end_chars = ['吗', '呢', '吧', '啊', '呀', '嘛', '咯', '哦', '嘿', '诶', '哼', '嗯', '哈']
    for char in sentence_end_chars:
        text = text.replace(char + '？', char + '？')
        if char + '？' not in text:
            text = text.replace(char, char + '？')
    
    period_before_chars = ['了', '的', '吧', '呢', '吗', '啊', '呀', '嘛']
    for char in period_before_chars:
        pattern = re.compile(re.escape(char) + r'(?=[\u4e00-\u9fa5])')
        text = pattern.sub(char + '。', text)
    
    text = re.sub(r'([。！？])+', r'\1', text)
    
    if text and text[-1] not in ['。', '！', '？']:
        text += '。'
    
    text = re.sub(r'([。！？])([^\n])', r'\1\n\2', text)
    
    return text.strip()

transcribers = {}
funasr_model = None


def init_funasr_model():
    """预加载FunASR模型（全局单例）"""
    global funasr_model
    if funasr_model is None:
        try:
            from funasr import AutoModel
            print('[FunASR] 开始预加载模型...')
            funasr_model = AutoModel(
                model="paraformer-zh",
                vad_model="fsmn-vad",
                vad_kwargs={"max_single_segment_time": 30000}
            )
            print('[FunASR] 模型预加载成功')
        except Exception as e:
            print(f'[FunASR] 模型预加载失败: {e}')
    return funasr_model


class FunASRRealtimeTranscriber:
    """
    使用FunASR进行实时语音转写
    """
    def __init__(self, language='zh', knowledge_items=None, audio_file_path=None, score_weights=None):
        self.language = language
        self.knowledge_items = knowledge_items or []
        self.score_weights = score_weights or None
        self.buffer = []
        self.sr = 16000
        self.chunk_duration = 2.0
        self.chunk_samples = int(self.sr * self.chunk_duration)
        self.max_buffer_duration = 8.0
        self.max_buffer_samples = int(self.sr * self.max_buffer_duration)
        self.silence_threshold = 0.015
        self.silence_duration = 0.3
        self.silence_samples = int(self.sr * self.silence_duration)
        self.silence_count = 0
        self.is_speaking = False
        self.transcription_history = []
        self.total_start_time = 0
        
        self.audio_file_path = audio_file_path
        self.all_audio_data = []
        
        self.model = funasr_model
        if self.model is None:
            print('[FunASR] 警告：模型未预加载，尝试实时加载')
            init_funasr_model()
            self.model = funasr_model

    def add_audio_chunk(self, audio_data):
        """
        接收音频块（numpy数组或bytes）
        返回转写结果
        同时保存音频数据用于最终生成音频文件
        """
        try:
            if isinstance(audio_data, bytes):
                if len(audio_data) == 0:
                    return None
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                audio_np = audio_data

            if len(audio_np) == 0:
                return None

            self.buffer.append(audio_np)
            self.all_audio_data.append(audio_np.copy())

            rms = np.sqrt(np.mean(audio_np ** 2))
            if rms > self.silence_threshold:
                self.is_speaking = True
                self.silence_count = 0
            else:
                self.silence_count += len(audio_np)

            total_samples = sum(len(chunk) for chunk in self.buffer)

            should_process = False
            if self.is_speaking and self.silence_count >= self.silence_samples:
                should_process = True
            elif total_samples >= self.max_buffer_samples:
                should_process = True

            if should_process:
                result = self._process_buffer()
                self.is_speaking = False
                self.silence_count = 0
                return result
        except Exception as e:
            print(f'[FunASR] 音频块处理错误: {e}')
        return None

    def _process_buffer(self):
        """处理缓冲区中的音频"""
        if not self.buffer:
            return None

        if self.model is None:
            print('[FunASR] 模型未加载')
            self.buffer = []
            return None

        audio = np.concatenate(self.buffer)
        self.buffer = []

        if len(audio) < int(self.sr * 0.5):
            return None

        try:
            result = self.model.generate(
                input=audio,
                cache={},
                language=self.language,
                use_itn=True,
                batch_size_s=60
            )

            if result and len(result) > 0:
                text = result[0]['text'].strip()
                text = _clean_transcription_text(text)
                if text:
                    segments = [{
                        'text': text,
                        'start': round(self.total_start_time, 2),
                        'end': round(self.total_start_time + len(audio) / self.sr, 2),
                        'confidence': 1.0
                    }]

                    self.transcription_history.append({
                        'text': text,
                        'start_time': self.total_start_time,
                        'end_time': self.total_start_time + len(audio) / self.sr,
                        'confidence': 1.0
                    })

                    self.total_start_time += len(audio) / self.sr

                    return {
                        'text': text,
                        'segments': segments,
                        'is_final': False
                    }
        except Exception as e:
            print(f'[FunASR] 转写错误: {e}')

        return None

    def save_audio_file(self):
        """保存录制的音频为WAV文件"""
        if not self.all_audio_data or not self.audio_file_path:
            return None

        try:
            full_audio = np.concatenate(self.all_audio_data)
            sf.write(self.audio_file_path, full_audio, self.sr)
            print(f'音频文件已保存: {self.audio_file_path}')
            return self.audio_file_path
        except Exception as e:
            print(f"保存音频文件失败: {e}")
            return None

    def get_final_result(self):
        """获取最终完整转写结果"""
        if self.buffer:
            total_samples = sum(len(chunk) for chunk in self.buffer)
            if total_samples >= int(self.sr * 0.3):
                result = self._process_buffer()
            else:
                result = None
        else:
            result = None
            
        full_text = ''.join([seg['text'] for seg in self.transcription_history])
        if result:
            full_text += result['text']

        # 构建 transcriptions 列表供前端分段显示
        transcriptions = []
        for seg in self.transcription_history:
            transcriptions.append({
                'speaker': 'SPEAKER_00',
                'text': seg['text'],
                'start_time': seg['start_time'],
                'end_time': seg['end_time'],
                'confidence': seg.get('confidence', 1.0)
            })

        return {
            'text': full_text.strip(),
            'segments': self.transcription_history.copy(),
            'transcriptions': transcriptions,
            'is_final': True
        }


def register_socketio_events(socketio):
    """注册WebSocket事件"""

    @socketio.on('connect')
    def handle_connect():
        print('Client connected for realtime transcription')

    @socketio.on('disconnect')
    def handle_disconnect():
        print('Client disconnected')

    @socketio.on('start_transcription')
    def handle_start(data):
        from flask import request
        from database import db
        from models import KnowledgeBase, ScoreWeight
        
        sid = request.sid
        language = data.get('language', 'zh')
        enable_compliance = data.get('enable_compliance', True)
        meeting_title = data.get('meeting_title', '实时会议')

        upload_folder = current_app.config.get('UPLOAD_FOLDER', 'uploads')
        file_id = str(uuid.uuid4())
        audio_file_path = os.path.join(upload_folder, f'{file_id}_realtime.wav')

        knowledge_items = []
        score_weights = None
        if enable_compliance:
            with current_app.app_context():
                knowledge_items = KnowledgeBase.query.filter_by(status='active').all()
                db_weights = ScoreWeight.query.all()
                if db_weights:
                    score_weights = {}
                    for w in db_weights:
                        score_weights[w.weight_name] = w.weight_value
                else:
                    score_weights = current_app.config.get('SCORE_WEIGHTS')

        transcribers[sid] = FunASRRealtimeTranscriber(
            language=language,
            knowledge_items=knowledge_items,
            audio_file_path=audio_file_path,
            score_weights=score_weights
        )

        socketio.emit('transcription_started', {
            'language': language,
            'compliance_enabled': enable_compliance,
            'meeting_title': meeting_title
        }, to=sid)
        print(f'Transcription started for session {sid}, audio file: {audio_file_path}')

    @socketio.on('audio_chunk')
    def handle_audio_chunk(data):
        from flask import request
        import base64
        
        sid = request.sid

        if sid not in transcribers:
            socketio.emit('error', {'message': '请先开始转写会话'}, to=sid)
            return

        try:
            if isinstance(data, dict) and 'audio' in data:
                audio_bytes = base64.b64decode(data['audio'])
            elif isinstance(data, str):
                audio_bytes = base64.b64decode(data)
            else:
                audio_bytes = data

            transcriber = transcribers[sid]
            result = transcriber.add_audio_chunk(audio_bytes)

            if result:
                socketio.emit('transcription_result', result, to=sid)

        except Exception as e:
            print(f'Audio chunk error: {e}')
            socketio.emit('error', {'message': f'音频处理错误: {str(e)}'}, to=sid)

    @socketio.on('stop_transcription')
    def handle_stop():
        from flask import request
        import threading
        
        sid = request.sid
        print(f'[停止转写] 收到停止请求，sid={sid}')

        try:
            if sid in transcribers:
                transcriber = transcribers[sid]
                print(f'[停止转写] 找到转写器，audio_file_path={transcriber.audio_file_path}')
                
                try:
                    audio_file_path = transcriber.save_audio_file()
                    print(f'[停止转写] 音频文件保存: {audio_file_path}')
                except Exception as e:
                    print(f'[停止转写] 保存音频文件失败: {e}')
                    audio_file_path = None
                
                try:
                    result = transcriber.get_final_result()
                    print(f'[停止转写] 最终转写结果: text长度={len(result.get("text", ""))}, segments={len(result.get("segments", []))}')
                except Exception as e:
                    print(f'[停止转写] 获取最终结果失败: {e}')
                    result = {'text': '', 'segments': []}

                result['audio_file'] = audio_file_path
                result['audio_duration'] = len(transcriber.all_audio_data) * 0.1
                print(f'[停止转写] 音频时长: {result["audio_duration"]}')

                try:
                    socketio.emit('transcription_stopped', {
                        'audio_file': audio_file_path,
                        'audio_duration': result['audio_duration'],
                        'text': result.get('text', '')
                    }, to=sid)
                    print(f'[停止转写] 已发送 transcription_stopped 事件')
                except Exception as e:
                    print(f'[停止转写] 发送事件失败: {e}')

                def _analyze_async():
                    try:
                        print(f'[分析线程] 开始执行分析，audio_file_path={audio_file_path}')
                        from modules.analysis_pipeline import analyze_audio
                        from modules.realtime_transcriber import _save_meeting_to_db
                        from app import app as flask_app

                        def progress_callback(percent, message):
                            try:
                                socketio.emit('realtime_analysis_progress', {
                                    'progress': percent,
                                    'message': message
                                }, to=sid)
                                print(f'[分析线程] 推送进度: {percent}% - {message}')
                            except Exception as e:
                                print(f'推送进度失败: {e}')

                        full_analysis = analyze_audio(
                        audio_file_path,
                        language=transcriber.language,
                        knowledge_items=transcriber.knowledge_items,
                        score_weights=transcriber.score_weights,
                        progress_callback=progress_callback,
                        transcription_text=result.get('text', ''),
                        transcriptions=result.get('transcriptions', [])
                    )
                        print(f'[分析线程] 分析完成，full_analysis={full_analysis is not None}')

                        if full_analysis:
                            result.update(full_analysis)
                            if 'compliance' in full_analysis and full_analysis['compliance']:
                                result['compliance_report'] = full_analysis['compliance']

                        if 'audio_duration' in result:
                            result['duration'] = int(result['audio_duration'])

                        with flask_app.app_context():
                            meeting_id = _save_meeting_to_db(result, audio_file_path, transcriber.knowledge_items)
                            print(f'[分析线程] 保存会议到数据库: meeting_id={meeting_id}')
                        
                        if meeting_id:
                            result['meeting_id'] = meeting_id
                            result['id'] = meeting_id

                        socketio.emit('transcription_final', result, to=sid)
                        print(f'[分析线程] 已发送 transcription_final 事件，meeting_id={meeting_id}')

                    except Exception as e:
                        print(f'Analysis thread error: {e}')
                        import traceback
                        traceback.print_exc()
                        socketio.emit('realtime_analysis_progress', {
                            'progress': -1,
                            'message': f'分析失败: {str(e)}'
                        }, to=sid)
                    finally:
                        if sid in transcribers:
                            del transcribers[sid]

                threading.Thread(target=_analyze_async, daemon=True).start()
                print(f'Transcription stopped for session {sid}, analysis started in background')
            
            else:
                socketio.emit('error', {'message': '没有活跃的转写会话'}, to=sid)
        except Exception as e:
            print(f'[停止转写] 主流程异常: {e}')
            import traceback
            traceback.print_exc()
