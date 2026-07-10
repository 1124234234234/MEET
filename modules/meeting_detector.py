"""
会议状态检测模块
基于音频活跃度自动判断会议启停状态
统计参会人员数量
"""
import numpy as np
import librosa


class MeetingDetector:
    """会议状态检测器：通过音频活跃度自动检测会议开始和结束"""

    def __init__(self, sample_rate=16000):
        self.sr = sample_rate
        self.silence_threshold = 0.01  
        self.speech_threshold = 0.03  
        self.min_speech_duration = 3.0  
        self.min_silence_duration = 60.0  
        self.min_meeting_duration = 60.0  
        
        self.is_meeting_active = False
        self.current_speech_duration = 0.0
        self.current_silence_duration = 0.0
        self.meeting_start_time = None
        self.meeting_end_time = None
        self.speech_segments = []
        self.current_speech_start = None

    def analyze_audio_chunk(self, audio_data, timestamp):
        """
        分析音频块，检测会议状态变化
        返回: {'event': 'start'|'end'|'ongoing'|None, 'timestamp': float}
        """
        if isinstance(audio_data, bytes):
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            audio_np = audio_data

        rms = librosa.feature.rms(y=audio_np)[0].mean()
        is_speech = rms > self.speech_threshold
        chunk_duration = len(audio_np) / self.sr

        result = None

        if is_speech:
            self.current_silence_duration = 0.0
            self.current_speech_duration += chunk_duration
            
            if self.current_speech_start is None:
                self.current_speech_start = timestamp

            if not self.is_meeting_active:
                if self.current_speech_duration >= self.min_speech_duration:
                    self.is_meeting_active = True
                    self.meeting_start_time = timestamp - self.current_speech_duration
                    self.current_speech_start = None
                    result = {
                        'event': 'start',
                        'timestamp': self.meeting_start_time,
                        'confidence': min(1.0, self.current_speech_duration / 10.0)
                    }
        else:
            self.current_speech_duration = 0.0
            self.current_silence_duration += chunk_duration
            
            if self.current_speech_start is not None:
                self.speech_segments.append({
                    'start': self.current_speech_start,
                    'end': timestamp
                })
                self.current_speech_start = None

            if self.is_meeting_active:
                if self.current_silence_duration >= self.min_silence_duration:
                    meeting_duration = timestamp - self.meeting_start_time
                    if meeting_duration >= self.min_meeting_duration:
                        self.is_meeting_active = False
                        self.meeting_end_time = timestamp - chunk_duration
                        result = {
                            'event': 'end',
                            'timestamp': self.meeting_end_time,
                            'duration': meeting_duration
                        }

        return result

    def get_meeting_status(self):
        """获取当前会议状态"""
        return {
            'is_active': self.is_meeting_active,
            'start_time': self.meeting_start_time,
            'end_time': self.meeting_end_time,
            'duration': self.meeting_end_time - self.meeting_start_time if self.meeting_end_time and self.meeting_start_time else None,
            'speech_segments_count': len(self.speech_segments),
            'speech_segments': self.speech_segments
        }

    def reset(self):
        """重置检测器"""
        self.is_meeting_active = False
        self.current_speech_duration = 0.0
        self.current_silence_duration = 0.0
        self.meeting_start_time = None
        self.meeting_end_time = None
        self.speech_segments = []
        self.current_speech_start = None


def count_participants(speaker_segments):
    """
    根据说话人分离结果统计参会人员数量
    """
    if not speaker_segments:
        return 0
    
    speakers = set()
    for seg in speaker_segments:
        speaker = seg.get('speaker', 'unknown')
        speakers.add(speaker)
    
    return len(speakers)


def analyze_participation_distribution(speaker_segments):
    """
    分析参会人员发言分布
    """
    if not speaker_segments:
        return {}
    
    speaker_stats = {}
    for seg in speaker_segments:
        speaker = seg.get('speaker', 'unknown')
        duration = seg.get('end', 0) - seg.get('start', 0)
        
        if speaker not in speaker_stats:
            speaker_stats[speaker] = {
                'duration': 0,
                'count': 0,
                'segments': []
            }
        
        speaker_stats[speaker]['duration'] += duration
        speaker_stats[speaker]['count'] += 1
        speaker_stats[speaker]['segments'].append(seg)
    
    total_duration = sum(stats['duration'] for stats in speaker_stats.values())
    for speaker, stats in speaker_stats.items():
        stats['percentage'] = (stats['duration'] / total_duration * 100) if total_duration > 0 else 0
    
    return speaker_stats