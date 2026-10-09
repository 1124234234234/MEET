"""本地 voice-api 服务的 Whisper/FunASR 兼容客户端。"""

import mimetypes
import os
import tempfile

import numpy as np
import requests
import soundfile as sf

try:
    from modules.whisper_utils import fix_traditional_chinese
except ImportError:
    try:
        from whisper_utils import fix_traditional_chinese
    except ImportError:
        def fix_traditional_chinese(text):
            return text


class VoiceAPIError(RuntimeError):
    """voice-api 不可用或返回非法响应。"""


class VoiceAPIWhisperClient:
    """提供与 Whisper ``model.transcribe`` 兼容的远程客户端。"""

    def __init__(self, base_url, timeout=1800):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout

    def check_health(self):
        try:
            response = requests.get(f'{self.base_url}/health', timeout=10)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise VoiceAPIError(f'voice-api 健康检查失败：{exc}') from exc
        if payload.get('status') != 'ok':
            raise VoiceAPIError(f'voice-api 状态异常：{payload}')
        return payload

    def transcribe(self, audio_path, language='zh', **_kwargs):
        mime_type = mimetypes.guess_type(audio_path)[0] or 'application/octet-stream'
        try:
            with open(audio_path, 'rb') as audio_file:
                response = requests.post(
                    f'{self.base_url}/transcribe',
                    files={
                        'audio': (
                            os.path.basename(audio_path),
                            audio_file,
                            mime_type,
                        )
                    },
                    data={'target_language': language},
                    timeout=self.timeout,
                )
            response.raise_for_status()
            payload = response.json()
        except (OSError, requests.RequestException, ValueError) as exc:
            raise VoiceAPIError(f'voice-api 转写失败：{exc}') from exc

        if payload.get('error'):
            raise VoiceAPIError(f'voice-api 转写失败：{payload["error"]}')

        text = payload.get('translated_text') or payload.get('transcribed_text') or ''
        text = fix_traditional_chinese(text.strip())
        duration = _audio_duration(audio_path)
        segments = []
        if text:
            segments.append({
                'text': text,
                'start': 0.0,
                'end': duration,
                'confidence': float(payload.get('confidence') or 0.0),
            })

        return {
            'text': text,
            'segments': segments,
            'language': payload.get('target_language') or payload.get('detected_language') or language,
            'detected_language': payload.get('detected_language'),
        }


class VoiceAPIFunASRAdapter:
    """提供 FunASR ``AutoModel.generate`` 兼容接口，用于实时音频块。"""

    def __init__(self, base_url, timeout=1800):
        self.client = VoiceAPIWhisperClient(base_url, timeout=timeout)

    def check_health(self):
        return self.client.check_health()

    def generate(self, input, language='zh', **_kwargs):
        audio = np.asarray(input, dtype=np.float32)
        if audio.size == 0:
            return []

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
            sf.write(temp_path, audio, 16000)
            result = self.client.transcribe(temp_path, language=language)
            return [{'text': result.get('text', '')}]
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)


def _audio_duration(audio_path):
    try:
        return round(float(sf.info(audio_path).duration), 3)
    except Exception:
        try:
            import librosa

            return round(float(librosa.get_duration(path=audio_path)), 3)
        except Exception:
            return 0.0
