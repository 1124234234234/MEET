"""
统一的音频读写工具。

集中处理「解码 + 重采样 + 单声道」，供语音识别、说话人分离、音频预处理共享。
优先使用 soundfile（wav/flac/ogg），遇到 mp3/m4a 等格式时回退到 imageio-ffmpeg，
避免依赖系统 PATH 中的 ffmpeg 可执行文件。
"""
import os
import subprocess
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

DEFAULT_SR = 16000


def resample(y, orig_sr, target_sr):
    """多相滤波重采样，纯 scipy 实现，不依赖 librosa/numba。"""
    if orig_sr == target_sr:
        return np.asarray(y, dtype=np.float32)
    g = gcd(int(orig_sr), int(target_sr))
    up = int(target_sr // g)
    down = int(orig_sr // g)
    return resample_poly(np.asarray(y, dtype=np.float32), up, down).astype(np.float32)


def _decode_with_ffmpeg(path, sr):
    import imageio_ffmpeg

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe, '-nostdin', '-threads', '0',
        '-i', path, '-f', 's16le', '-ac', '1',
        '-acodec', 'pcm_s16le', '-ar', str(sr), '-',
    ]
    result = subprocess.run(cmd, capture_output=True, check=True)
    audio = np.frombuffer(result.stdout, dtype=np.int16)
    return audio.astype(np.float32) / 32768.0


def load_audio(path, sr=DEFAULT_SR):
    """
    读取音频为指定采样率的单声道 float32 数组。

    先尝试 soundfile，失败则用内置 ffmpeg 解码；两者都失败时抛 RuntimeError。
    """
    if not path or not os.path.exists(path):
        raise RuntimeError(f'音频文件不存在: {path}')

    try:
        y, orig_sr = sf.read(path, always_2d=False, dtype='float32')
        y = np.asarray(y, dtype=np.float32)
        if y.ndim > 1:
            y = y.mean(axis=1)
        return resample(y, orig_sr, sr)
    except Exception:
        pass

    try:
        return _decode_with_ffmpeg(path, sr)
    except Exception as exc:
        raise RuntimeError(f'无法加载音频文件 {path}: {exc}') from exc


def save_wav(path, y, sr=DEFAULT_SR):
    """写出 16bit PCM wav，自动创建目录。"""
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    sf.write(path, np.asarray(y, dtype=np.float32), sr, subtype='PCM_16')
    return path


def get_duration(path, sr=DEFAULT_SR):
    """返回音频时长（秒），失败时返回 0.0。"""
    try:
        info = sf.info(path)
        return float(info.frames) / float(info.samplerate)
    except Exception:
        pass
    try:
        y = load_audio(path, sr=sr)
        return len(y) / float(sr)
    except Exception:
        return 0.0
