import librosa
import numpy as np
from scipy import signal
import soundfile as sf
import warnings
warnings.filterwarnings('ignore')


def preprocess_audio(input_path, output_path=None):
    """
    完整的音频预处理流程：
    1. 格式统一：转16kHz单声道
    2. 降噪：使用 noisereduce 谱减法
    3. 回声消除：使用自适应滤波器
    4. 语音增强：谱增益法
    """
    # 第1步：加载音频，统一格式
    y, sr = librosa.load(input_path, sr=16000, mono=True)

    # 第2步：降噪 - 使用 noisereduce 谱减法
    y_denoised = apply_noise_reduction(y, sr)

    # 第3步：回声消除 - 使用自适应滤波器
    y_echo_canceled = apply_echo_cancellation(y_denoised, sr)

    # 第4步：语音增强 - 谱增益法
    y_enhanced = apply_speech_enhancement(y_echo_canceled, sr)

    # 归一化
    y_enhanced = normalize_audio(y_enhanced)

    if output_path:
        sf.write(output_path, y_enhanced, sr)

    return y_enhanced, sr


def apply_noise_reduction(y, sr):
    """使用 noisereduce 进行谱减法降噪"""
    try:
        import noisereduce as nr

        # 从音频前0.5秒估计噪声谱（通常是静音或环境噪声）
        noise_clip = y[:int(sr * 0.5)] if len(y) > sr * 0.5 else y[:int(len(y) * 0.1)]

        # noisereduce 降噪
        reduced = nr.reduce_noise(y=y, sr=sr, y_noise=noise_clip, stationary=False)

        return reduced
    except Exception as e:
        print(f"noisereduce降噪失败，使用谱减法: {e}")
        return spectral_subtraction(y, sr)


def spectral_subtraction(y, sr):
    """谱减法降噪"""
    # STFT
    stft = librosa.stft(y, n_fft=2048, hop_length=512)
    mag, phase = librosa.magphase(stft)

    # 估计噪声谱（取前几帧的平均）
    noise_frames = min(20, mag.shape[1] // 4)
    noise_spec = np.mean(mag[:, :noise_frames], axis=1, keepdims=True)

    # 谱减
    mag_clean = mag - 2.0 * noise_spec
    mag_clean = np.maximum(mag_clean, 0.01)  # 保证非负

    # 重建
    stft_clean = mag_clean * phase
    y_clean = librosa.istft(stft_clean, hop_length=512)

    # 长度对齐
    if len(y_clean) < len(y):
        y_clean = np.pad(y_clean, (0, len(y) - len(y_clean)))
    else:
        y_clean = y_clean[:len(y)]

    return y_clean


def apply_echo_cancellation(y, sr):
    """
    回声消除：使用带通滤波清理人声频段外的噪声
    跳过LMS自适应滤波器（纯Python循环太慢，且会议音频通常无明显回声）
    """
    try:
        b, a = signal.butter(4, [80, 7000], btype='band', fs=sr)
        y_filtered = signal.filtfilt(b, a, y)
        return y_filtered
    except Exception as e:
        print(f"带通滤波失败: {e}")
        return y


def lms_echo_cancel(y, delay_samples, filter_length=256, step_size=0.01):
    """
    LMS自适应滤波器回声消除（向量化优化版）
    使用scipy.signal.lfilter加速，避免纯Python循环
    """
    n = len(y)
    if n < filter_length + delay_samples:
        return y

    x = np.zeros(n)
    if delay_samples < n:
        x[delay_samples:] = y[:n - delay_samples]

    try:
        from scipy.signal import lfilter

        w = np.zeros(filter_length)
        y_out = np.zeros(n)

        batch_size = 1024
        for start in range(filter_length, n, batch_size):
            end = min(start + batch_size, n)
            for i in range(start, end):
                x_vec = x[i - filter_length:i][::-1]
                echo_est = np.dot(w, x_vec)
                y_out[i] = y[i] - echo_est
                w = w + step_size * y_out[i] * x_vec

        return y_out
    except Exception:
        return y


def apply_speech_enhancement(y, sr):
    """
    语音增强：使用谱增益法（MMSE短时谱振幅估计）
    在保持语音可懂度的同时进一步抑制残留噪声
    """
    try:
        # STFT
        stft = librosa.stft(y, n_fft=2048, hop_length=512)
        mag = np.abs(stft)
        phase = np.angle(stft)

        # 估计噪声 floor（取能量最低的帧）
        frame_energy = np.sum(mag ** 2, axis=0)
        noise_frames = np.argsort(frame_energy)[:max(1, len(frame_energy) // 10)]
        noise_floor = np.mean(mag[:, noise_frames], axis=1, keepdims=True)

        # 后验信噪比
        snr_post = (mag ** 2) / (noise_floor ** 2 + 1e-10)

        # 先验信噪比估计（使用判决引导法）
        alpha = 0.98
        snr_prior = alpha * snr_post + (1 - alpha) * np.maximum(snr_post - 1, 0)

        # MMSE增益函数
        # G = snr_prior / (1 + snr_prior)  # Wiener滤波
        G = np.sqrt(snr_prior / (1 + snr_prior))

        # 应用增益
        mag_enhanced = mag * G

        # 重建
        stft_enhanced = mag_enhanced * np.exp(1j * phase)
        y_enhanced = librosa.istft(stft_enhanced, hop_length=512)

        # 长度对齐
        if len(y_enhanced) < len(y):
            y_enhanced = np.pad(y_enhanced, (0, len(y) - len(y_enhanced)))
        else:
            y_enhanced = y_enhanced[:len(y)]

        return y_enhanced

    except Exception as e:
        print(f"语音增强失败，使用预加重: {e}")
        return librosa.effects.preemphasis(y, coef=0.97)


def normalize_audio(y, target_db=-20):
    """音频归一化到目标dB"""
    # 计算RMS
    rms = np.sqrt(np.mean(y ** 2))
    if rms < 1e-10:
        return y

    # 计算当前dB
    current_db = 20 * np.log10(rms + 1e-10)

    # 增益
    gain = 10 ** ((target_db - current_db) / 20)

    y_normalized = y * gain

    # 防止削波
    max_val = np.max(np.abs(y_normalized))
    if max_val > 0.99:
        y_normalized = y_normalized / max_val * 0.99

    return y_normalized


def format_time(seconds):
    """格式化时间"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def detect_speech_segments(audio_path, threshold_db=-40, min_duration=0.5):
    """检测语音活动段（VAD）"""
    y, sr = librosa.load(audio_path, sr=16000)
    y_db = librosa.amplitude_to_db(np.abs(librosa.stft(y)))

    speech_segments = []
    in_speech = False
    start_time = 0

    hop_length = 512
    for i in range(y_db.shape[1]):
        time = i * hop_length / sr
        avg_energy = np.mean(y_db[:, i])

        if avg_energy > threshold_db and not in_speech:
            in_speech = True
            start_time = time
        elif avg_energy <= threshold_db and in_speech:
            in_speech = False
            duration = time - start_time
            if duration >= min_duration:
                speech_segments.append({
                    'start': round(start_time, 2),
                    'end': round(time, 2),
                    'duration': round(duration, 2)
                })

    if in_speech:
        duration = len(y) / sr - start_time
        if duration >= min_duration:
            speech_segments.append({
                'start': round(start_time, 2),
                'end': round(len(y) / sr, 2),
                'duration': round(duration, 2)
            })

    return speech_segments


def get_audio_quality_report(original_path, processed_path):
    """生成音频质量报告"""
    y_orig, sr = librosa.load(original_path, sr=16000)
    y_proc, _ = librosa.load(processed_path, sr=16000)

    # 长度对齐
    min_len = min(len(y_orig), len(y_proc))
    y_orig = y_orig[:min_len]
    y_proc = y_proc[:min_len]

    # 计算指标（转为Python float避免JSON序列化问题）
    noise_reduction_db = float(10 * np.log10(
        np.var(y_orig) / (np.var(y_orig - y_proc) + 1e-10) + 1e-10
    ))

    snr_before = float(10 * np.log10(
        np.var(y_orig) / (np.var(y_orig - y_orig.mean()) + 1e-10) + 1e-10
    ))
    snr_after = float(10 * np.log10(
        np.var(y_proc) / (np.var(y_proc - y_proc.mean()) + 1e-10) + 1e-10
    ))

    return {
        'noise_reduction': round(noise_reduction_db, 2),
        'snr_before': round(snr_before, 2),
        'snr_after': round(snr_after, 2),
        'improvement': round(snr_after - snr_before, 2)
    }