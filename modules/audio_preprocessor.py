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
    回声消除：使用自适应滤波器（LMS算法）
    原理：通过自适应滤波器估计回声路径，从信号中减去回声
    """
    try:
        # 使用自适应LMS滤波器消除回声
        # 检测回声：如果信号中有周期性的延迟重复，认为是回声

        # 自相关检测回声延迟
        corr = np.correlate(y, y, mode='full')
        corr = corr[len(corr)//2:]  # 取正半部分

        # 找到自相关峰值（排除零延迟）
        if len(corr) > int(sr * 0.1):
            search_start = int(sr * 0.01)  # 从10ms开始搜索
            search_end = int(sr * 0.3)     # 搜索到300ms
            search_range = corr[search_start:min(search_end, len(corr))]

            if len(search_range) > 0:
                peak_idx = np.argmax(search_range) + search_start
                peak_ratio = corr[peak_idx] / (corr[0] + 1e-10)

                # 如果存在明显回声（自相关峰值大于0.3）
                if peak_ratio > 0.3:
                    delay_samples = peak_idx
                    y = lms_echo_cancel(y, delay_samples)

        # 使用带通滤波进一步清理
        # 保留人声频率范围 80Hz - 7000Hz
        b, a = signal.butter(4, [80, 7000], btype='band', fs=sr)
        y_filtered = signal.filtfilt(b, a, y)

        return y_filtered

    except Exception as e:
        print(f"回声消除失败，使用带通滤波: {e}")
        b, a = signal.butter(4, [80, 7000], btype='band', fs=sr)
        return signal.filtfilt(b, a, y)


def lms_echo_cancel(y, delay_samples, filter_length=256, step_size=0.01):
    """
    LMS自适应滤波器回声消除
    """
    n = len(y)
    y_out = np.zeros(n)

    # 远端参考信号（延迟信号）
    x = np.zeros(n)
    if delay_samples < n:
        x[delay_samples:] = y[:n - delay_samples]

    # 自适应滤波器权重
    w = np.zeros(filter_length)

    for i in range(filter_length, n):
        # 输入向量
        x_vec = x[i - filter_length:i][::-1]

        # 滤波器输出（估计的回声）
        echo_est = np.dot(w, x_vec)

        # 误差信号（消除回声后的信号）
        y_out[i] = y[i] - echo_est

        # LMS权重更新
        w = w + step_size * y_out[i] * x_vec

    return y_out


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