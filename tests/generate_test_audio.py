"""
生成测试音频文件
用于测试多说话人分离、噪音场景等
"""
import numpy as np
import soundfile as sf


def generate_sine_wave(freq, duration, sr=16000, amplitude=0.3):
    """生成正弦波音频"""
    t = np.linspace(0, duration, int(sr * duration))
    return amplitude * np.sin(2 * np.pi * freq * t)


def add_white_noise(audio, noise_level=0.01):
    """添加白噪声"""
    noise = np.random.normal(0, noise_level, len(audio))
    return audio + noise


def generate_multi_speaker_audio(output_path, sr=16000):
    """
    生成模拟多人会议的音频
    不同频率代表不同说话人
    """
    duration = 10  # 10秒
    
    # 说话人A - 较低频率 (300Hz)
    speaker_a = np.zeros(int(sr * duration))
    a_segments = [(0.5, 2.0), (4.0, 5.5), (7.0, 8.5)]
    for start, end in a_segments:
        seg = generate_sine_wave(300, end - start, sr, 0.4)
        speaker_a[int(start*sr):int(start*sr)+len(seg)] = seg
    
    # 说话人B - 中等频率 (600Hz)
    speaker_b = np.zeros(int(sr * duration))
    b_segments = [(2.5, 3.8), (5.8, 7.0), (8.8, 9.5)]
    for start, end in b_segments:
        seg = generate_sine_wave(600, end - start, sr, 0.35)
        speaker_b[int(start*sr):int(start*sr)+len(seg)] = seg
    
    # 说话人C - 较高频率 (900Hz)
    speaker_c = np.zeros(int(sr * duration))
    c_segments = [(1.0, 2.2), (6.0, 7.2)]
    for start, end in c_segments:
        seg = generate_sine_wave(900, end - start, sr, 0.3)
        speaker_c[int(start*sr):int(start*sr)+len(seg)] = seg
    
    # 混合
    mixed = speaker_a + speaker_b + speaker_c
    
    # 添加环境噪声
    mixed = add_white_noise(mixed, noise_level=0.005)
    
    # 归一化
    mixed = mixed / np.max(np.abs(mixed)) * 0.95
    
    sf.write(output_path, mixed, sr)
    
    print(f"✅ 多说话人测试音频已生成: {output_path}")
    print(f"   时长: {duration}秒")
    print(f"   采样率: {sr}Hz")
    print(f"   说话人A时间段: {a_segments}")
    print(f"   说话人B时间段: {b_segments}")
    print(f"   说话人C时间段: {c_segments}")
    
    return output_path


def generate_noisy_audio(output_path, sr=16000):
    """生成带噪音的测试音频"""
    duration = 5
    
    # 干净的语音信号
    clean_signal = generate_sine_wave(440, duration, sr, 0.3)
    
    # 添加强噪声
    noise = np.random.normal(0, 0.05, len(clean_signal))
    noisy_signal = clean_signal + noise
    
    # 归一化
    noisy_signal = noisy_signal / np.max(np.abs(noisy_signal)) * 0.95
    
    sf.write(output_path, noisy_signal, sr)
    
    print(f"✅ 噪音测试音频已生成: {output_path}")
    print(f"   时长: {duration}秒")
    print(f"   噪声水平: 高")
    
    return output_path


def generate_echo_audio(output_path, sr=16000):
    """生成带回声的测试音频"""
    duration = 5
    
    # 原始信号
    original = generate_sine_wave(500, duration, sr, 0.4)
    
    # 添加回声 (100ms延迟)
    delay_samples = int(sr * 0.1)
    echo = np.zeros_like(original)
    echo[delay_samples:] = original[:-delay_samples] * 0.4
    
    # 混合
    with_echo = original + echo
    
    # 归一化
    with_echo = with_echo / np.max(np.abs(with_echo)) * 0.95
    
    sf.write(output_path, with_echo, sr)
    
    print(f"✅ 回声测试音频已生成: {output_path}")
    print(f"   时长: {duration}秒")
    print(f"   回声延迟: 100ms")
    
    return output_path


def generate_silent_audio(output_path, sr=16000):
    """生成静音/低音量音频"""
    duration = 3
    
    # 几乎静音
    silent = np.random.normal(0, 0.001, int(sr * duration))
    
    sf.write(output_path, silent, sr)
    
    print(f"✅ 静音测试音频已生成: {output_path}")
    print(f"   时长: {duration}秒")
    print(f"   音量: 极低 (模拟会议间隙)")
    
    return output_path


if __name__ == '__main__':
    print("=" * 60)
    print("生成测试音频文件")
    print("=" * 60 + "\n")
    
    import os
    os.makedirs('test_audio_files', exist_ok=True)
    
    generate_multi_speaker_audio('test_audio_files/multi_speaker.wav')
    generate_noisy_audio('test_audio_files/noisy.wav')
    generate_echo_audio('test_audio_files/with_echo.wav')
    generate_silent_audio('test_audio_files/silent.wav')
    
    print("\n" + "=" * 60)
    print("所有测试音频已生成完毕")
    print("=" * 60)
    print("\n文件列表:")
    for f in os.listdir('test_audio_files'):
        filepath = os.path.join('test_audio_files', f)
        size = os.path.getsize(filepath)
        print(f"   {f}: {size/1024:.1f} KB")
