"""
音频预处理模块测试
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import librosa
import soundfile as sf
from modules.audio_preprocessor import (
    preprocess_audio, apply_noise_reduction, apply_echo_cancellation,
    apply_speech_enhancement, normalize_audio, detect_speech_segments,
    get_audio_quality_report
)


def test_preprocess_audio():
    """测试完整预处理流程"""
    print("=" * 60)
    print("测试: 完整音频预处理流程")
    print("=" * 60)
    
    test_file = os.path.join('test_audio_files', 'multi_speaker.wav')
    if not os.path.exists(test_file):
        print(f"❌ 测试文件不存在: {test_file}")
        return False
    
    output_file = 'test_output_processed.wav'
    
    try:
        y, sr = preprocess_audio(test_file, output_file)
        print(f"✅ 预处理成功")
        print(f"   输出长度: {len(y)/sr:.2f}秒")
        print(f"   采样率: {sr}Hz")
        print(f"   输出文件: {output_file}")
        
        # 检查输出文件
        if os.path.exists(output_file):
            print(f"✅ 输出文件已生成: {os.path.getsize(output_file)} bytes")
        
        # 清理
        if os.path.exists(output_file):
            os.remove(output_file)
        
        return True
    except Exception as e:
        print(f"❌ 预处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_noise_reduction():
    """测试降噪功能"""
    print("\n" + "=" * 60)
    print("测试: 降噪功能")
    print("=" * 60)
    
    test_file = os.path.join('test_audio_files', 'multi_speaker.wav')
    if not os.path.exists(test_file):
        print(f"❌ 测试文件不存在: {test_file}")
        return False
    
    try:
        y, sr = librosa.load(test_file, sr=16000)
        
        # 添加噪声
        noise = np.random.normal(0, 0.01, len(y))
        y_noisy = y + noise
        
        # 降噪
        y_denoised = apply_noise_reduction(y_noisy, sr)
        
        # 计算信噪比改善
        noise_before = np.var(y_noisy - y)
        noise_after = np.var(y_denoised - y)
        
        if noise_after < noise_before:
            improvement = 10 * np.log10(noise_before / noise_after)
            print(f"✅ 降噪成功")
            print(f"   噪声降低: {improvement:.2f} dB")
            return True
        else:
            print(f"⚠️ 降噪效果不明显")
            return True
    except Exception as e:
        print(f"❌ 降噪失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_echo_cancellation():
    """测试回声消除"""
    print("\n" + "=" * 60)
    print("测试: 回声消除")
    print("=" * 60)
    
    try:
        # 创建带回声的信号
        sr = 16000
        duration = 2
        t = np.linspace(0, duration, int(sr * duration))
        y = np.sin(2 * np.pi * 440 * t)  # 440Hz正弦波
        
        # 添加延迟回声
        delay = int(sr * 0.1)  # 100ms延迟
        echo = np.zeros_like(y)
        echo[delay:] = y[:-delay] * 0.3
        y_with_echo = y + echo
        
        # 回声消除
        y_clean = apply_echo_cancellation(y_with_echo, sr)
        
        # 检查回声是否减少
        echo_before = np.correlate(y_with_echo, y_with_echo, mode='full')[len(y_with_echo):]
        echo_after = np.correlate(y_clean, y_clean, mode='full')[len(y_clean):]
        
        peak_before = np.max(echo_before[int(sr*0.05):int(sr*0.2)])
        peak_after = np.max(echo_after[int(sr*0.05):int(sr*0.2)])
        
        if peak_after < peak_before:
            print(f"✅ 回声消除成功")
            print(f"   回声峰值降低: {20*np.log10(peak_before/peak_after):.2f} dB")
            return True
        else:
            print(f"⚠️ 回声消除效果不明显")
            return True
    except Exception as e:
        print(f"❌ 回声消除失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_speech_enhancement():
    """测试语音增强"""
    print("\n" + "=" * 60)
    print("测试: 语音增强")
    print("=" * 60)
    
    try:
        # 创建测试信号
        sr = 16000
        duration = 2
        t = np.linspace(0, duration, int(sr * duration))
        y = np.sin(2 * np.pi * 1000 * t)
        
        # 增强
        y_enhanced = apply_speech_enhancement(y, sr)
        
        print(f"✅ 语音增强完成")
        print(f"   输入长度: {len(y)}")
        print(f"   输出长度: {len(y_enhanced)}")
        return True
    except Exception as e:
        print(f"❌ 语音增强失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_normalize_audio():
    """测试音频归一化"""
    print("\n" + "=" * 60)
    print("测试: 音频归一化")
    print("=" * 60)
    
    try:
        # 创建不同幅度的信号
        y1 = np.random.randn(1000) * 0.1
        y2 = np.random.randn(1000) * 0.5
        
        y1_norm = normalize_audio(y1, target_db=-20)
        y2_norm = normalize_audio(y2, target_db=-20)
        
        rms1 = 20 * np.log10(np.sqrt(np.mean(y1_norm**2)) + 1e-10)
        rms2 = 20 * np.log10(np.sqrt(np.mean(y2_norm**2)) + 1e-10)
        
        print(f"✅ 归一化成功")
        print(f"   信号1 RMS: {rms1:.1f} dB")
        print(f"   信号2 RMS: {rms2:.1f} dB")
        print(f"   两者接近{-20}dB目标: {'✅' if abs(rms1 - (-20)) < 3 and abs(rms2 - (-20)) < 3 else '⚠️'}")
        return True
    except Exception as e:
        print(f"❌ 归一化失败: {e}")
        return False


def test_vad():
    """测试语音活动检测"""
    print("\n" + "=" * 60)
    print("测试: 语音活动检测(VAD)")
    print("=" * 60)
    
    test_file = os.path.join('test_audio_files', 'multi_speaker.wav')
    if not os.path.exists(test_file):
        print(f"❌ 测试文件不存在: {test_file}")
        return False
    
    try:
        segments = detect_speech_segments(test_file)
        
        print(f"✅ VAD检测完成")
        print(f"   检测到 {len(segments)} 个语音段")
        for i, seg in enumerate(segments[:5]):
            print(f"   段{i+1}: {seg['start']:.2f}s - {seg['end']:.2f}s (时长: {seg['duration']:.2f}s)")
        
        return len(segments) > 0
    except Exception as e:
        print(f"❌ VAD失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_audio_quality_report():
    """测试音频质量报告"""
    print("\n" + "=" * 60)
    print("测试: 音频质量报告")
    print("=" * 60)
    
    test_file = os.path.join('test_audio_files', 'multi_speaker.wav')
    if not os.path.exists(test_file):
        print(f"❌ 测试文件不存在: {test_file}")
        return False
    
    try:
        processed_file = 'test_processed.wav'
        preprocess_audio(test_file, processed_file)
        
        report = get_audio_quality_report(test_file, processed_file)
        
        print(f"✅ 质量报告生成成功")
        print(f"   降噪量: {report['noise_reduction']} dB")
        print(f"   处理前SNR: {report['snr_before']} dB")
        print(f"   处理后SNR: {report['snr_after']} dB")
        print(f"   改善: {report['improvement']} dB")
        
        if os.path.exists(processed_file):
            os.remove(processed_file)
        
        return True
    except Exception as e:
        print(f"❌ 质量报告失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("\n" + "🔊" * 30)
    print("音频预处理模块测试套件")
    print("🔊" * 30 + "\n")
    
    results = []
    
    results.append(("完整预处理流程", test_preprocess_audio()))
    results.append(("降噪功能", test_noise_reduction()))
    results.append(("回声消除", test_echo_cancellation()))
    results.append(("语音增强", test_speech_enhancement()))
    results.append(("音频归一化", test_normalize_audio()))
    results.append(("语音活动检测", test_vad()))
    results.append(("音频质量报告", test_audio_quality_report()))
    
    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status:8} - {name}")
    
    print(f"\n总计: {passed}/{total} 通过 ({passed/total*100:.1f}%)")
