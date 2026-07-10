"""
运行所有测试
"""
import sys
import os
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def run_test(test_file):
    """运行单个测试文件"""
    print("\n" + "=" * 80)
    print(f"运行: {test_file}")
    print("=" * 80)
    
    test_path = os.path.join(os.path.dirname(__file__), test_file)
    
    try:
        result = subprocess.run(
            [sys.executable, test_path],
            capture_output=False,
            text=True,
            timeout=300
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"❌ 测试超时: {test_file}")
        return False
    except Exception as e:
        print(f"❌ 运行失败: {e}")
        return False


if __name__ == '__main__':
    print("\n" + "🧪" * 40)
    print("会议智能分析系统 - 完整测试套件")
    print("🧪" * 40 + "\n")
    
    # 先生成测试音频
    print("\n" + "-" * 80)
    print("步骤1: 生成测试音频文件")
    print("-" * 80)
    
    try:
        from tests.generate_test_audio import (
            generate_multi_speaker_audio,
            generate_noisy_audio,
            generate_echo_audio,
            generate_silent_audio
        )
        os.makedirs('test_audio_files', exist_ok=True)
        generate_multi_speaker_audio('test_audio_files/multi_speaker.wav')
        generate_noisy_audio('test_audio_files/noisy.wav')
        generate_echo_audio('test_audio_files/with_echo.wav')
        generate_silent_audio('test_audio_files/silent.wav')
    except Exception as e:
        print(f"⚠️ 生成测试音频失败 (不影响后续测试): {e}")
    
    # 运行各模块测试
    print("\n" + "-" * 80)
    print("步骤2: 运行模块测试")
    print("-" * 80)
    
    tests = [
        ("test_audio_preprocessor.py", "音频预处理模块"),
        ("test_text_analyzer.py", "文本分析模块"),
        ("test_compliance.py", "合规检查模块"),
    ]
    
    results = []
    for test_file, test_name in tests:
        passed = run_test(test_file)
        results.append((test_name, passed))
    
    # API测试 (可选，需要服务运行)
    print("\n" + "-" * 80)
    print("步骤3: API接口测试 (需要服务运行)")
    print("-" * 80)
    
    import requests
    try:
        response = requests.get("http://127.0.0.1:5000/api/health", timeout=2)
        if response.status_code == 200:
            passed = run_test("test_api.py")
            results.append(("API接口测试", passed))
        else:
            print("⚠️ 服务未运行，跳过API测试")
            results.append(("API接口测试", None))
    except:
        print("⚠️ 服务未运行，跳过API测试")
        results.append(("API接口测试", None))
    
    # 汇总
    print("\n" + "=" * 80)
    print("📊 测试汇总报告")
    print("=" * 80)
    
    for name, passed in results:
        if passed is None:
            status = "⏭️ 跳过"
        elif passed:
            status = "✅ 通过"
        else:
            status = "❌ 失败"
        print(f"{status:8} - {name}")
    
    passed_count = sum(1 for _, p in results if p is True)
    total_count = sum(1 for _, p in results if p is not None)
    
    print(f"\n总计: {passed_count}/{total_count} 通过 ({passed_count/total_count*100:.1f}%)")
    
    if passed_count == total_count:
        print("\n🎉 所有测试通过！")
    else:
        print("\n⚠️ 部分测试未通过，请检查日志")
