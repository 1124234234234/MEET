"""
运行全部测试。

包含：
  1. 语音链路量化评测（tests/test_voice_pipeline.py）
     - 转写字错误率、说话人混淆率、要点覆盖、风险去重、接口冒烟
     - 上传分析路径端到端（后台线程 + 应用上下文）
  2. 实时转写 HTTP 接口端到端测试（tests/test_realtime_http.py）
     - 内置页面使用的 /api/realtime/* 全流程、48kHz 重采样、空会话、合规摘要
  3. 实时转写 Socket.IO 兼容测试（tests/test_realtime.py）
     - 事件接线、时间戳推进、时长计算、分析与入库
  4. 历史模块单元测试（音频预处理 / 文本分析 / 合规检查）

内存说明：
    每个测试进程都要自己加载一套模型（FunASR 声纹识别 + 标点 + 句向量，
    启用大模型摘要时再加 Qwen2.5-1.5B），峰值约 5GB。
    因此默认把 ENABLE_TEXT_MODELS 设为 false（摘要退回 TextRank 抽取式），
    省下约 3GB；需要连大模型摘要一起测时加 --with-text-models。
    另外测试进程与正在运行的服务会各占一套模型，跑之前请先停掉服务。

用法：
    python tests/run_all_tests.py                    # 默认（不含大模型摘要）
    python tests/run_all_tests.py --with-text-models # 连大模型摘要一起测
    python tests/run_all_tests.py --quick            # 只跑不需要模型的用例
"""
import os
import subprocess
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))

# 单个测试进程大致需要的内存（MB）；低于这个值就明确提示而不是让它原生崩溃
REQUIRED_FREE_MB = 5000


def available_memory_mb():
    """读取系统可用物理内存（MB），失败返回 None。"""
    if os.name != 'nt':
        try:
            with open('/proc/meminfo') as handle:
                for line in handle:
                    if line.startswith('MemAvailable:'):
                        return int(line.split()[1]) // 1024
        except OSError:
            return None
        return None
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ('dwLength', ctypes.c_ulong),
                ('dwMemoryLoad', ctypes.c_ulong),
                ('ullTotalPhys', ctypes.c_ulonglong),
                ('ullAvailPhys', ctypes.c_ulonglong),
                ('ullTotalPageFile', ctypes.c_ulonglong),
                ('ullAvailPageFile', ctypes.c_ulonglong),
                ('ullTotalVirtual', ctypes.c_ulonglong),
                ('ullAvailVirtual', ctypes.c_ulonglong),
                ('ullAvailExtendedVirtual', ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return int(status.ullAvailPhys // (1024 * 1024))
    except Exception:
        return None


def service_is_running():
    """检查服务是否正在监听（占用另一套模型内存）。"""
    import socket

    try:
        from config import Config

        ports = [int(Config.PORT)]
    except Exception:
        ports = [5001]
    ports += [5000, 5002]

    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.3)
            if sock.connect_ex(('127.0.0.1', port)) == 0:
                return port
    return None


def preflight(with_text_models):
    """跑之前的资源检查：内存不足或服务在跑时给出可操作的提示。"""
    problems = []

    free = available_memory_mb()
    if free is not None:
        needed = REQUIRED_FREE_MB + (3000 if with_text_models else 0)
        print(f'[检查] 可用内存 {free} MB，本次预计需要约 {needed} MB')
        if free < needed:
            problems.append(
                f'可用内存不足（{free} MB < 约 {needed} MB）。'
                '请关闭一些程序后重试；或先不加 --with-text-models 跑（摘要退回抽取式）。'
            )

    port = service_is_running()
    if port:
        problems.append(
            f'检测到服务正在运行（端口 {port}）。服务与测试各占一套模型内存，'
            '请先停止服务（在启动窗口按 Ctrl+C）再跑测试。'
        )

    if problems:
        print()
        for text in problems:
            print(f'[提示] {text}')
        print()
    return not problems or free is None and not port


def run(test_file, timeout=1800, env=None):
    """运行单个测试脚本，返回 (是否通过, 说明)。"""
    path = os.path.join(TESTS_DIR, test_file)
    if not os.path.exists(path):
        return None, '文件不存在'

    print('\n' + '=' * 78)
    print(f'运行: {test_file}')
    print('=' * 78)
    try:
        result = subprocess.run(
            [sys.executable, '-X', 'utf8', path],
            cwd=BASE_DIR,
            timeout=timeout,
            env=env,
        )
        code = result.returncode
        if code == 0:
            return True, '退出码 0'
        # Windows 原生崩溃码：多为内存不足
        if code in (3221225477, -1073741819):
            return False, '原生内存崩溃（0xC0000005），通常是内存不足'
        return False, f'退出码 {code}'
    except subprocess.TimeoutExpired:
        print(f'❌ 测试超时: {test_file}')
        return False, '超时'
    except Exception as exc:
        print(f'❌ 运行失败: {exc}')
        return False, str(exc)


def ensure_test_audio():
    """确保带标注的测试音频存在（不存在则生成）。"""
    audio_dir = os.path.join(TESTS_DIR, 'test_audio_files')
    required = ['gt_meeting.wav', 'gt_meeting.json', 'gt_2spk.wav', 'gt_2spk.json']
    if all(os.path.exists(os.path.join(audio_dir, name)) for name in required):
        return True

    print('\n' + '-' * 78)
    print('生成带标注的测试音频（tests/make_ground_truth_audio.py）')
    print('-' * 78)
    script = os.path.join(TESTS_DIR, 'make_ground_truth_audio.py')
    try:
        result = subprocess.run([sys.executable, '-X', 'utf8', script],
                                cwd=BASE_DIR, timeout=600)
        return result.returncode == 0
    except Exception as exc:
        print(f'⚠️ 生成测试音频失败（依赖 Windows 语音合成）: {exc}')
        return False


def main():
    args = sys.argv[1:]
    with_text_models = '--with-text-models' in args
    quick = '--quick' in args

    print('\n' + '🧪' * 30)
    print('会议室智能合规分析系统 - 完整测试套件')
    print('🧪' * 30)

    healthy = preflight(with_text_models)
    if not healthy:
        print('[提示] 仍会继续运行；若出现崩溃请按上面的提示处理后重试。\n')

    has_audio = ensure_test_audio()

    # 默认关闭大模型摘要以省内存（约 3GB）；摘要退回 TextRank 抽取式
    env = os.environ.copy()
    if not with_text_models:
        env['ENABLE_TEXT_MODELS'] = 'false'
        print('[信息] ENABLE_TEXT_MODELS=false（摘要走抽取式，省内存；'
              '要测大模型摘要请加 --with-text-models）')
    else:
        env['ENABLE_TEXT_MODELS'] = 'true'

    plans = [('test_voice_pipeline.py', 2400), ('test_realtime_http.py', 1800),
             ('test_realtime.py', 1800)]
    if not quick:
        plans += [('test_audio_preprocessor.py', 900), ('test_text_analyzer.py', 900),
                  ('test_compliance.py', 900)]

    results = []
    for name, timeout in plans:
        passed, detail = run(name, timeout=timeout, env=env)
        results.append((name, passed, detail))
        time.sleep(2)  # 给上一个进程一点时间释放内存

    print('\n' + '=' * 78)
    print('📊 测试汇总')
    print('=' * 78)
    for name, passed, detail in results:
        if passed is None:
            print(f'  ⏭️  {name:32} 跳过（{detail}）')
        elif passed:
            print(f'  ✅ {name:32} 通过')
        else:
            print(f'  ❌ {name:32} 失败（{detail}）')

    counted = [(n, p) for n, p, _ in results if p is not None]
    passed_count = sum(1 for _, p in counted if p)
    total = len(counted)
    if total:
        print(f'\n总计: {passed_count}/{total} 通过 ({passed_count / total * 100:.0f}%)')

    if not has_audio:
        print('\n⚠️ 未生成测试音频，语音链路指标测试会被跳过或失败。')
        print('   （需要 Windows 语音合成支持）')

    if total and passed_count == total:
        print('\n🎉 所有测试通过！')
        return 0
    print('\n⚠️ 部分测试未通过，请查看上方日志。')
    return 1


if __name__ == '__main__':
    sys.exit(main())

