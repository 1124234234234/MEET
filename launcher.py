"""
会议室智能合规分析系统 · 本地一键启动器

做的事情：
  1. 检查 Python 版本与依赖是否齐全（缺什么直接告诉你装什么）
  2. 检查本地模型是否就位（FunASR 识别/分段/标点、句向量、摘要模型）
  3. 挑一个可用端口（默认取 config.py 的 PORT，被占用时自动顺延）
  4. 启动服务，轮询 /api/health 等到就绪，然后自动打开浏览器
  5. Ctrl+C 停止时会一并结束子进程

用法：
    python launcher.py                # 正常启动
    python launcher.py --check        # 只做环境自检，不启动服务
    python launcher.py --port 8080    # 指定端口
    python launcher.py --no-browser   # 不自动打开浏览器
"""
import argparse
import importlib
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# 运行必需的第三方包（导入名 -> pip 包名）
REQUIRED_PACKAGES = [
    ('flask', 'flask'),
    ('flask_cors', 'flask-cors'),
    ('flask_socketio', 'flask-socketio'),
    ('flask_sqlalchemy', 'flask-sqlalchemy'),
    ('numpy', 'numpy'),
    ('scipy', 'scipy'),
    ('soundfile', 'soundfile'),
    ('librosa', 'librosa'),
    ('sklearn', 'scikit-learn'),
    ('jieba', 'jieba'),
    ('funasr', 'funasr'),
    ('torch', 'torch'),
    ('transformers', 'transformers'),
    ('sentence_transformers', 'sentence-transformers'),
    ('opencc', 'opencc-python-reimplemented'),
    ('snownlp', 'snownlp'),
    ('noisereduce', 'noisereduce'),
    ('sumy', 'sumy'),
    ('nltk', 'nltk'),
]

GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
CYAN = '\033[96m'
RESET = '\033[0m'


def enable_ansi():
    """让 Windows 控制台支持 ANSI 颜色。"""
    if os.name == 'nt':
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            pass


def say(message, color=''):
    print(f'{color}{message}{RESET}' if color else message)


def check_python():
    version = sys.version_info
    ok = version >= (3, 10)
    detail = f'{version.major}.{version.minor}.{version.micro}'
    if ok:
        say(f'  [OK]   Python {detail}', GREEN)
    else:
        say(f'  [FAIL] Python {detail}（需要 3.10 及以上）', RED)
    return ok


def check_packages():
    missing = []
    for module_name, pip_name in REQUIRED_PACKAGES:
        try:
            importlib.import_module(module_name)
        except Exception:
            missing.append(pip_name)

    if missing:
        say(f'  [FAIL] 缺少 {len(missing)} 个依赖包', RED)
        for name in missing:
            say(f'         - {name}', YELLOW)
        say('         安装命令：', CYAN)
        say(f'         python -m pip install {" ".join(missing)}', CYAN)
        say('         或一次性安装全部依赖：', CYAN)
        say('         python -m pip install -r requirements.txt', CYAN)
        return False

    say(f'  [OK]   依赖包齐全（{len(REQUIRED_PACKAGES)} 项）', GREEN)
    return True


def check_models():
    try:
        from modules import model_registry

        status = model_registry.status()
    except Exception as exc:
        say(f'  [FAIL] 无法读取模型状态: {exc}', RED)
        return False

    labels = {
        'funasr_asr': '语音识别模型 paraformer-large',
        'funasr_vad': '语音分段模型 fsmn-vad',
        'funasr_punc': '标点恢复模型 ct-punc',
        'embedding_model': '句向量模型 bge-small-zh-v1.5',
        'summary_model': '摘要模型 Qwen2.5-1.5B-Instruct',
    }
    all_ok = True
    for key, label in labels.items():
        ready = bool(status.get(key))
        if ready:
            say(f'  [OK]   {label}', GREEN)
        else:
            # 识别模型缺失会导致完全不可用；其它模型缺失只影响对应功能
            critical = key in ('funasr_asr',)
            all_ok = all_ok and not critical
            say(f'  [{"FAIL" if critical else "WARN"}] {label} 缺失（查 models/ 目录）',
                RED if critical else YELLOW)
    return all_ok


def pick_port(preferred):
    """从 preferred 开始找一个空闲端口。"""
    for candidate in range(preferred, preferred + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(('127.0.0.1', candidate))
                return candidate
            except OSError:
                continue
    return preferred


def port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(('127.0.0.1', port)) == 0


def wait_for_health(port, timeout=180):
    """轮询健康检查接口，返回 (是否就绪, 最近一次错误)。"""
    deadline = time.time() + timeout
    url = f'http://127.0.0.1:{port}/api/health'
    last_error = ''
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if response.status == 200:
                    return True, ''
        except urllib.error.URLError as exc:
            last_error = str(exc)
        except Exception as exc:
            last_error = str(exc)
        time.sleep(0.5)
    return False, last_error


def health_models(port):
    """读取健康检查里的模型就绪状态。"""
    import json

    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/health', timeout=5) as response:
            payload = json.loads(response.read().decode('utf-8'))
            return payload.get('models', {})
    except Exception:
        return {}


def run_checks():
    say('\n[1/3] 运行环境检查', CYAN)
    python_ok = check_python()
    packages_ok = check_packages()
    say('\n[2/3] 本地模型检查', CYAN)
    models_ok = check_models()
    return python_ok and packages_ok and models_ok


def main():
    enable_ansi()
    parser = argparse.ArgumentParser(description='会议室智能合规分析系统 启动器')
    parser.add_argument('--port', type=int, default=None, help='服务端口（默认取 config.py）')
    parser.add_argument('--host', default=None, help='监听地址（默认取 config.py）')
    parser.add_argument('--no-browser', action='store_true', help='不自动打开浏览器')
    parser.add_argument('--check', action='store_true', help='只做环境自检，不启动服务')
    parser.add_argument('--no-preload', action='store_true', help='跳过模型预加载')
    args = parser.parse_args()

    say('=' * 62)
    say('  会议室智能合规分析系统 · 语音转写与合规比对')
    say('=' * 62)

    healthy = run_checks()

    if args.check:
        say('\n' + '=' * 62)
        if healthy:
            say('自检通过，可以启动服务。', GREEN)
            return 0
        say('自检发现问题，请按上面提示处理后重试。', YELLOW)
        return 1

    if not healthy:
        say('\n检测到环境问题，仍会尝试启动（有问题可先按提示修复）。', YELLOW)

    say('\n[3/3] 启动服务', CYAN)

    from config import Config

    preferred = args.port or int(Config.PORT)
    port = pick_port(preferred)
    if port != preferred:
        say(f'  端口 {preferred} 被占用，改用 {port}', YELLOW)
    else:
        say(f'  端口 {port}', GREEN)

    host = args.host or Config.HOST

    env = os.environ.copy()
    env['PORT'] = str(port)
    env['HOST'] = host
    if args.no_preload:
        env['PRELOAD_MODELS'] = 'false'
    # 让子进程用 UTF-8 输出，避免中文日志在 Windows 控制台乱码
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUTF8'] = '1'

    say('  正在启动服务进程...', CYAN)
    process = subprocess.Popen(
        [sys.executable, '-X', 'utf8', os.path.join(BASE_DIR, 'app.py')],
        cwd=BASE_DIR,
        env=env,
    )

    url = f'http://127.0.0.1:{port}'
    say('  等待服务就绪（首次启动需加载模型，约 30~60 秒）...', CYAN)

    try:
        ready, error = wait_for_health(port, timeout=240)
    except KeyboardInterrupt:
        process.terminate()
        return 0

    if not ready:
        say(f'  [FAIL] 服务未能就绪：{error}', RED)
        say('         请查看上面的服务日志排查问题。', YELLOW)
        process.terminate()
        return 1

    models = health_models(port)
    say('  [OK]   服务已就绪', GREEN)

    if not models.get('ready', True):
        say('  [提示] 模型仍在后台加载中，首次上传/实时转写会自动等待加载完成。', YELLOW)

    say('')
    say('=' * 62)
    say(f'  访问地址: {url}', GREEN)
    say('  实时转写: 打开页面后点击「实时转写」开始录音')
    say('  第三方接口: POST ' + url + '/api/v1/transcribe')
    say('  按 Ctrl+C 停止服务')
    say('=' * 62)
    say('')

    if not args.no_browser:
        webbrowser.open(url)

    try:
        process.wait()
    except KeyboardInterrupt:
        say('\n正在停止服务...', CYAN)
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        say('服务已停止。', GREEN)
    return 0


if __name__ == '__main__':
    sys.exit(main())
