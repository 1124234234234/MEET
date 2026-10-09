"""
本地模型装载中心（离线优先）

设计目标：
1. 所有模型优先使用项目 models/ 目录下的本地副本，避免无网络环境下
   huggingface / modelscope 反复重试导致的长时间卡顿（旧实现会重试 5 次）。
2. 全局单例：同一个模型只加载一次，多个模块共享，避免重复占用内存。
   旧实现里 text_analyzer 与 compliance_checker 各自加载了一份 bge 向量模型。
3. 模型不可用时返回 None，由调用方降级，不让异常冒到业务流程外。
"""
import os
import threading
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
FUNASR_DIR = os.path.join(MODELS_DIR, 'funasr')

_lock = threading.RLock()

_funasr_pipeline = None
_funasr_vad_fine = None
_sentence_transformers = {}


def _snapshot_dir(root):
    """modelscope 缓存目录结构为 <root>/snapshots/master，兼容直接放模型文件的情况。"""
    candidate = os.path.join(root, 'snapshots', 'master')
    if os.path.isdir(candidate):
        return candidate
    return root


def find_local_model(*keywords, root=None):
    """
    在 models/（或指定 root）下按关键字模糊查找本地模型目录。

    例如 find_local_model('paraformer') 返回 paraformer-large ASR 模型目录。
    找不到返回 None。
    """
    root = root or MODELS_DIR
    if not os.path.isdir(root):
        return None

    lowered = [k.lower() for k in keywords]
    matches = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        if all(k in name.lower() for k in lowered):
            matches.append(path)

    # 关键字少的优先（例如 'vad' 不应该匹配到名字更长的非目标目录）
    for path in matches:
        snapshot = _snapshot_dir(path)
        # 必须真的包含模型权重才算可用
        has_weight = any(
            os.path.exists(os.path.join(snapshot, f))
            for f in ('model.pt', 'model.safetensors', 'pytorch_model.bin')
        )
        if has_weight:
            return snapshot
    return None


def funasr_model_paths():
    """返回本地 FunASR 模型路径字典，缺失的键为 None。"""
    return {
        'asr': find_local_model('paraformer', root=FUNASR_DIR),
        'vad': find_local_model('vad', root=FUNASR_DIR),
        'punc': find_local_model('punc', root=FUNASR_DIR),
    }


def has_local_funasr():
    paths = funasr_model_paths()
    return paths['asr'] is not None


def get_funasr_pipeline():
    """
    加载并缓存「语音识别 + 标点恢复」流水线（paraformer-large + ct-punc）。

    返回 AutoModel 实例；本地模型缺失时返回 None。
    """
    global _funasr_pipeline
    if _funasr_pipeline is not None:
        return _funasr_pipeline

    with _lock:
        if _funasr_pipeline is not None:
            return _funasr_pipeline

        paths = funasr_model_paths()
        if not paths['asr']:
            print('[模型] 未找到本地 FunASR 识别模型，跳过加载')
            return None

        try:
            from funasr import AutoModel

            kwargs = {
                'model': paths['asr'],
                'disable_update': True,
                'device': 'cpu',
                'disable_pbar': True,
            }
            if paths['punc']:
                kwargs['punc_model'] = paths['punc']
                print('[模型] 标点恢复模型已启用（ct-punc）')
            if paths['vad']:
                kwargs['vad_model'] = paths['vad']
                kwargs['vad_kwargs'] = {
                    'max_end_silence_time': 300,
                    'max_single_segment_time': 15000,
                }

            print('[模型] 正在加载 FunASR 语音识别模型...')
            _funasr_pipeline = AutoModel(**kwargs)
            print('[模型] FunASR 语音识别模型加载完成')
        except Exception as exc:
            print(f'[模型] FunASR 加载失败: {exc}')
            _funasr_pipeline = None

    return _funasr_pipeline


def get_vad_model():
    """
    加载并缓存单独的 VAD 模型，用于「细粒度语音分段」。

    分段粒度直接决定说话人分离的基本单元，所以这里用较短的静音容忍时间，
    让停顿也切分成独立片段（FunASR 默认 800ms 会把整段连续语音并成一片）。
    """
    global _funasr_vad_fine
    if _funasr_vad_fine is not None:
        return _funasr_vad_fine

    with _lock:
        if _funasr_vad_fine is not None:
            return _funasr_vad_fine

        vad_path = funasr_model_paths()['vad']
        if not vad_path:
            print('[模型] 未找到本地 VAD 模型')
            return None

        try:
            from funasr import AutoModel

            _funasr_vad_fine = AutoModel(
                model=vad_path,
                disable_update=True,
                device='cpu',
                disable_pbar=True,
                max_end_silence_time=300,
                max_single_segment_time=15000,
            )
            print('[模型] VAD 模型加载完成')
        except Exception as exc:
            print(f'[模型] VAD 加载失败: {exc}')
            _funasr_vad_fine = None

    return _funasr_vad_fine


def get_sentence_transformer(language='zh'):
    """
    加载并缓存句向量模型（中文使用本地 bge-small-zh-v1.5）。

    本地存在时强制 local_files_only，避免离线环境下向 HuggingFace 发请求后
    重试等待；本地不存在时才退回在线模型名。
    """
    if language == 'zh':
        local_dir = os.path.join(MODELS_DIR, 'bge-small-zh-v1.5')
        model_ref = local_dir if os.path.isdir(local_dir) else 'BAAI/bge-small-zh-v1.5'
        local = os.path.isdir(local_dir)
    else:
        local_dir = os.path.join(MODELS_DIR, 'all-MiniLM-L6-v2')
        model_ref = local_dir if os.path.isdir(local_dir) else 'sentence-transformers/all-MiniLM-L6-v2'
        local = os.path.isdir(local_dir)

    cache_key = str(model_ref)
    if cache_key in _sentence_transformers:
        return _sentence_transformers[cache_key]

    with _lock:
        if cache_key in _sentence_transformers:
            return _sentence_transformers[cache_key]

        try:
            from sentence_transformers import SentenceTransformer

            kwargs = {'local_files_only': True} if local else {}
            print(f'[模型] 正在加载句向量模型: {model_ref}')
            model = SentenceTransformer(model_ref, **kwargs)
            _sentence_transformers[cache_key] = model
            print('[模型] 句向量模型加载完成')
        except Exception as exc:
            print(f'[模型] 句向量模型加载失败: {exc}')
            return None

    return _sentence_transformers[cache_key]


def warmup(include_transcriber=True):
    """启动预热：把最常用的模型提前加载好，避免首次请求超时。"""
    if include_transcriber:
        get_funasr_pipeline()
    get_sentence_transformer('zh')


def status():
    """返回本地模型可用性概览，供启动脚本/健康检查展示。"""
    paths = funasr_model_paths()
    bge = os.path.isdir(os.path.join(MODELS_DIR, 'bge-small-zh-v1.5'))
    qwen = os.path.isdir(os.path.join(MODELS_DIR, 'Qwen2.5-1.5B-Instruct'))
    return {
        'funasr_asr': paths['asr'],
        'funasr_vad': paths['vad'],
        'funasr_punc': paths['punc'],
        'embedding_model': bge,
        'summary_model': qwen,
        'ready': paths['asr'] is not None and bge,
    }
