# -*- coding: utf-8 -*-
"""
把运行所需的模型下载到项目 models/ 目录。

需要联网。若部署机器不能上网，请在能上网的机器上执行本脚本，或在原机器上
直接拷贝整个 models/ 目录（约 5GB）到部署目录，效果相同。

用法：
    python tools/download_models.py              # 全部 5 个模型
    python tools/download_models.py --only asr   # 只下语音识别三件套（约 2GB）
    python tools/download_models.py --only text  # 只下句向量与摘要模型（约 3GB）
    python tools/download_models.py --check      # 只检查本地是否齐全

国内网络可先设置镜像（任一）：
    set MODELSCOPE_ENDPOINT=https://www.modelscope.cn
    set HF_ENDPOINT=https://hf-mirror.com

下载结果与系统读取路径一致：
    models/funasr/iic--speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/
    models/funasr/iic--speech_fsmn_vad_zh-cn-16k-common-pytorch/
    models/funasr/iic--punc_ct-transformer_cn-en-common-vocab471067-large/
    models/bge-small-zh-v1.5/
    models/Qwen2.5-1.5B-Instruct/
"""
import argparse
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
FUNASR_DIR = os.path.join(MODELS_DIR, 'funasr')

FUNASR_MODELS = (
    'iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
    'iic/speech_fsmn_vad_zh-cn-16k-common-pytorch',
    'iic/punc_ct-transformer_cn-en-common-vocab471067-large',
)
HF_MODELS = (
    ('BAAI/bge-small-zh-v1.5', 'bge-small-zh-v1.5'),
    ('Qwen/Qwen2.5-1.5B-Instruct', 'Qwen2.5-1.5B-Instruct'),
)


def target_of(model_id, root):
    """ModelScope 缓存目录习惯用 owner--name，和现有 models/ 结构保持一致。"""
    return os.path.join(root, model_id.replace('/', '--'))


def already_there(path):
    for name in ('model.pt', 'model.safetensors', 'pytorch_model.bin'):
        if os.path.exists(os.path.join(path, name)):
            return True
    snapshots = os.path.join(path, 'snapshots', 'master')
    if os.path.isdir(snapshots):
        return any(os.path.exists(os.path.join(snapshots, name))
                   for name in ('model.pt', 'model.safetensors', 'pytorch_model.bin'))
    return False


def download_modelscope(model_id, target, retries=3):
    from modelscope import snapshot_download

    os.makedirs(target, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            print(f'  [下载] {model_id} -> {target}（第 {attempt} 次）', flush=True)
            snapshot_download(model_id, local_dir=target)
            return True
        except Exception as exc:
            print(f'  [失败] {exc}', flush=True)
            if attempt < retries:
                time.sleep(10)
    return False


def download_hf(model_id, target, retries=3):
    from huggingface_hub import snapshot_download

    os.makedirs(target, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            print(f'  [下载] {model_id} -> {target}（第 {attempt} 次）', flush=True)
            snapshot_download(repo_id=model_id, local_dir=target)
            return True
        except Exception as exc:
            print(f'  [失败] {exc}', flush=True)
            if attempt < retries:
                time.sleep(10)
    return False


def report():
    """打印本地模型状态（与 /api/health 的 models 字段同一套判据）。"""
    found = {}
    for model_id in FUNASR_MODELS:
        path = target_of(model_id, FUNASR_DIR)
        found[model_id] = already_there(path)
    for model_id, folder in HF_MODELS:
        found[model_id] = os.path.isdir(os.path.join(MODELS_DIR, folder)) and \
            (already_there(os.path.join(MODELS_DIR, folder))
             or os.path.exists(os.path.join(MODELS_DIR, folder, 'config.json')))
    print('本地模型检查：')
    for model_id, ok in found.items():
        print(f'  [{"OK" if ok else "缺失"}] {model_id}')
    return all(found.values())


def main():
    parser = argparse.ArgumentParser(description='下载运行所需的本地模型')
    parser.add_argument('--only', choices=('asr', 'text', 'all'), default='all')
    parser.add_argument('--check', action='store_true', help='只检查，不下载')
    args = parser.parse_args()

    if args.check:
        return 0 if report() else 1

    os.makedirs(FUNASR_DIR, exist_ok=True)
    failures = []
    if args.only in ('asr', 'all'):
        print('== 语音识别：paraformer + VAD + 标点 ==')
        for model_id in FUNASR_MODELS:
            target = target_of(model_id, FUNASR_DIR)
            if already_there(target):
                print(f'  [跳过] {model_id} 已存在')
                continue
            if not download_modelscope(model_id, target):
                failures.append(model_id)
    if args.only in ('text', 'all'):
        print('== 文本模型：句向量 + 摘要 ==')
        for model_id, folder in HF_MODELS:
            target = os.path.join(MODELS_DIR, folder)
            if already_there(target) or os.path.exists(os.path.join(target, 'config.json')):
                print(f'  [跳过] {model_id} 已存在')
                continue
            if not download_hf(model_id, target):
                failures.append(model_id)

    ok = report()
    if failures:
        print('\n以下模型下载失败：')
        for model_id in failures:
            print('  -', model_id)
        print('可重试本脚本，或从原机器直接拷贝整个 models/ 目录。')
        return 1
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
