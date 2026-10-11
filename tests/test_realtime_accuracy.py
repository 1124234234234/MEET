"""
实时转写链路准确率评测（回放真实音频，按浏览器实际块大小逐块喂给实时会话）。

为什么要单独测这条链路：
    实时转写的音频是「边录边发、按静音断句」的，识别单元比文件转写短，
    是整条语音链路里最容易出问题的部分 —— 实测同一个失败样本，细粒度断句
    切出 24 句碎片、平均每句 4.9 字，文本几乎不可读。这里把带标注的测试音频
    按 500ms 的块喂给 FunASRRealtimeTranscriber（与内置页面一致），对最终
    文本算字错误率 CER，保证优化实时链路时不会退化（需求：WER < 10%）。

覆盖：
  1. 回放带标注音频（gt_meeting / gt_2spk），CER < 10%
  2. 纯静音不应产出任何文本（回归：旧版会把静音识别成「分下。」等幻觉文本）
  3. 分句时间戳连续、覆盖整段录音（时间戳不能漂移或倒退）

运行：
    python tests/test_realtime_accuracy.py
"""
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_DIR = os.path.join(BASE_DIR, 'tests', 'test_audio_files')

SCENARIOS = ['gt_meeting', 'gt_2spk']
# 与内置页面一致：每 500ms 发一块 16bit PCM
CHUNK_MS = 500
CER_TARGET = 0.10

RESULTS = []


def record(name, passed, detail=''):
    RESULTS.append((name, passed, detail))
    print(f'  [{"PASS" if passed else "FAIL"}] {name}' + (f' — {detail}' if detail else ''))
    return passed


def strip_punct(text):
    return re.sub(r'[\s，。！？、；：“”‘’（）《》【】,.!?;:"\'()\[\]<>—…·-]', '', text or '')


def edit_distance(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def character_error_rate(reference, hypothesis):
    ref = strip_punct(reference)
    hyp = strip_punct(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def replay(audio, sr, *, sample_rate=None):
    """
    按 500ms 分块回放音频，返回 (最终结果, 实时会话对象)。

    音频先转成 16bit PCM 字节流 —— 与浏览器 MediaRecorder 送给服务端的
    格式一致，顺带验证服务端的字节解码与采样率处理。
    """
    from modules.funasr_transcriber import FunASRRealtimeTranscriber

    transcriber = FunASRRealtimeTranscriber(
        language='zh',
        sample_rate=sample_rate or sr,
    )
    step = int(sr * CHUNK_MS / 1000)
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    for start in range(0, len(pcm), step):
        transcriber.add_audio_chunk(pcm[start:start + step].tobytes())
    return transcriber.get_final_result(), transcriber


def test_realtime_cer():
    """回放带标注音频，实时链路 CER 必须达标（需求 WER < 10%）。"""
    print('\n[1] 实时转写字错误率（回放真实音频，500ms 分块）')
    from modules import model_registry
    from modules.audio_io import load_audio

    # 先把模型加载出来再计时：首次调用要 20 多秒（加载 paraformer/vad/punc），
    # 混进 RTF 会把实时率算成 0.5 以上，与稳态（约 0.05）差一个数量级。
    started = time.time()
    model_registry.get_vad_model()
    model_registry.get_funasr_pipeline()
    print(f'         （模型预热 {time.time() - started:.1f}s，不计入 RTF）')

    for scenario in SCENARIOS:
        gt_path = os.path.join(AUDIO_DIR, f'{scenario}.json')
        wav = os.path.join(AUDIO_DIR, f'{scenario}.wav')
        if not os.path.exists(gt_path) or not os.path.exists(wav):
            record(f'{scenario} 实时 CER', False,
                   '缺少标注音频，请先运行 tests/make_ground_truth_audio.py')
            continue

        with open(gt_path, encoding='utf-8') as handle:
            gt = json.load(handle)

        audio = load_audio(wav, sr=16000)
        started = time.time()
        result, transcriber = replay(audio, 16000)
        elapsed = time.time() - started

        cer = character_error_rate(gt['full_text'], result['text'])
        segments = result['transcriptions']
        avg_len = (sum(len(s['text']) for s in segments) / len(segments)) if segments else 0
        duration = transcriber.duration_seconds()
        rtf = elapsed / duration if duration else 0

        record(
            f'{scenario} 实时 CER={cer:.2%} (<{CER_TARGET:.0%}，RTF={rtf:.3f}，'
            f'{len(segments)} 句，平均 {avg_len:.1f} 字/句)',
            cer < CER_TARGET,
        )
        if segments:
            print(f'         首句: {segments[0]["text"][:40]}')

        # 时间戳必须单调不倒退，且最后一句的结束时间贴着录音末尾
        # （偏小说明尾部音频被丢掉，偏大说明时间戳重复累加）
        starts_ok = all(
            segments[i]['start_time'] <= segments[i + 1]['start_time']
            for i in range(len(segments) - 1)
        ) if len(segments) > 1 else True
        last_end = segments[-1]['end_time'] if segments else 0
        covered = bool(segments) and duration - 2.5 <= last_end <= duration + 0.5
        record(f'{scenario} 实时分句时间戳单调且覆盖到录音末尾',
               starts_ok and covered,
               f'末尾 {last_end}s / 录音 {duration:.1f}s')


def test_silence_no_hallucination():
    """纯静音/极低噪声不应产出文本（旧版会识别出「分下。」这类幻觉内容）。"""
    print('\n[2] 纯静音不产生幻觉文本')
    rng = np.random.default_rng(20261011)
    silence = (rng.standard_normal(16000 * 6).astype(np.float32) * 0.0008)
    result, _ = replay(silence, 16000)
    text = (result.get('text') or '').strip()
    record('6 秒静音无输出文本', not text, f'输出: {text[:60]!r}' if text else '（无输出）')


def test_timestamp_progress():
    """跨多个识别单元的会话：时间戳按音频推进，不因静音块漂移。"""
    print('\n[3] 静音占比高时时间戳仍与音频对齐')
    from modules.audio_io import load_audio

    wav = os.path.join(AUDIO_DIR, 'gt_meeting.wav')
    if not os.path.exists(wav):
        record('时间戳对齐', False, '缺少标注音频')
        return

    audio = load_audio(wav, sr=16000)
    # 在录音中间插入 3 秒静音再回放，模拟会议中的停顿
    padded = np.concatenate([audio[:len(audio) // 2],
                             np.zeros(16000 * 3, dtype=np.float32),
                             audio[len(audio) // 2:]])
    result, transcriber = replay(padded, 16000)
    duration = transcriber.duration_seconds()
    segments = result['transcriptions']
    if not segments:
        record('时间戳对齐', False, '没有识别结果')
        return

    # 静音段之后的分句时间戳应当已经过了插入的静音区间
    after = [s for s in segments if s['start_time'] > len(audio) / 16000 / 2]
    detail = (f'{len(segments)} 句，录音 {duration:.1f}s，'
              f'末尾 {segments[-1]["end_time"]:.1f}s')
    record('时间戳与音频对齐（含 3 秒静音）',
           bool(after) and segments[-1]['end_time'] >= duration - 5.0, detail)


def main():
    print('=' * 68)
    print('实时转写链路准确率评测')
    print('=' * 68)
    for func in (test_realtime_cer, test_silence_no_hallucination, test_timestamp_progress):
        try:
            func()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            record(func.__name__, False, f'异常: {exc}')

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print('\n' + '=' * 68)
    print(f'结果: {passed}/{len(RESULTS)} 通过')
    print('=' * 68)
    return 0 if passed == len(RESULTS) else 1


if __name__ == '__main__':
    sys.exit(main())
