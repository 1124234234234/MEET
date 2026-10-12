"""
说话人识别回归测试：真实录音 + 合成音频的说话人数与标注正确性。

为什么单独有这个文件：
    说话人分离牵扯两条互相矛盾的判据 —— 并得太狠，真实存在但发言少的人会被
    并掉（会议里所有人都标成 SPEAKER_00）；并得太松，单说话人录音会被拆出
    一个「幽灵说话人」。两者都真实发生过，因此这里把口径固定下来，改动
    speaker_diarization 后必须跑通。

覆盖：
  1. 人数判定：合成 3 人会议 → 3；合成 2 人对话 → 2；真实单侧通话 → 1
  2. 混淆率：合成场景保持 < 5%（需求指标）
  3. 少言说话人不被并掉（构造「主说话人 + 两个窗口的次说话人」场景）
  4. 碎片簇判据本身（单测 _merge_fragment_clusters）

运行：python tests/test_speaker_recognition.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_DIR = os.path.join(BASE_DIR, 'tests', 'test_audio_files')

CONFUSION_TARGET = 0.05
RESULTS = []


def record(name, passed, detail=''):
    RESULTS.append((name, passed, detail))
    print(f'  [{"PASS" if passed else "FAIL"}] {name}' + (f' — {detail}' if detail else ''))
    return passed


# ---------------------------------------------------------------- 人数与混淆率

def test_speaker_counts():
    """合成音频的人数必须完全正确，真实录音必须判成单人。"""
    print('\n[1] 说话人数判定')
    from modules import speaker_diarization

    cases = [
        ('gt_meeting.wav', 3, '合成 3 人会议'),
        ('gt_2spk.wav', 2, '合成 2 人对话'),
        (os.path.join(BASE_DIR, '民族大道辅路.m4a'), 1, '真实单侧通话 1'),
        (os.path.join(BASE_DIR, '民族大道辅路 2.m4a'), 1, '真实单侧通话 2'),
    ]
    for filename, expected, label in cases:
        path = filename if os.path.isabs(filename) else os.path.join(AUDIO_DIR, filename)
        if not os.path.exists(path):
            record(f'{label} 人数', False, f'缺少音频 {os.path.basename(path)}')
            continue
        try:
            turns = speaker_diarization.diarize(path)
        except Exception as exc:
            record(f'{label} 人数', False, f'{exc.__class__.__name__}: {exc}')
            continue
        speakers = {turn['speaker'] for turn in turns}
        record(f'{label} 人数 {len(speakers)}（期望 {expected}）',
               len(speakers) == expected, f'{len(turns)} 段')


def test_confusion_rate():
    """合成场景的逐帧混淆率必须达标（需求 < 5%）。"""
    print('\n[2] 混淆率（需求指标）')
    from modules import speaker_diarization

    for scenario, expected in (('gt_meeting', 3), ('gt_2spk', 2)):
        wav = os.path.join(AUDIO_DIR, f'{scenario}.wav')
        gt_path = os.path.join(AUDIO_DIR, f'{scenario}.json')
        if not (os.path.exists(wav) and os.path.exists(gt_path)):
            record(f'{scenario} 混淆率', False, '缺少标注音频')
            continue
        with open(gt_path, encoding='utf-8') as handle:
            gt = json.load(handle)
        reference = [{'start': t['start'], 'end': t['end'], 'speaker': t['speaker']}
                     for t in gt['turns']]
        metrics = speaker_diarization.evaluate(reference, speaker_diarization.diarize(wav))
        record(f'{scenario} 混淆率={metrics["confusion_rate"]:.2%} (<{CONFUSION_TARGET:.0%})，'
               f'人数 {metrics["hyp_speakers"]}/{metrics["ref_speakers"]}',
               metrics['confusion_rate'] < CONFUSION_TARGET
               and metrics['speaker_count_error'] == 0)


def test_attribution():
    """「谁在什么时候说了什么」：逐句说话人归属正确率。"""
    print('\n[3] 逐句说话人归属')
    from modules import asr_engine, speaker_diarization

    wav = os.path.join(AUDIO_DIR, 'gt_meeting.wav')
    gt_path = os.path.join(AUDIO_DIR, 'gt_meeting.json')
    if not (os.path.exists(wav) and os.path.exists(gt_path)):
        record('说话人归属', False, '缺少标注音频')
        return
    with open(gt_path, encoding='utf-8') as handle:
        gt = json.load(handle)

    asr = asr_engine.transcribe(wav, language='zh')
    diar = speaker_diarization.diarize(wav)
    segments = [{'speaker': 'SPEAKER_00', 'text': s['text'],
                 'start_time': s['start'], 'end_time': s['end']} for s in asr['segments']]
    labelled = speaker_diarization.assign_speakers_to_segments(segments, diar)

    correct = checked = 0
    for segment in labelled:
        center = (segment['start_time'] + segment['end_time']) / 2
        truth = next((turn['speaker'] for turn in gt['turns']
                      if turn['start'] <= center <= turn['end']), None)
        if truth is None:
            continue
        checked += 1
        correct += 1 if segment['speaker'] == truth else 0
    if not checked:
        record('说话人归属', False, '没有可比对的分句')
        return
    accuracy = correct / checked
    record(f'说话人归属正确率 {accuracy:.1%}（{correct}/{checked} 句）', accuracy >= 0.9)


# ---------------------------------------------------------------- 少言说话人 / 碎片簇

def test_quiet_speaker_kept():
    """
    发言很少但确实在场的说话人不能被并掉。

    构造：40 个窗口的主说话人（占比 87%）+ 3 个窗口的次说话人（6.5%，不连续）+
    1 个 1 窗口/1s 的碎片（2%，连续）。期望：碎片被并掉，次说话人保留 → 2 人。
    """
    print('\n[4] 少言说话人不被并掉')
    from modules import speaker_diarization as sd

    rng = np.random.default_rng(20261012)
    main_centroid = rng.normal(size=64)
    second_centroid = main_centroid + rng.normal(scale=0.9, size=64)
    fragment = main_centroid + rng.normal(scale=0.5, size=64)

    def unit(vector):
        return vector / np.linalg.norm(vector)

    embeddings = []
    for _ in range(40):
        embeddings.append(unit(main_centroid + rng.normal(scale=0.1, size=64)))
    for _ in range(3):
        embeddings.append(unit(second_centroid + rng.normal(scale=0.1, size=64)))
    embeddings.append(unit(fragment))
    embeddings = np.asarray(embeddings, dtype=np.float32)

    # 窗口时间：主说话人 0~40s，次说话人分散在 45s / 60s / 75s（三处独立发言），
    # 碎片紧跟主说话人（40.5s，1 个窗口）
    windows = [(i * 1.0, i * 1.0 + 1.5) for i in range(40)]
    windows.append((40.5, 42.0))
    windows += [(45.0, 46.5), (60.0, 61.5), (75.0, 76.5)]

    labels = sd._cluster_once(embeddings, 3)
    merged = sd._merge_fragment_clusters(embeddings, labels, windows)
    speakers = len(set(int(v) for v in merged))
    record(f'少言说话人保留、碎片并掉（人数 {speakers}，期望 2）', speakers == 2)


def test_fragment_criteria():
    """碎片簇判据：1~2 个窗口、两秒内、单段、占比小 → 并；否则保留。"""
    print('\n[5] 碎片簇判据')
    from modules import speaker_diarization as sd

    rng = np.random.default_rng(7)
    base = rng.normal(size=64)
    base = base / np.linalg.norm(base)

    def cluster(center, count, jitter=0.05):
        return [center + rng.normal(scale=jitter, size=64) for _ in range(count)]

    other = -base  # 另一个说话人（方向相反）
    embeddings = np.asarray(cluster(base, 30) + cluster(other, 20) + cluster(base * 0.9, 2),
                            dtype=np.float32)
    windows = [(i * 1.0, i * 1.0 + 1.5) for i in range(52)]
    labels = sd._cluster_once(embeddings, 3)
    merged = sd._merge_fragment_clusters(embeddings, labels, windows)
    record('碎片簇被并与最近簇（人数 2）', len(set(int(v) for v in merged)) == 2)

    # 同样 2 个窗口，但占比够大（20 个窗口里占 10%）→ 视为真说话人，保留
    embeddings2 = np.asarray(cluster(base, 18) + cluster(other, 2), dtype=np.float32)
    windows2 = [(i * 1.0, i * 1.0 + 1.5) for i in range(20)]
    labels2 = sd._cluster_once(embeddings2, 2)
    merged2 = sd._merge_fragment_clusters(embeddings2, labels2, windows2)
    record('占比 10% 的 2 窗口簇保留（人数 2）', len(set(int(v) for v in merged2)) == 2)


# ---------------------------------------------------------------- 实时增量标注

def test_realtime_labelling():
    """实时转写过程中也要能区分说话人（不再一律 SPEAKER_00）。"""
    print('\n[6] 实时转写的增量说话人标注')
    from modules.audio_io import load_audio
    from modules.funasr_transcriber import FunASRRealtimeTranscriber

    wav = os.path.join(AUDIO_DIR, 'gt_2spk.wav')
    if not os.path.exists(wav):
        record('实时说话人标注', False, '缺少标注音频')
        return

    audio = load_audio(wav, sr=16000)
    transcriber = FunASRRealtimeTranscriber(language='zh', sample_rate=16000)
    step = int(16000 * 0.5)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    live_speakers = set()
    for start in range(0, len(pcm), step):
        result = transcriber.add_audio_chunk(pcm[start:start + step].tobytes())
        if result:
            for seg in result['history']:
                live_speakers.add(seg['speaker'])

    final = transcriber.get_final_result()
    final_speakers = {seg['speaker'] for seg in final['transcriptions']}
    record(f'录制过程中已区分说话人 {sorted(live_speakers)}（期望 ≥2）',
           len(live_speakers) >= 2)
    record(f'最终结果说话人数 {len(final_speakers)}（期望 2）', len(final_speakers) == 2)
    record(f'每句都带说话人标注（{len(final["transcriptions"])} 句）',
           all(seg.get('speaker') for seg in final['transcriptions']))


def main():
    print('=' * 68)
    print('说话人识别回归测试')
    print('=' * 68)
    for func in (test_speaker_counts, test_confusion_rate, test_attribution,
                 test_quiet_speaker_kept, test_fragment_criteria, test_realtime_labelling):
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
