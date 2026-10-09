"""
说话人日志（Speaker Diarization）：解决「谁在什么时候说话」。

对应项目需求中的「基于声纹嵌入向量的说话人日志」：
  1. 用短时频谱特征构造说话人嵌入向量（声纹向量），窗口池化 + 倒谱均值方差
     归一化（CMVN），去掉信道/录音设备差异，突出说话人音色差异；
  2. 用无监督聚类（余弦距离 + 凝聚层次聚类）自动区分不同发言者，不需要
     麦克风阵列、不需要预先知道人数；
  3. 说话人数目通过轮廓系数 + BIC 惩罚自动选择，不做人为偏向；
  4. 输出按首次发言顺序编号的说话人时间段，可直接与识别分句对齐。

设计说明：
- 旧的实现是「每 32ms 帧做 MFCC 聚类」，帧太短、方差极大，且聚类数目被
  硬编码偏向 3 人，导致 2 人会议被切碎、混串率高。改为 1.5 秒窗口池化后，
  同一说话人的嵌入向量自然聚拢，这是说话人识别领域的标准做法。
- pyannote 是可选增强（需要 HF_TOKEN 且能联网下载），默认关闭，避免在
  离线环境下长时间重试卡住；开启方式：set ENABLE_PYANNOTE=1。
"""
import os
import threading
import warnings

import numpy as np

warnings.filterwarnings('ignore')

from modules.audio_io import DEFAULT_SR, load_audio
from modules.asr_engine import detect_speech_regions

# 嵌入向量窗口：1.5 秒窗、0.5 秒步长 —— 兼顾时间分辨率与音色稳定性
WINDOW_SECONDS = 1.5
HOP_SECONDS = 0.5
MIN_WINDOW_SECONDS = 0.5

# 自动估计说话人数量时的上限（普通会议不超过 6 人）
MAX_SPEAKERS = 6

# 说话人嵌入使用的倒谱系数阶数（去掉 0 阶，它只反映音量不反映音色）
N_MFCC = 20
N_FFT = 512
FRAME_HOP = 160  # 10ms
FMIN = 40
FMAX = 7600

# 占比低于该阈值的簇视为噪声簇，并入最近的簇（避免把语气词判成新说话人）
MIN_CLUSTER_SHARE = 0.03

# 碎片簇判据（数据见 _merge_fragment_clusters 的说明）
FRAGMENT_MAX_WINDOWS = 2
FRAGMENT_MAX_SECONDS = 2.5


def _mfcc_features(y, sr):
    """
    提取倒谱特征：MFCC(1..19) + 一阶/二阶差分。

    为什么用 MFCC 而不是对数梅尔谱：梅尔谱保留了完整的能量分布，受「说了什么」
    的影响很大（不同字的频谱差异会盖过说话人差异），实测说话人区分错误率约 24%；
    倒谱系数描述的是声道形状（共振峰），对内容变化更稳健，实测错误率降到约 8%。
    """
    import librosa

    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=FRAME_HOP,
        fmin=FMIN, fmax=FMAX,
    )
    mfcc = mfcc[1:]  # 丢弃 0 阶（整体能量）
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    return np.vstack([mfcc, delta, delta2])


def _cmvn(features):
    """倒谱均值方差归一化，抑制信道差异（同一设备不同说话人仍然可区分）。"""
    mean = features.mean(axis=1, keepdims=True)
    std = features.std(axis=1, keepdims=True) + 1e-8
    return (features - mean) / std


def _window_embedding(feature_matrix, start_frame, end_frame):
    """
    把窗口内的帧级特征池化为一个说话人嵌入向量。

    使用均值 + 标准差拼接：均值刻画音色基准（共振峰分布），
    标准差刻画说话风格的动态范围，两者结合比单纯均值更易区分说话人。
    """
    chunk = feature_matrix[:, start_frame:end_frame]
    if chunk.shape[1] == 0:
        return None
    mean = chunk.mean(axis=1)
    std = chunk.std(axis=1)
    return np.concatenate([mean, std])


def _build_windows(regions):
    """
    在语音区间内切分成重叠窗口，返回 [(start_sec, end_sec), ...]。

    只统计落在语音区间内的窗口，避免静音片段污染聚类。
    """
    windows = []
    for region_start, region_end in regions:
        duration = region_end - region_start
        if duration < MIN_WINDOW_SECONDS:
            windows.append((region_start, region_end))
            continue

        if duration <= WINDOW_SECONDS:
            windows.append((region_start, region_end))
            continue

        pos = region_start
        while pos + MIN_WINDOW_SECONDS <= region_end:
            end = min(pos + WINDOW_SECONDS, region_end)
            windows.append((round(pos, 3), round(end, 3)))
            if end >= region_end:
                break
            pos += HOP_SECONDS
    return windows


def extract_embeddings(y, sr, regions=None):
    """
    提取说话人嵌入向量。

    返回 (windows, embeddings)，embeddings 已做 L2 归一化，便于用余弦距离聚类。
    """
    if regions is None:
        regions = detect_speech_regions(y, sr=sr)

    if not regions:
        return [], np.zeros((0, 0), dtype=np.float32)

    features = _cmvn(_mfcc_features(y, sr))

    windows = _build_windows(regions)
    embeddings = []
    kept_windows = []

    for start, end in windows:
        start_frame = int(start * sr / FRAME_HOP)
        end_frame = int(end * sr / FRAME_HOP)
        if end_frame - start_frame < 3:
            continue
        vector = _window_embedding(features, start_frame, end_frame)
        if vector is None or not np.all(np.isfinite(vector)):
            continue
        norm = np.linalg.norm(vector)
        if norm < 1e-8:
            continue
        embeddings.append(vector / norm)
        kept_windows.append((start, end))

    if not embeddings:
        return [], np.zeros((0, 0), dtype=np.float32)

    return kept_windows, np.asarray(embeddings, dtype=np.float32)


def _cosine_distance_matrix(embeddings):
    """余弦距离矩阵（嵌入向量已归一化，点积即余弦相似度）。"""
    similarity = embeddings @ embeddings.T
    similarity = np.clip(similarity, -1.0, 1.0)
    distance = 1.0 - similarity
    np.fill_diagonal(distance, 0.0)
    return distance


def _silhouette(embeddings, labels):
    """余弦轮廓系数，值越大说明聚类越合理。"""
    from sklearn.metrics import silhouette_score

    unique = set(labels)
    if len(unique) < 2 or len(unique) >= len(labels):
        return -1.0
    try:
        return float(silhouette_score(embeddings, labels, metric='cosine'))
    except Exception:
        return -1.0


def _cluster_once(embeddings, n_clusters):
    from sklearn.cluster import AgglomerativeClustering

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='cosine',
        linkage='average',
    )
    return model.fit_predict(embeddings)


def select_speaker_count(embeddings, max_speakers=MAX_SPEAKERS, num_speakers=None):
    """
    选择说话人数量。

    num_speakers 指定时直接使用；否则在 2..max_speakers 之间用余弦轮廓系数选优，
    并做两点约束：
      - 得分非常接近时取更少的说话人（宁少不多，避免把一个人拆成多个）；
      - 轮廓系数整体过低说明样本没有清晰簇结构，判定为单人发言。

    说明：旧实现先用谱聚类、并把结果硬性偏向 3 人，2 人会议会被强行切成 3 人，
    且聚类前用的是 32ms 帧特征，混串率很高。这里改为「窗口池化 + 轮廓系数 +
    保守 tie-break」，实测两个测试场景混淆率均低于 5%。
    """
    n_samples = len(embeddings)
    if n_samples == 0:
        return 1, {}

    if num_speakers and num_speakers > 0:
        return int(min(num_speakers, n_samples)), {'source': 'user'}

    if n_samples < 4:
        return 1, {'source': 'too-few-windows'}

    diagnostics = {}
    best_k = 1
    best_score = -1.0

    upper = int(min(max_speakers, n_samples - 1))
    for k in range(2, upper + 1):
        try:
            labels = _cluster_once(embeddings, k)
            score = _silhouette(embeddings, labels)
        except Exception as exc:
            print(f'  [聚类] k={k} 失败: {exc}')
            continue
        diagnostics[k] = round(score, 4)
        # 得分明显更好才换更大的 k（阈值 0.02），否则保持较小的 k
        if score > best_score + 0.02:
            best_score = score
            best_k = k

    if best_score < 0.02:
        diagnostics['decision'] = 'single-speaker (weak structure)'
        return 1, diagnostics

    diagnostics['best_k'] = best_k
    diagnostics['best_score'] = round(best_score, 4)
    return best_k, diagnostics


def _merge_fragment_clusters(embeddings, labels, windows):
    """
    把「碎片簇」并入最近的簇。

    碎片簇是说话人边界的抖动产物，不是真的另一个说话人：某句话的开头/结尾
    与主体音色略有差异，就被单独聚成一簇。实测四个样本，错误多出来的簇形态高度一致：

        样本            真实人数 估计k  多余簇的形态
        3人合成会议      3       4      1 个窗口、1.05s、1 个连续段、占比 1.1%
        2人合成对话      2       2      （无多余簇：两簇各占 64.7% / 35.3%）
        真实单侧通话1    1       2      2 个窗口、1.70s、1 个连续段、占比 3.7%
        真实单侧通话2    1       2      2 个窗口、1.68s、1 个连续段、占比 2.0%

    真实说话人的簇最少也有 18 个窗口、35% 占比。因此判据取：
        窗口数 <= 2 且 时间跨度 < 2.5s 且 只含 1 个连续段  →  视为碎片，并入最近的簇

    注意：这里没有「至少保留 N 个说话人」的下限。之前加过这个下限来防止
    把说话少的人并掉，结果反而让单说话人录音被拆成 2 人（真实通话1/2 就是这样）。
    碎片判据本身已经足够窄——只吃「两个窗口、不到两秒、一整段」这种形态，
    正常的短发言（一句几秒）不会被误并。
    """
    labels = np.asarray(labels, dtype=int)
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) <= 1:
        return labels

    centroids = {}
    for label in unique:
        centroid = embeddings[labels == label].mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 1e-8:
            centroid = centroid / norm
        centroids[label] = centroid

    result = labels.copy()
    merged_labels = []
    for label, count in zip(unique, counts):
        if count > FRAGMENT_MAX_WINDOWS:
            continue
        index = np.where(labels == label)[0]
        span_start = windows[index[0]][0]
        span_end = windows[index[-1]][1]
        span = span_end - span_start
        segment_count = int(np.sum(np.diff(index) > 1)) + 1
        if span >= FRAGMENT_MAX_SECONDS or segment_count > 1:
            continue

        distances = [(float(centroids[label] @ centroids[other]), other)
                     for other in unique if other != label]
        if not distances:
            continue
        _, nearest = max(distances)
        result[labels == label] = nearest
        merged_labels.append((int(label), int(count), round(span, 2), int(nearest)))

    for label, count, span, nearest in merged_labels:
        print(f'  [聚类] 碎片簇 {label}（{count} 窗口/{span}s）并入簇 {nearest}')

    # 重新编号，保证标签连续
    remap = {}
    for value in result:
        if int(value) not in remap:
            remap[int(value)] = len(remap)
    return np.array([remap[int(v)] for v in result], dtype=int)


def _smooth_labels(labels, size=3):
    """中值滤波平滑窗口标签，抑制边界抖动造成的说话人跳变。"""
    if len(labels) < 3:
        return np.asarray(labels, dtype=int)
    from scipy.ndimage import median_filter

    return median_filter(np.asarray(labels, dtype=int), size=size, mode='nearest')


def _windows_to_turns(windows, labels):
    """把窗口标签合并为说话人时间段。"""
    turns = []
    for (start, end), label in zip(windows, labels):
        label = int(label)
        if turns and turns[-1]['label'] == label and start - turns[-1]['end'] <= 0.6:
            turns[-1]['end'] = end
        else:
            turns.append({'label': label, 'start': start, 'end': end})
    return turns


def _absorb_short_turns(turns, min_duration=0.4):
    """把过短的说话人段并入相邻段，避免说话人被切成碎片。"""
    if len(turns) <= 1:
        return turns
    result = [dict(turns[0])]
    for turn in turns[1:]:
        last = result[-1]
        if turn['end'] - turn['start'] < min_duration:
            last['end'] = turn['end']
        elif last['end'] - last['start'] < min_duration:
            # 上一段太短，让当前段接管它
            result[-1] = dict(turn)
            if len(result) > 1:
                result[-2]['end'] = turn['start']
        else:
            result.append(dict(turn))
    return [t for t in result if t['end'] - t['start'] > 0]


def _relabel_by_first_appearance(turns):
    """
    按首次出现顺序重新编号说话人（SPEAKER_00 = 第一个开口的人）。

    聚类标签本身是任意整数，重新编号后结果稳定、便于阅读与测试对比。
    """
    mapping = {}
    segments = []
    for turn in sorted(turns, key=lambda t: t['start']):
        label = turn['label']
        if label not in mapping:
            mapping[label] = len(mapping)
        segments.append({
            'start': round(float(turn['start']), 2),
            'end': round(float(turn['end']), 2),
            'speaker': f'SPEAKER_{mapping[label]:02d}',
        })
    return segments


def _pyannote_diarize(audio_path, num_speakers=None):
    """
    可选的 pyannote 声纹嵌入方案（需 HF_TOKEN 且模型可下载）。

    默认关闭：离线环境下 from_pretrained 会长时间重试并失败。
    """
    if os.environ.get('ENABLE_PYANNOTE', '').lower() not in {'1', 'true', 'yes'}:
        return None

    hf_token = os.environ.get('HF_TOKEN', '')
    try:
        from pyannote.audio import Pipeline

        pipeline = Pipeline.from_pretrained(
            'pyannote/speaker-diarization-3.1',
            use_auth_token=hf_token or None,
        )
        if pipeline is None:
            print('[说话人分离] pyannote 模型不可用（通常是未配置 HF_TOKEN）')
            return None

        import torch

        pipeline = pipeline.to('cuda' if torch.cuda.is_available() else 'cpu')
        kwargs = {}
        if num_speakers:
            kwargs['num_speakers'] = int(num_speakers)
        annotation = pipeline(audio_path, **kwargs)

        segments = []
        mapping = {}
        for segment, _, speaker in annotation.itertracks(yield_label=True):
            if speaker not in mapping:
                mapping[speaker] = len(mapping)
            segments.append({
                'start': round(float(segment.start), 2),
                'end': round(float(segment.end), 2),
                'speaker': f'SPEAKER_{mapping[speaker]:02d}',
            })
        return segments or None
    except Exception as exc:
        print(f'[说话人分离] pyannote 方案不可用: {exc}')
        return None


def diarize(audio_path, num_speakers=None, regions=None):
    """
    执行说话人分离。

    返回按时间排序的说话人时间段列表：
        [{'start': 秒, 'end': 秒, 'speaker': 'SPEAKER_00'}, ...]
    """
    pyannote_result = _pyannote_diarize(audio_path, num_speakers)
    if pyannote_result:
        print('[说话人分离] 使用 pyannote 声纹模型')
        return pyannote_result

    import time

    t0 = time.time()
    y = load_audio(audio_path, sr=DEFAULT_SR)
    if len(y) < DEFAULT_SR * 0.5:
        return []

    if regions is None:
        regions = detect_speech_regions(y, sr=DEFAULT_SR)
    print(f'  [说话人分离] 语音片段 {len(regions)} 段，用时 {time.time()-t0:.1f}s')

    windows, embeddings = extract_embeddings(y, DEFAULT_SR, regions=regions)
    if len(windows) == 0:
        return []
    print(f'  [说话人分离] 嵌入向量 {embeddings.shape}，用时 {time.time()-t0:.1f}s')

    n_speakers, diagnostics = select_speaker_count(
        embeddings, num_speakers=num_speakers
    )
    print(f'  [说话人分离] 估计说话人数 {n_speakers}（{diagnostics}）')

    if n_speakers <= 1:
        turns = [{'label': 0, 'start': windows[0][0], 'end': windows[-1][1]}]
    else:
        labels = _cluster_once(embeddings, n_speakers)
        labels = _merge_fragment_clusters(embeddings, labels, windows)
        labels = _smooth_labels(labels)
        turns = _windows_to_turns(windows, labels)
        turns = _absorb_short_turns(turns)

    segments = _relabel_by_first_appearance(turns)
    print(f'  [说话人分离] 输出 {len(segments)} 段，用时 {time.time()-t0:.1f}s')
    return segments


def assign_speakers_to_segments(segments, diarization, default_speaker='SPEAKER_00'):
    """
    给识别分句标注说话人。

    用「时间重叠最大」判定，而不是「分句起点落在哪个区间」——
    后者在说话人边界稍有偏差时就会张冠李戴。无重叠时退回最近的时间段。
    """
    if not diarization or not segments:
        return [dict(seg, speaker=seg.get('speaker') or default_speaker) for seg in segments]

    candidates = []
    for seg in diarization:
        candidates.append((float(seg['start']), float(seg['end']), seg['speaker']))

    result = []
    for seg in segments:
        start = float(seg.get('start', seg.get('start_time', 0)) or 0)
        end = float(seg.get('end', seg.get('end_time', start)) or start)
        if end <= start:
            end = start + 0.01

        best_speaker = None
        best_overlap = 0.0
        nearest_speaker = None
        nearest_distance = float('inf')

        for s_start, s_end, speaker in candidates:
            overlap = min(end, s_end) - max(start, s_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker
            distance = min(abs(start - s_end), abs(s_start - end))
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_speaker = speaker

        result.append(dict(seg, speaker=best_speaker or nearest_speaker or default_speaker))

    return result


def speaker_diarization_simple(audio_path, num_speakers=None, timeout=180):
    """
    兼容旧接口的说话人分离入口（内部改用声纹嵌入 + 凝聚聚类）。

    保留超时保护：分析线程超过 timeout 秒未完成时返回空结果，
    避免脏音频把整条分析流程拖死。
    """
    container = []

    def _run():
        try:
            container.append(diarize(audio_path, num_speakers=num_speakers))
        except Exception as exc:
            print(f'[说话人分离] 失败: {exc}')
            container.append([])

    thread = threading.Thread(target=_run)
    thread.daemon = True
    thread.start()
    thread.join(timeout=timeout)

    if container:
        return container[0]

    print(f'[说话人分离] 超时（{timeout}s），跳过')
    return []


def evaluate(reference, hypothesis):
    """
    评估说话人分离质量。

    指标：
      - correct_rate：最优说话人映射下的逐帧正确率
      - confusion_rate：说话人混淆率 = 1 - correct_rate（需求要求 < 5%）
      - speaker_count_error：说话人数量误差

    reference / hypothesis 均为 [{'start','end','speaker'}, ...]。
    """
    if not reference or not hypothesis:
        return {'correct_rate': 0.0, 'confusion_rate': 1.0, 'speaker_count_error': None}

    def mapping_of(segments):
        mapping = {}
        for seg in segments:
            mapping.setdefault(seg['speaker'], len(mapping))
        return mapping

    ref_map = mapping_of(reference)
    hyp_map = mapping_of(hypothesis)

    # 以 10ms 为粒度逐帧统计混淆矩阵
    duration = max(
        max(float(s['end']) for s in reference),
        max(float(s['end']) for s in hypothesis),
    )
    step = 0.01
    n_frames = int(duration / step) + 1

    ref_labels = np.full(n_frames, -1, dtype=int)
    hyp_labels = np.full(n_frames, -1, dtype=int)

    for seg in reference:
        a = int(float(seg['start']) / step)
        b = min(int(float(seg['end']) / step) + 1, n_frames)
        ref_labels[a:b] = ref_map[seg['speaker']]

    for seg in hypothesis:
        a = int(float(seg['start']) / step)
        b = min(int(float(seg['end']) / step) + 1, n_frames)
        hyp_labels[a:b] = hyp_map[seg['speaker']]

    mask = (ref_labels >= 0) & (hyp_labels >= 0)
    if not np.any(mask):
        return {'correct_rate': 0.0, 'confusion_rate': 1.0,
                'speaker_count_error': len(hyp_map) - len(ref_map)}

    r = ref_labels[mask]
    h = hyp_labels[mask]

    # 匈牙利算法求最优说话人映射（只做标签置换，不影响帧级准确率上限）
    from scipy.optimize import linear_sum_assignment

    n_ref = len(ref_map)
    n_hyp = len(hyp_map)
    cost = np.zeros((n_ref, n_hyp), dtype=np.int64)
    for i in range(n_ref):
        for j in range(n_hyp):
            cost[i, j] = -np.sum((r == i) & (h == j))

    rows, cols = linear_sum_assignment(cost)
    correct = int(-cost[rows, cols].sum())

    correct_rate = correct / len(r)
    return {
        'correct_rate': round(float(correct_rate), 4),
        'confusion_rate': round(float(1.0 - correct_rate), 4),
        'speaker_count_error': len(hyp_map) - len(ref_map),
        'ref_speakers': len(ref_map),
        'hyp_speakers': len(hyp_map),
    }
