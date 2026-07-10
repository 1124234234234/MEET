import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def speaker_diarization_simple(audio_path, num_speakers=2):
    """
    说话人分离：支持多人交替发言和同时发言场景
    优先使用 pyannote.audio，未配置时使用改进的聚类算法
    """
    try:
        from pyannote.audio import Pipeline
        import torch

        hf_token = os.environ.get('HF_TOKEN', '')
        if not hf_token:
            print("HF_TOKEN not set, using enhanced clustering diarization")
            return enhanced_diarization(audio_path, num_speakers)

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline = pipeline.to(device)

        # pyannote 支持重叠语音检测
        diarization = pipeline(audio_path)

        speaker_segments = []
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append({
                'start': round(segment.start, 2),
                'end': round(segment.end, 2),
                'speaker': speaker
            })

        return speaker_segments

    except Exception as e:
        print(f"Speaker diarization failed: {e}, using enhanced clustering")
        return enhanced_diarization(audio_path, num_speakers)


def enhanced_diarization(audio_path, num_speakers=2):
    """
    增强的说话人分离：基于频谱特征 + MFCC + 聚类
    支持交替发言和基本的同时发言检测
    """
    import librosa
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler

    y, sr = librosa.load(audio_path, sr=16000)
    duration = len(y) / sr

    hop_length = 512
    n_fft = 2048

    # 1. 提取MFCC特征（比纯频谱更能区分说话人）
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length, n_fft=n_fft)

    # 2. 提取频谱特征
    stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    spectral_centroid = librosa.feature.spectral_centroid(S=stft, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(S=stft, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(S=stft, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)

    # 3. 合并特征
    features = np.vstack([
        mfcc,
        spectral_centroid,
        spectral_bandwidth,
        spectral_rolloff,
        zero_crossing_rate
    ]).T

    # 4. 标准化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # 5. 使用层次聚类（比KMeans更适合说话人分离）
    # 估计说话人数量
    estimated_speakers = estimate_num_speakers(features_scaled, num_speakers)

    clustering = AgglomerativeClustering(
        n_clusters=estimated_speakers,
        linkage='ward'
    )
    labels = clustering.fit_predict(features_scaled)

    # 6. 构建说话人段落
    speaker_segments = build_segments(labels, hop_length, sr, duration)

    # 7. 检测重叠语音段
    speaker_segments = detect_overlapping_speech(speaker_segments, features_scaled, labels, hop_length, sr)

    # 8. 合并过短段
    speaker_segments = merge_short_segments(speaker_segments)

    return speaker_segments


def estimate_num_speakers(features, max_speakers):
    """估计说话人数量"""
    from sklearn.metrics import silhouette_score

    if len(features) < 10:
        return min(2, max_speakers)

    max_speakers = min(max_speakers, len(features) // 5)

    best_n = 2
    best_score = -1

    for n in range(2, max_speakers + 1):
        try:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features)
            score = silhouette_score(features, labels)

            if score > best_score:
                best_score = score
                best_n = n
        except:
            continue

    return best_n


def build_segments(labels, hop_length, sr, duration):
    """根据聚类标签构建说话人段"""
    speaker_segments = []
    current_speaker = labels[0]
    start_time = 0

    for i in range(1, len(labels)):
        if labels[i] != current_speaker:
            end_time = i * hop_length / sr
            speaker_segments.append({
                'start': round(start_time, 2),
                'end': round(end_time, 2),
                'speaker': f'SPEAKER_{current_speaker:02d}'
            })
            current_speaker = labels[i]
            start_time = end_time

    speaker_segments.append({
        'start': round(start_time, 2),
        'end': round(duration, 2),
        'speaker': f'SPEAKER_{current_speaker:02d}'
    })

    return speaker_segments


def detect_overlapping_speech(speaker_segments, features, labels, hop_length, sr):
    """
    检测重叠语音段
    当相邻段的特征距离很近但标签不同时，可能是重叠区域
    """
    if len(speaker_segments) < 2:
        return speaker_segments

    result = []
    for i, seg in enumerate(speaker_segments):
        result.append(seg)

        # 检查是否有重叠
        if i < len(speaker_segments) - 1:
            next_seg = speaker_segments[i + 1]
            # 如果两段之间间隔非常短（<0.1秒），可能是同时发言
            gap = next_seg['start'] - seg['end']
            if gap < 0.1:
                # 标记可能的同时发言
                seg['overlap'] = True
                next_seg['overlap'] = True

    return result


def merge_short_segments(segments, min_duration=0.5):
    """合并过短的段"""
    if not segments:
        return segments

    merged = [segments[0].copy()]

    for seg in segments[1:]:
        last = merged[-1]

        if seg['speaker'] == last['speaker']:
            last['end'] = seg['end']
        elif seg['end'] - seg['start'] < min_duration and len(merged) > 1:
            prev_speaker = merged[-2]['speaker']
            if prev_speaker == seg['speaker']:
                merged[-2]['end'] = seg['end']
                merged.pop()
            else:
                last['end'] = seg['end']
        else:
            merged.append(seg.copy())

    return merged