import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def speaker_diarization_simple(audio_path, num_speakers=None, timeout=60):
    """
    说话人分离：支持多人交替发言和同时发言场景
    优先使用 pyannote.audio，未配置时使用改进的聚类算法
    添加超时机制，避免分析过长时间卡住
    num_speakers=None 时自动估计说话人数量
    """
    import threading

    result_container = []

    def run_diarization():
        try:
            from pyannote.audio import Pipeline
            import torch

            hf_token = os.environ.get('HF_TOKEN', '')
            if not hf_token:
                print("HF_TOKEN not set, using enhanced clustering diarization")
                result_container.append(enhanced_diarization(audio_path, num_speakers))
                return

            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )

            device = "cuda" if torch.cuda.is_available() else "cpu"
            pipeline = pipeline.to(device)

            diarization = pipeline(audio_path)

            speaker_segments = []
            for segment, _, speaker in diarization.itertracks(yield_label=True):
                speaker_segments.append({
                    'start': round(segment.start, 2),
                    'end': round(segment.end, 2),
                    'speaker': speaker
                })

            result_container.append(speaker_segments)

        except Exception as e:
            print(f"Speaker diarization failed: {e}, using enhanced clustering")
            try:
                result_container.append(enhanced_diarization(audio_path, num_speakers))
            except Exception as e2:
                print(f"Enhanced diarization also failed: {e2}")
                result_container.append([])

    thread = threading.Thread(target=run_diarization)
    thread.daemon = True
    thread.start()
    thread.join(timeout=timeout)

    if result_container:
        return result_container[0]
    else:
        print(f"Speaker diarization timeout ({timeout}s), skipping")
        return []


def enhanced_diarization(audio_path, num_speakers=None):
    """
    增强的说话人分离：基于MFCC + 多种特征 + 优化聚类
    支持自动估计说话人数量
    """
    import time
    t0 = time.time()
    
    import librosa
    from sklearn.cluster import AgglomerativeClustering, SpectralClustering
    from sklearn.preprocessing import StandardScaler
    from sklearn.mixture import GaussianMixture

    y, sr = librosa.load(audio_path, sr=16000)
    duration = len(y) / sr
    print(f"  [diarization] audio loaded: {duration:.1f}s, {time.time()-t0:.1f}s")

    hop_length = 512
    n_fft = 2048

    # 1. 提取多种声学特征
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=hop_length, n_fft=n_fft)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    
    # 色度特征（音调相关）
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length, n_fft=n_fft)
    
    # 频谱质心（音色相关）
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length, n_fft=n_fft)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length, n_fft=n_fft)
    
    # 过零率
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)
    
    # 均方根能量
    rms = librosa.feature.rms(y=y, hop_length=hop_length)
    
    print(f"  [diarization] features extracted: MFCC={mfcc.shape}, {time.time()-t0:.1f}s")

    # 2. 组合所有特征
    features = np.vstack([
        mfcc,
        mfcc_delta * 0.5,
        mfcc_delta2 * 0.3,
        chroma * 0.3,
        spectral_centroid * 0.1,
        spectral_bandwidth * 0.1,
        zcr * 0.2,
        rms * 0.1,
    ]).T
    
    print(f"  [diarization] combined features: {features.shape}, {time.time()-t0:.1f}s")

    # 3. 标准化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # 4. 语音活动检测 - 只对有语音的帧进行聚类
    rms_flat = rms.flatten()
    rms_threshold = np.percentile(rms_flat, 20)
    voice_mask = rms_flat > rms_threshold
    
    print(f"  [diarization] voice frames: {np.sum(voice_mask)}/{len(voice_mask)}, {time.time()-t0:.1f}s")
    
    if np.sum(voice_mask) < 10:
        return []
    
    voice_features = features_scaled[voice_mask]

    # 5. 估计说话人数量
    if num_speakers is None or num_speakers <= 0:
        estimated_n = estimate_num_speakers_advanced(voice_features, max_speakers=6)
        print(f"  [diarization] estimated speakers: {estimated_n}, {time.time()-t0:.1f}s")
    else:
        estimated_n = num_speakers
        print(f"  [diarization] using specified speakers: {estimated_n}")

    # 6. 使用GMM + 层次聚类的组合方案
    try:
        # 先用GMM初始化
        gmm = GaussianMixture(
            n_components=estimated_n,
            random_state=42,
            n_init=3,
            max_iter=100
        )
        gmm_labels = gmm.fit_predict(voice_features)
        
        # 再用层次聚类精炼
        clustering = AgglomerativeClustering(
            n_clusters=estimated_n,
            linkage='ward',
            metric='euclidean'
        )
        labels_voice = clustering.fit_predict(voice_features)
        
        print(f"  [diarization] clustering done: {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"  [diarization] advanced clustering failed: {e}, using basic")
        clustering = AgglomerativeClustering(
            n_clusters=estimated_n,
            linkage='ward'
        )
        labels_voice = clustering.fit_predict(voice_features)

    # 7. 将标签映射回所有帧
    n_frames = len(features_scaled)
    labels = np.zeros(n_frames, dtype=int)
    voice_indices = np.where(voice_mask)[0]
    
    for i, idx in enumerate(voice_indices):
        labels[idx] = labels_voice[i]
    
    # 非语音帧标记为-1（后面会合并到相邻说话人）
    for i in range(n_frames):
        if not voice_mask[i]:
            # 找最近的语音帧
            left = i - 1
            right = i + 1
            while left >= 0 and not voice_mask[left]:
                left -= 1
            while right < n_frames and not voice_mask[right]:
                right += 1
            
            if left >= 0 and right < n_frames:
                labels[i] = labels[left] if (i - left) < (right - i) else labels[right]
            elif left >= 0:
                labels[i] = labels[left]
            elif right < n_frames:
                labels[i] = labels[right]

    # 8. 中值滤波平滑标签（减少抖动）
    from scipy.ndimage import median_filter
    labels_smooth = median_filter(labels, size=5)

    # 9. 构建说话人段落
    speaker_segments = build_segments(labels_smooth, hop_length, sr, duration)

    # 10. 优化后处理
    speaker_segments = merge_short_segments(speaker_segments, min_duration=0.3)
    speaker_segments = merge_same_speaker_adjacent(speaker_segments)
    
    print(f"  [diarization] total: {time.time()-t0:.1f}s, segments: {len(speaker_segments)}")
    
    return speaker_segments


def estimate_num_speakers_advanced(features, max_speakers=5):
    """
    估计说话人数量
    使用轮廓系数 + BIC准则综合判断
    """
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score
    from sklearn.cluster import KMeans
    import numpy as np
    
    n_samples = len(features)
    if n_samples < 100:
        return 2
    
    # 采样加速
    if n_samples > 3000:
        indices = np.random.choice(n_samples, 3000, replace=False)
        features_sample = features[indices]
    else:
        features_sample = features
    
    best_n = 2
    best_score = -1
    
    # 只尝试2-5个说话人
    for n in range(2, min(max_speakers, 5) + 1):
        try:
            # 用KMeans快速聚类计算轮廓系数
            kmeans = KMeans(n_clusters=n, random_state=42, n_init=5, max_iter=100)
            labels = kmeans.fit_predict(features_sample)
            
            # 轮廓系数越大越好
            if len(set(labels)) > 1:
                sil_score = silhouette_score(features_sample, labels, sample_size=min(500, len(features_sample)))
            else:
                sil_score = -1
            
            # 惩罚过多的说话人（轻微惩罚，避免过度分类）
            penalty = (n - 2) * 0.05
            adjusted_score = sil_score - penalty
            
            print(f"    [estimate] n={n}, silhouette={sil_score:.4f}, adjusted={adjusted_score:.4f}")
            
            if adjusted_score > best_score:
                best_score = adjusted_score
                best_n = n
        except Exception as e:
            print(f"    [estimate] n={n} failed: {e}")
            continue
    
    print(f"    [estimate] best_n={best_n}, score={best_score:.4f}")
    return best_n


def merge_same_speaker_adjacent(segments):
    """合并相邻的相同说话人段"""
    if not segments:
        return segments
    
    merged = [segments[0].copy()]
    
    for seg in segments[1:]:
        last = merged[-1]
        if seg['speaker'] == last['speaker']:
            last['end'] = seg['end']
        else:
            merged.append(seg.copy())
    
    return merged


def estimate_num_speakers(features, max_speakers):
    """估计说话人数量（快速版本）"""
    from sklearn.metrics import silhouette_score
    from sklearn.cluster import KMeans

    n_samples = len(features)
    if n_samples < 10:
        return min(2, max_speakers)

    # 限制最大尝试次数，避免太慢
    max_speakers = min(max_speakers, 4)  # 最多尝试 4 个说话人
    max_speakers = min(max_speakers, n_samples // 10)  # 确保有足够样本

    if max_speakers < 2:
        return 2

    # 采样加速：如果样本太多，只用部分样本评估
    if n_samples > 500:
        sample_indices = np.random.choice(n_samples, 500, replace=False)
        features_sample = features[sample_indices]
    else:
        features_sample = features

    best_n = 2
    best_score = -1

    # 只尝试 2, 3, 4 个说话人
    for n in [2, 3, 4]:
        if n > max_speakers:
            break
        try:
            kmeans = KMeans(n_clusters=n, random_state=42, n_init=3, max_iter=100)
            labels_sample = kmeans.fit_predict(features_sample)
            score = silhouette_score(features_sample, labels_sample, sample_size=min(200, len(features_sample)))

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