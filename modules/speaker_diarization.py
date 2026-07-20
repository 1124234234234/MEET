import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def speaker_diarization_simple(audio_path, num_speakers=None, timeout=60):
    """
    说话人分离：支持多人交替发言和同时发言场景
    优先级：pyannote.audio > 增强聚类算法（基于MFCC+谱聚类）
    添加超时机制，避免分析过长时间卡住
    num_speakers=None 时自动估计说话人数量
    """
    import threading

    result_container = []

    def run_diarization():
        # 方案1：pyannote.audio（需要HF_TOKEN，效果最好）
        try:
            from pyannote.audio import Pipeline
            import torch

            hf_token = os.environ.get('HF_TOKEN', '')

            if hf_token:
                print("[说话人分离] 使用HF_TOKEN加载pyannote模型...")
                pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=hf_token
                )
            else:
                print("[说话人分离] 未配置HF_TOKEN，尝试无token加载pyannote模型...")
                try:
                    pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1"
                    )
                except Exception as e_no_token:
                    print(f"[说话人分离] 无token加载失败（pyannote为受限模型，需配置HF_TOKEN）: {e_no_token}")
                    print("[说话人分离] 降级到增强聚类算法")
                    result_container.append(enhanced_diarization(audio_path, num_speakers))
                    return

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
            print(f"[说话人分离] pyannote加载失败: {e}")
            print("[说话人分离] 降级到增强聚类算法")
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

    # 2. 组合所有特征 - 加重MFCC和频谱特征权重（更突出音色差异）
    features = np.vstack([
        mfcc,
        mfcc_delta * 0.8,
        mfcc_delta2 * 0.5,
        chroma * 0.2,
        spectral_centroid * 0.3,
        spectral_bandwidth * 0.3,
        zcr * 0.1,
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

    # 6. 使用谱聚类 + GMM的组合方案（谱聚类更适合说话人音色聚类）
    try:
        # 先用谱聚类（对非球形簇效果更好）
        from sklearn.cluster import SpectralClustering
        clustering = SpectralClustering(
            n_clusters=estimated_n,
            random_state=42,
            affinity='nearest_neighbors',
            n_neighbors=min(30, len(voice_features) // 10),
            assign_labels='kmeans'
        )
        labels_voice = clustering.fit_predict(voice_features)
        
        print(f"  [diarization] spectral clustering done: {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"  [diarization] spectral clustering failed: {e}, using agglomerative")
        try:
            # 备用：GMM初始化 + 层次聚类
            gmm = GaussianMixture(
                n_components=estimated_n,
                random_state=42,
                n_init=3,
                max_iter=100
            )
            gmm_labels = gmm.fit_predict(voice_features)
            
            clustering = AgglomerativeClustering(
                n_clusters=estimated_n,
                linkage='ward',
                metric='euclidean'
            )
            labels_voice = clustering.fit_predict(voice_features)
        except Exception as e2:
            print(f"  [diarization] all clustering failed: {e2}, using basic KMeans")
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=estimated_n, random_state=42, n_init=5)
            labels_voice = kmeans.fit_predict(voice_features)

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
    使用轮廓系数 + 肘部法则综合判断
    对中文会议场景做优化：至少2人，优先尝试2-4人
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
    
    # 策略1：轮廓系数
    sil_scores = {}
    for n in range(2, min(max_speakers, 5) + 1):
        try:
            kmeans = KMeans(n_clusters=n, random_state=42, n_init=5, max_iter=100)
            labels = kmeans.fit_predict(features_sample)
            
            if len(set(labels)) > 1:
                sil_score = silhouette_score(features_sample, labels, sample_size=min(500, len(features_sample)))
            else:
                sil_score = -1
            
            sil_scores[n] = sil_score
            print(f"    [estimate] n={n}, silhouette={sil_score:.4f}")
        except Exception as e:
            print(f"    [estimate] n={n} failed: {e}")
            sil_scores[n] = -1
    
    # 策略2：GMM BIC（贝叶斯信息准则，越小越好）
    bic_scores = {}
    for n in range(2, min(max_speakers, 5) + 1):
        try:
            gmm = GaussianMixture(n_components=n, random_state=42, n_init=3, max_iter=100)
            gmm.fit(features_sample)
            bic_scores[n] = gmm.bic(features_sample)
            print(f"    [estimate] n={n}, BIC={bic_scores[n]:.1f}")
        except Exception as e:
            bic_scores[n] = float('inf')
    
    # 综合判断：
    # 1. 如果轮廓系数n=3和n=2差距不大（<0.02），偏向n=3（会议场景更常见）
    # 2. 如果BIC显示n=3明显更小，选n=3
    best_n = 2
    
    sil_n2 = sil_scores.get(2, -1)
    sil_n3 = sil_scores.get(3, -1)
    sil_n4 = sil_scores.get(4, -1)
    
    bic_n2 = bic_scores.get(2, float('inf'))
    bic_n3 = bic_scores.get(3, float('inf'))
    bic_n4 = bic_scores.get(4, float('inf'))
    
    # 找BIC最小的（注意BIC越小越好）
    min_bic_n = min(bic_scores, key=bic_scores.get) if bic_scores else 2
    
    # 决策逻辑：
    # - 会议场景通常2-4人
    # - 轮廓系数差距在0.02以内时，选择BIC更优的
    # - 如果都差不多，偏向3人（更符合会议场景）
    if sil_n3 >= sil_n2 - 0.02 and bic_n3 < bic_n2:
        best_n = 3
    elif sil_n2 > sil_n3 and sil_n2 - sil_n3 > 0.03:
        best_n = 2
    elif sil_n3 >= 0:
        best_n = 3
    
    # 如果n=4的BIC明显更好，且轮廓系数不低
    if best_n == 3 and sil_n4 >= sil_n3 - 0.02 and bic_n4 < bic_n3 * 0.98:
        best_n = 4
    
    print(f"    [estimate] best_n={best_n} (sil={sil_scores.get(best_n, 0):.4f}, BIC={bic_scores.get(best_n, 0):.1f})")
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