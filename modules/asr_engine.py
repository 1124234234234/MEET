"""
统一语音识别引擎。

主引擎：FunASR paraformer-large + fsmn-vad + ct-punc（本地模型）
  - 中文识别准确率优于 Whisper，且自带标点恢复模型（不再用正则猜标点）
  - CPU 上实时率约 0.09，比 Whisper medium 快一个数量级
  - 支持热词增强（hotword），对应需求中的「热词增强的语音识别解码网络」

备选引擎：Whisper（本地缓存模型），在 FunASR 不可用时自动接管。

两个引擎都返回统一结构：
    {
        'text': 全文,
        'segments': [{'start': 秒, 'end': 秒, 'text': 分句文本, 'confidence': float}],
        'language': 语言代码,
        'engine': 引擎名称,
    }
分句粒度直接来自 VAD 语音段，因此每个片段都带精确时间戳，可直接用于
"谁在什么时候说了什么" 以及后续按时间定位风险内容。
"""
import os

import numpy as np

from modules import model_registry
from modules.audio_io import DEFAULT_SR, load_audio

# 单个语音片段过短会导致识别不稳定，过短片段并入相邻片段
MIN_SEGMENT_SECONDS = 0.25
# 长语音片段上限，超过则强制切分，避免单段过长影响识别与时间定位
MAX_SEGMENT_SECONDS = 15.0
# 伪语音片段判据：相对能量低于本段录音语音中位数的该比例，且时长不超过阈值时丢弃
NOISE_REL_RMS_RATIO = 0.45
NOISE_MAX_SECONDS = 1.5
# 分片之间的最小静音间隔（秒），用于把超长语音段切开时的估算
DEFAULT_HOTWORDS = (
    '风险等级 投资限制 业绩提醒义务 风险告知书 合规 投资者适当性 '
    '录音录像 持证上岗 保本保收益 零风险 稳赚不赔'
)


def _normalize_hotwords(hotwords):
    if not hotwords:
        return None
    if isinstance(hotwords, (list, tuple, set)):
        words = [str(w).strip() for w in hotwords if str(w).strip()]
    else:
        words = [str(hotwords).strip()]
    return ' '.join(words) if words else None


def _split_long_regions(regions, max_seconds=MAX_SEGMENT_SECONDS):
    """把超过 max_seconds 的语音段按时长均分，保证每段都有合理时间范围。"""
    result = []
    for start, end in regions:
        duration = end - start
        if duration <= max_seconds:
            result.append((start, end))
            continue
        parts = int(duration // max_seconds) + 1
        step = duration / parts
        for i in range(parts):
            result.append((round(start + i * step, 3), round(start + (i + 1) * step, 3)))
    return result


def _merge_short_regions(regions, min_seconds=MIN_SEGMENT_SECONDS):
    """
    吸收掉时长过短的语音段（多为语气词或瞬时噪声），避免切出无意义的碎片。

    注意：只按时长判断，不能按「与上一段的间隔」判断 —— 相邻语音段之间
    通常只有几百毫秒停顿，那是正常的分句边界，合并会把整场会议并成一段。
    """
    if not regions:
        return []
    merged = [list(regions[0])]
    for start, end in regions[1:]:
        if end - start < min_seconds:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    # 首段过短时并入后一段，避免开头碎片单独成句
    if len(merged) > 1 and merged[0][1] - merged[0][0] < min_seconds:
        merged[1][0] = merged[0][0]
        merged.pop(0)
    return [(round(s, 3), round(e, 3)) for s, e in merged]


def _energy_vad(y, sr, frame_ms=30, min_speech=0.3):
    """兜底能量 VAD（FunASR VAD 不可用时使用）。"""
    hop = max(1, int(sr * frame_ms / 1000))
    if len(y) < hop:
        return []
    n_frames = len(y) // hop
    frames = y[:n_frames * hop].reshape(n_frames, hop)
    energy = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    if not np.any(energy > 0):
        return []
    threshold = max(np.percentile(energy, 25), energy.max() * 0.06)

    regions = []
    in_speech = False
    start = 0.0
    for i, value in enumerate(energy):
        t = i * hop / sr
        if value > threshold and not in_speech:
            in_speech = True
            start = t
        elif value <= threshold and in_speech:
            in_speech = False
            if t - start >= min_speech:
                regions.append((start, t))
    if in_speech and len(y) / sr - start >= min_speech:
        regions.append((start, len(y) / sr))
    return [(round(s, 3), round(e, 3)) for s, e in regions]


def detect_speech_regions(audio_input, sr=DEFAULT_SR):
    """
    返回语音片段列表 [(start_sec, end_sec), ...]。

    优先使用本地 FunASR VAD（细粒度参数），否则退回能量 VAD。
    audio_input 可以是文件路径或 numpy 音频数组。
    """
    if isinstance(audio_input, np.ndarray):
        y = np.asarray(audio_input, dtype=np.float32)
        duration = len(y) / sr
        source = y
    else:
        y = load_audio(audio_input, sr=sr)
        duration = len(y) / sr
        source = audio_input

    regions = []
    vad = model_registry.get_vad_model()
    if vad is not None:
        try:
            result = vad.generate(input=source, cache={})
            if result and result[0].get('value'):
                regions = [(ms[0] / 1000.0, ms[1] / 1000.0) for ms in result[0]['value']]
        except Exception as exc:
            print(f'[ASR] FunASR VAD 失败，回退能量 VAD: {exc}')
            regions = []

    if not regions:
        regions = _energy_vad(y, sr)

    regions = _merge_short_regions(_split_long_regions(regions))
    if not regions and duration > 0:
        # 完全没有检测到语音时，整段作为一个片段，保证不会丢内容
        regions = [(0.0, round(duration, 3))]
    return regions


def build_hotwords(knowledge_items=None, extra=None):
    """
    组装热词表（行业词汇增强）。

    来源三部分，去重后合并：
      1. 配置项 Config.ASR_HOTWORDS（默认的合规/金融行业词汇，可用环境变量覆盖）
      2. 知识库里已有条目的关键词（用户维护的「行业词汇库」）
      3. 调用方额外指定的词（例如实时会话传入的 hotwords）

    对应需求「利用行业词汇库建立热词增强的语音识别解码网络」。
    """
    words = []
    seen = set()

    def add(value):
        if not value:
            return
        items = value if isinstance(value, (list, tuple, set)) else str(value).split()
        for item in items:
            word = str(item).strip()
            if word and word not in seen:
                seen.add(word)
                words.append(word)

    try:
        from config import Config

        add(Config.ASR_HOTWORDS)
    except Exception:
        pass

    for item in knowledge_items or []:
        raw = getattr(item, 'keywords', None)
        if raw is None:
            continue
        try:
            import json as _json

            parsed = _json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(parsed, list):
                for kw in parsed:
                    word = str(kw).strip()
                    # 过短或过长的关键词不适合做热词
                    if 2 <= len(word) <= 12:
                        add(word)
        except Exception:
            continue

    add(extra)
    return words or None


def _drop_noise_regions(audio, sr, regions):
    """
    丢掉「能量极低且极短」的伪语音片段。

    VAD 偶尔会把呼吸声、碰麦声、电流噪声当成语音，交给识别模型后会产生
    幻觉文本（实测出现过「有二十。」「I.」这类片段）。这类片段有稳定的特征：
    能量显著低于本段录音的语音水平，而且时长短。

    实测（4 个样本）：真实语音片段的相对能量在 0.60~1.89 之间，
    而被误判的那个伪语音片段是 0.41 且只有 0.97 秒。
    因此用「相对能量 < 0.45 且时长 < 1.5s」作判据；片段太少时不做判断
    （中位数不可靠）。
    """
    if len(regions) < 4:
        return regions, []

    rms = []
    for start, end in regions:
        seg = audio[int(start * sr):int(end * sr)]
        rms.append(float(np.sqrt((seg ** 2).mean())) if len(seg) else 0.0)
    median = float(np.median(rms))
    if median <= 1e-8:
        return regions, []

    kept, dropped = [], []
    for (start, end), energy in zip(regions, rms):
        duration = end - start
        if energy < median * NOISE_REL_RMS_RATIO and duration < NOISE_MAX_SECONDS:
            dropped.append((start, end, energy, duration))
        else:
            kept.append((start, end))
    return kept, dropped


class FunASREngine:
    """基于 FunASR 的识别引擎（推荐）。"""

    name = 'funasr'

    def available(self):
        return model_registry.has_local_funasr()

    def transcribe_array(self, audio, sr=DEFAULT_SR, language='zh', hotwords=None, regions=None):
        """识别 numpy 音频，按语音片段切分返回分句结果。"""
        pipeline = model_registry.get_funasr_pipeline()
        if pipeline is None:
            raise RuntimeError('FunASR 模型不可用')

        if sr != DEFAULT_SR:
            from modules.audio_io import resample
            audio = resample(audio, sr, DEFAULT_SR)
            sr = DEFAULT_SR

        audio = np.asarray(audio, dtype=np.float32)
        if audio.size == 0:
            return {'text': '', 'segments': [], 'language': language, 'engine': self.name}

        if regions is None:
            regions = detect_speech_regions(audio, sr=sr)

        regions, dropped = _drop_noise_regions(audio, sr, regions)
        if dropped:
            for start, end, energy, duration in dropped:
                print(f'[ASR] 丢弃疑似噪声片段 [{start:.2f}-{end:.2f}] '
                      f'时长 {duration:.2f}s 相对能量 {energy:.4f}')

        chunks = []
        spans = []
        for start, end in regions:
            seg = audio[int(start * sr):int(end * sr)]
            if len(seg) < int(0.05 * sr):
                continue
            chunks.append(seg)
            spans.append((start, end))

        if not chunks:
            return {'text': '', 'segments': [], 'language': language, 'engine': self.name}

        generate_kwargs = {
            'cache': {},
            'language': 'zh' if language == 'zh' else language,
            'use_itn': True,
            'batch_size_s': 60,
        }
        # 热词增强：调用方没给就用配置里的默认行业词汇（见 build_hotwords），
        # 这样各条链路都不会漏掉行业词加权
        hotword = _normalize_hotwords(hotwords) or _normalize_hotwords(build_hotwords())
        if hotword:
            generate_kwargs['hotword'] = hotword

        results = pipeline.generate(input=chunks, **generate_kwargs)

        segments = []
        texts = []
        for span, item in zip(spans, results):
            text = (item.get('text') or '').strip()
            if not text:
                continue
            segments.append({
                'start': round(float(span[0]), 2),
                'end': round(float(span[1]), 2),
                'text': text,
                'confidence': 1.0,
            })
            texts.append(text)

        return {
            'text': ''.join(texts),
            'segments': segments,
            'language': language,
            'engine': self.name,
        }


class WhisperEngine:
    """基于 Whisper 的识别引擎（备选）。"""

    name = 'whisper'

    def __init__(self, model_name=None):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        import whisper

        from config import Config

        name = self.model_name or Config.WHISPER_MODEL
        try:
            self._model = whisper.load_model(name)
        except Exception as exc:
            print(f'[ASR] Whisper {name} 加载失败({exc})，改用 small')
            self._model = whisper.load_model('small')
        return self._model

    def available(self):
        try:
            import whisper  # noqa: F401
            return True
        except Exception:
            return False

    def transcribe_array(self, audio, sr=DEFAULT_SR, language='zh', hotwords=None, regions=None):
        from modules.whisper_utils import fix_traditional_chinese

        model = self._load()
        audio = np.asarray(audio, dtype=np.float32)
        if sr != DEFAULT_SR:
            from modules.audio_io import resample
            audio = resample(audio, sr, DEFAULT_SR)

        result = model.transcribe(audio, language=language, condition_on_previous_text=False)
        segments = []
        for seg in result.get('segments', []):
            segments.append({
                'start': round(float(seg.get('start', 0)), 2),
                'end': round(float(seg.get('end', 0)), 2),
                'text': fix_traditional_chinese((seg.get('text') or '').strip()),
                'confidence': float(seg.get('confidence', 0) or 0),
            })
        return {
            'text': fix_traditional_chinese((result.get('text') or '').strip()),
            'segments': segments,
            'language': result.get('language', language),
            'engine': self.name,
        }


class RemoteVoiceEngine:
    """
    远端 voice-api 引擎（边缘分布式部署模式）。

    当配置了 VOICE_API_URL 时，识别请求转发到远端服务，本机不加载识别模型，
    适合算力有限的会议室端设备。接口保持与其它引擎一致。
    """

    name = 'remote'

    def __init__(self, base_url=None, timeout=1800):
        from config import Config

        self.base_url = (base_url or Config.VOICE_API_URL or '').rstrip('/')
        self.timeout = timeout or Config.VOICE_API_TIMEOUT
        self._client = None

    def available(self):
        return bool(self.base_url)

    def _get_client(self):
        if self._client is None:
            from voice_api_client import VoiceAPIWhisperClient

            self._client = VoiceAPIWhisperClient(self.base_url, timeout=self.timeout)
            self._client.check_health()
        return self._client

    def transcribe_array(self, audio, sr=DEFAULT_SR, language='zh', hotwords=None, regions=None):
        import os
        import tempfile

        from modules.audio_io import save_wav

        audio = np.asarray(audio, dtype=np.float32)
        if audio.size == 0:
            return {'text': '', 'segments': [], 'language': language, 'engine': self.name}

        handle, temp_path = tempfile.mkstemp(suffix='.wav')
        os.close(handle)
        try:
            save_wav(temp_path, audio, sr)
            payload = self._get_client().transcribe(temp_path, language=language)
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

        text = (payload.get('text') or '').strip()
        segments = []
        if text:
            segments.append({
                'start': 0.0,
                'end': round(len(audio) / sr, 2),
                'text': text,
                'confidence': 1.0,
            })
        return {
            'text': text,
            'segments': segments,
            'language': payload.get('language') or language,
            'engine': self.name,
        }


_engines = {}


def get_engine(prefer='auto'):
    """
    获取识别引擎实例（按 prefer 缓存单例）。

    优先级：远端 voice-api（若已配置）> FunASR > Whisper。
    prefer='funasr' / 'whisper' 可强制指定，'auto' 走上述优先级。
    """
    prefer = (prefer or 'auto').lower()
    if prefer in _engines:
        return _engines[prefer]

    engine = None

    if prefer == 'auto':
        remote = RemoteVoiceEngine()
        if remote.available():
            print(f'[ASR] 使用远端 voice-api: {remote.base_url}')
            engine = remote

    if engine is None and prefer in ('auto', 'funasr'):
        candidate = FunASREngine()
        if candidate.available():
            engine = candidate
        elif prefer == 'funasr':
            raise RuntimeError('FunASR 本地模型不可用，请检查 models/funasr 目录')

    if engine is None:
        candidate = WhisperEngine()
        if candidate.available():
            engine = candidate
            if prefer == 'auto':
                print('[ASR] FunASR 不可用，使用 Whisper 引擎')

    if engine is None:
        raise RuntimeError('没有可用的语音识别引擎')

    _engines[prefer] = engine
    return engine


def transcribe(audio_path, language='zh', hotwords=None, prefer='auto', regions=None):
    """对音频文件执行识别，返回统一结构的结果。"""
    audio = load_audio(audio_path, sr=DEFAULT_SR)
    engine = get_engine(prefer)
    result = engine.transcribe_array(audio, sr=DEFAULT_SR, language=language,
                                     hotwords=hotwords, regions=regions)
    result['duration'] = round(len(audio) / DEFAULT_SR, 2)
    return result
