"""
Whisper 转写工具函数。

统一处理繁简转换：
Whisper 中文转写常夹杂繁体字，而项目其它环节（关键词、合规比对、摘要）
都按简体处理，所以识别结果统一走一遍 opencc 的 t2s 转换。
FunASR 引擎不会输出繁体，本函数对它等效于空操作，保留是为了让两条
识别链路（FunASR / Whisper）的收尾逻辑保持一致。
"""

try:
    from opencc import OpenCC

    _cc = OpenCC('t2s')
    OPENCC_AVAILABLE = True
except ImportError:
    OPENCC_AVAILABLE = False


def fix_traditional_chinese(text):
    """将繁体中文转换为简体中文；opencc 不可用时原样返回。"""
    if not text:
        return text
    if OPENCC_AVAILABLE:
        return _cc.convert(text)
    return text


def transcribe_with_fix(model, audio_path, language='zh', **kwargs):
    """
    使用传入的模型转写音频，并对结果做繁简后处理。

    参数:
        model: 兼容 whisper 接口的模型对象（含 transcribe 方法）
        audio_path: 音频文件路径
        language: 语言代码
        **kwargs: 透传给 model.transcribe 的其它参数

    返回:
        转写结果字典（text 与各 segment 的 text 已转简体）
    """
    transcribe_kwargs = {
        'language': language,
        # 关闭上文条件化，避免长会议中错误文本被反复放大
        'condition_on_previous_text': False,
    }
    transcribe_kwargs.update(kwargs)

    result = model.transcribe(audio_path, **transcribe_kwargs)

    if 'text' in result:
        result['text'] = fix_traditional_chinese(result['text'])

    for seg in result.get('segments', []) or []:
        if 'text' in seg:
            seg['text'] = fix_traditional_chinese(seg['text'])

    return result
