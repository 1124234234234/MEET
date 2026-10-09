"""
音频分析流水线。

把「识别 → 说话人分离 → 文本分析 → 合规比对」串成一条可复用的流程，
供上传音频分析、实时转写结束后的分析、以及第三方转写接口共用。

统一入口 analyze_audio() 的参数与返回值都与旧版本保持兼容，
但修正了两处会让功能静默失效的问题：
  1. 旧版本调用了不存在的 enable_diarization 参数（第三方接口必然 500）；
  2. 旧版本用 `'segments' in locals()` 判断，该变量从未定义，
     导致说话人分离结果永远没有写回分句，全文都是 SPEAKER_00。
"""
import os
import time

from modules import speaker_diarization


def _update(progress_callback, percent, message):
    print(f'[分析进度] {percent}% - {message}')
    if progress_callback:
        try:
            progress_callback(percent, message)
        except Exception as exc:
            print(f'进度回调失败: {exc}')


def analyze_audio(
    audio_path,
    language='zh',
    knowledge_items=None,
    score_weights=None,
    progress_callback=None,
    transcription_text=None,
    transcriptions=None,
    enable_diarization=True,
    hotwords=None,
    asr_engine_prefer=None,
    detected_language=None,
):
    """
    对音频文件执行完整分析。

    Args:
        audio_path: 音频文件路径
        language: 语言代码
        knowledge_items: 知识库条目列表（SQLAlchemy 对象）
        score_weights: 合规评分权重
        progress_callback: 进度回调 callback(percent, message)
        transcription_text: 已有转写文本；提供时跳过重新识别（实时转写复用）
        transcriptions: 已有分句列表；提供时复用并补齐说话人
        enable_diarization: 是否执行说话人分离
        hotwords: 热词表（行业词汇），用于提升专有名词识别率
        asr_engine_prefer: 'auto' / 'funasr' / 'whisper'

    Returns:
        分析结果字典；失败返回 None。
    """
    try:
        from modules.audio_preprocessor import preprocess_audio, get_audio_quality_report
        from modules.text_analyzer import (
            analyze_sentiment,
            analyze_topic,
            extract_action_items,
            extract_decisions,
            extract_keywords,
            generate_summary,
        )
        from modules.compliance_checker import calculate_compliance_score, get_score_level

        started = time.time()
        segments = None

        # ---------- 1. 识别 ----------
        if transcription_text:
            _update(progress_callback, 10, '使用已有转写文本...')
            full_text = transcription_text
            segments = [dict(seg) for seg in (transcriptions or [])]
            if not segments:
                segments = [{
                    'speaker': 'SPEAKER_00',
                    'text': full_text,
                    'start_time': 0,
                    'end_time': 0,
                    'confidence': 1.0,
                }]
            analysis_audio_path = audio_path
            print(f'[分析流水线] 复用已有转写，文本长度 {len(full_text)}')
        else:
            from modules import asr_engine

            _update(progress_callback, 10, '正在识别语音...')
            _update(progress_callback, 25, '正在分段与识别...')
            # 热词增强：知识库关键词 + 配置默认行业词汇（需求：行业词汇库热词解码）
            if hotwords is None:
                hotwords = asr_engine.build_hotwords(knowledge_items)
            result = asr_engine.transcribe(
                audio_path,
                language=language,
                hotwords=hotwords,
                prefer=asr_engine_prefer or 'auto',
            )
            full_text = result['text']
            segments = [{
                'speaker': 'SPEAKER_00',
                'text': seg['text'],
                'start_time': seg['start'],
                'end_time': seg['end'],
                'confidence': seg.get('confidence', 1.0),
            } for seg in result['segments']]
            detected_language = result.get('language') or detected_language or language
            print(f'[分析流水线] 识别完成，引擎={result.get("engine")}，'
                  f'耗时 {time.time()-started:.1f}s，{len(segments)} 句')

            # 非实时文本路径才需要预处理音频供后续质量报告对比
            analysis_audio_path = audio_path

        if not full_text or not full_text.strip():
            _update(progress_callback, -1, '未识别到有效语音内容')
            return None

        # ---------- 2. 说话人分离 ----------
        speaker_segments = []
        if enable_diarization and segments:
            _update(progress_callback, 40, '正在进行说话人分离...')
            t0 = time.time()
            try:
                speaker_segments = speaker_diarization.speaker_diarization_simple(audio_path)
                if speaker_segments:
                    segments = speaker_diarization.assign_speakers_to_segments(
                        segments, speaker_segments
                    )
                    speaker_count = len({s['speaker'] for s in segments})
                    print(f'[分析流水线] 说话人分离完成，{speaker_count} 位说话人，'
                          f'耗时 {time.time()-t0:.1f}s')
                else:
                    print('[分析流水线] 说话人分离无结果，按单一说话人处理')
            except Exception as exc:
                print(f'[分析流水线] 说话人分离失败: {exc}')

        # ---------- 3. 文本分析 ----------
        _update(progress_callback, 55, '正在提取关键词...')
        keywords = extract_keywords(full_text, top_n=10, language=language)

        _update(progress_callback, 65, '正在分析主题...')
        topics = analyze_topic(full_text, language=language)

        _update(progress_callback, 75, '正在生成会议摘要...')
        summary = generate_summary(full_text, max_length=300, language=language)

        _update(progress_callback, 80, '正在提取待办事项与决议...')
        action_items = extract_action_items(full_text, language=language)
        decisions = extract_decisions(full_text, language=language)

        sentiment = analyze_sentiment(full_text, language=language)

        # ---------- 4. 音频质量报告 ----------
        audio_quality = None
        processed_path = audio_path.rsplit('.', 1)[0] + '_processed.wav'
        try:
            if os.path.exists(processed_path) and processed_path != audio_path:
                audio_quality = get_audio_quality_report(audio_path, processed_path)
        except Exception as exc:
            print(f'音频质量报告失败: {exc}')

        # ---------- 5. 合规比对 ----------
        compliance_result = None
        if knowledge_items:
            _update(progress_callback, 88, '正在进行合规检查...')
            t0 = time.time()
            compliance_result = calculate_compliance_score(
                full_text,
                knowledge_items,
                score_weights=score_weights,
                transcription_segments=segments,
            )
            compliance_result['score_level'] = get_score_level(compliance_result['total_score'])
            print(f'[分析流水线] 合规检查完成，耗时 {time.time()-t0:.1f}s，'
                  f'得分 {compliance_result["total_score"]}')

        _update(progress_callback, 100, '分析完成')
        print(f'[分析流水线] 全部完成，总耗时 {time.time()-started:.1f}s')

        return {
            'text': full_text,
            'transcriptions': segments,
            'keywords': keywords,
            'topics': topics,
            'summary': summary,
            'sentiment': sentiment,
            'action_items': action_items,
            'decisions': decisions,
            'audio_quality': audio_quality,
            'compliance_report': compliance_result,
            'compliance': compliance_result,
            'speaker_segments': speaker_segments,
            'duration': round(max((s.get('end_time', 0) or 0) for s in segments), 2) if segments else 0,
            'language': detected_language or language,
        }

    except Exception as exc:
        print(f'音频分析失败: {exc}')
        import traceback
        traceback.print_exc()
        _update(progress_callback, -1, f'分析失败: {exc}')
        return None


def analyze_recorded_audio(audio_path, language='zh', knowledge_items=None,
                           score_weights=None, progress_callback=None,
                           enable_diarization=True, hotwords=None):
    """对已录制音频做完整分析（实时转写停止后调用）。"""
    return analyze_audio(
        audio_path,
        language=language,
        knowledge_items=knowledge_items,
        score_weights=score_weights,
        progress_callback=progress_callback,
        enable_diarization=enable_diarization,
        hotwords=hotwords,
    )
