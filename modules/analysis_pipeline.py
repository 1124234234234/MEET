"""
音频分析流水线模块
支持进度回调，用于实时转写后分析等场景
"""
import os
import time


def analyze_audio(audio_path, language='zh', knowledge_items=None, score_weights=None, progress_callback=None):
    """
    对音频文件进行完整分析
    包括：音频预处理、说话人分离、关键词提取、主题分析、摘要生成、合规检查等

    Args:
        audio_path: 音频文件路径
        language: 语言
        knowledge_items: 知识库条目列表
        score_weights: 合规评分权重
        progress_callback: 进度回调函数，签名为 callback(percent, message)

    Returns:
        分析结果字典，失败返回None
    """
    def _update_progress(percent, message):
        print(f'[分析进度] {percent}% - {message}')
        if progress_callback:
            try:
                progress_callback(percent, message)
            except Exception as e:
                print(f'进度回调失败: {e}')

    try:
        from modules.audio_preprocessor import preprocess_audio, get_audio_quality_report
        from modules.speaker_diarization import speaker_diarization_simple
        from modules.text_analyzer import extract_keywords, analyze_topic, generate_summary, analyze_sentiment
        from modules.compliance_checker import calculate_compliance_score, get_score_level
        from modules.whisper_utils import transcribe_with_fix
        from app import app as flask_app

        _update_progress(3, '正在预处理音频...')
        _t0 = time.time()

        processed_path = audio_path.rsplit('.', 1)[0] + '_processed.wav'
        try:
            preprocess_audio(audio_path, processed_path)
            if os.path.exists(processed_path):
                analysis_audio_path = processed_path
            else:
                analysis_audio_path = audio_path
        except Exception as e:
            print(f'音频预处理失败: {e}')
            analysis_audio_path = audio_path

        _update_progress(5, '正在初始化模型...')

        from app import init_whisper_model, whisper_model
        init_whisper_model()

        _update_progress(10, '正在转写音频...')
        print(f'[分析流水线] 开始转写音频: {analysis_audio_path}')
        _t1 = time.time()

        if whisper_model is None:
            print('[分析流水线] Whisper模型未加载')
            _update_progress(-1, '模型加载失败')
            return None

        result = transcribe_with_fix(whisper_model, analysis_audio_path, language=language)
        full_text = result['text']
        print(f'[分析流水线] 转写完成，耗时 {time.time()-_t1:.1f}s，文本长度: {len(full_text)}')

        _update_progress(40, '正在进行说话人分离...')
        speaker_segments = []
        try:
            _t2 = time.time()
            speaker_segments = speaker_diarization_simple(analysis_audio_path)
            print(f'[分析流水线] 说话人分离完成，耗时 {time.time()-_t2:.1f}s')
        except Exception as e:
            print(f'说话人分离失败: {e}')

        transcriptions = []
        for seg in result['segments']:
            speaker = 'SPEAKER_00'
            if speaker_segments:
                for ss in speaker_segments:
                    if ss['start'] <= seg['start'] <= ss['end']:
                        speaker = ss['speaker']
                        break
            transcriptions.append({
                'speaker': speaker,
                'text': seg['text'],
                'start_time': seg['start'],
                'end_time': seg['end'],
                'confidence': seg.get('confidence', 0.0)
            })

        _update_progress(55, '正在提取关键词...')
        _t3 = time.time()
        keywords = extract_keywords(full_text, top_n=10)
        print(f'[分析流水线] 关键词提取完成，耗时 {time.time()-_t3:.1f}s')

        _update_progress(65, '正在分析主题...')
        _t4 = time.time()
        topics = analyze_topic(full_text)
        print(f'[分析流水线] 主题分析完成，耗时 {time.time()-_t4:.1f}s')

        _update_progress(75, '正在生成会议摘要...')
        _t5 = time.time()
        print(f'[分析流水线] 开始生成摘要，文本长度: {len(full_text)}')
        summary = generate_summary(full_text, max_length=300)
        print(f'[分析流水线] 摘要生成完成，耗时 {time.time()-_t5:.1f}s')

        _update_progress(80, '正在分析情绪...')
        _t6 = time.time()
        sentiment = analyze_sentiment(full_text)
        print(f'[分析流水线] 情绪分析完成，耗时 {time.time()-_t6:.1f}s')

        audio_quality = None
        try:
            audio_quality = get_audio_quality_report(audio_path, analysis_audio_path)
        except Exception as e:
            print(f'音频质量报告失败: {e}')

        compliance_result = None
        if knowledge_items:
            _update_progress(85, '正在进行合规检查...')
            _t7 = time.time()
            compliance_result = calculate_compliance_score(
                full_text,
                knowledge_items,
                score_weights=score_weights,
                transcription_segments=transcriptions
            )
            compliance_result['score_level'] = get_score_level(compliance_result['total_score'])
            print(f'[分析流水线] 合规检查完成，耗时 {time.time()-_t7:.1f}s，得分: {compliance_result["total_score"]}')

        _update_progress(100, '分析完成')
        print(f'[分析流水线] 全部分析完成，总耗时 {time.time()-_t0:.1f}s')

        return {
            'text': full_text,
            'transcriptions': transcriptions,
            'keywords': keywords,
            'topics': topics,
            'summary': summary,
            'sentiment': sentiment,
            'audio_quality': audio_quality,
            'compliance': compliance_result,
            'speaker_segments': speaker_segments
        }

    except Exception as e:
        print(f'音频分析失败: {e}')
        import traceback
        traceback.print_exc()
        if progress_callback:
            try:
                progress_callback(-1, f'分析失败: {str(e)}')
            except Exception:
                pass
        return None
