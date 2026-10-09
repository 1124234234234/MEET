"""
语音链路量化评测（对应个人项目「研究内容四：系统性能评估」）。

覆盖指标：
  - 转写准确率：字错误率 CER（需求：WER < 10%）
  - 说话人日志：说话人混淆率（需求：< 5%）、说话人数目是否正确
  - 说话人归属：每句转写是否归属到正确的说话人
  - 合规比对：必传要点遗漏检测、风险重复计数、否定语境、异常知识库数据
  - 文本结构化：待办事项 / 决议结论提取、摘要与情绪分析

运行：
    python tests/test_voice_pipeline.py            # 全部用例
    python tests/test_voice_pipeline.py --quick     # 跳过需要模型的重型用例

先运行 tests/make_ground_truth_audio.py 生成带标注的测试音频。
"""
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_DIR = os.path.join(BASE_DIR, 'tests', 'test_audio_files')

SCENARIOS = ['gt_meeting', 'gt_2spk']

# 需求指标阈值
CER_TARGET = 0.10          # 字错误率 < 10%
CONFUSION_TARGET = 0.05    # 说话人混淆率 < 5%

RESULTS = []


def record(name, passed, detail=''):
    RESULTS.append((name, passed, detail))
    mark = 'PASS' if passed else 'FAIL'
    print(f'  [{mark}] {name}' + (f' — {detail}' if detail else ''))
    return passed


def strip_punct(text):
    """去掉标点与空白，只保留用于比对的内容字符。"""
    import re
    return re.sub(r'[\s，。！？、；：“”‘’（）《》【】,.!?;:"\'()\[\]<>—…·-]', '', text or '')


def edit_distance(a, b):
    """Levenshtein 编辑距离（滚动数组，按字符）。"""
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
            current.append(min(
                previous[j] + 1,        # 删除
                current[j - 1] + 1,     # 插入
                previous[j - 1] + (ca != cb),  # 替换
            ))
        previous = current
    return previous[-1]


def character_error_rate(reference, hypothesis):
    """字错误率 = 编辑距离 / 参考文本长度。"""
    ref = strip_punct(reference)
    hyp = strip_punct(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def load_ground_truth(scenario):
    path = os.path.join(AUDIO_DIR, f'{scenario}.json')
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def ground_truth_turns(gt):
    return [{'start': t['start'], 'end': t['end'], 'speaker': t['speaker']} for t in gt['turns']]


# ---------------------------------------------------------------- 语音链路

def test_transcription_accuracy():
    """
    转写字错误率（需求 WER < 10%）。

    这里测的是**用户实际走的链路**：自适应预处理 → 识别（带热词）。
    旧版本直接测原始音频，会掩盖「预处理把准确率拉低」的问题——
    实测原链路在 2 人对话上 CER 高达 23%，而原始音频只有 4%。
    """
    print('\n[1] 转写准确率（字错误率 CER，上传链路）')
    import tempfile

    from modules import asr_engine
    from modules.audio_preprocessor import preprocess_audio

    for scenario in SCENARIOS:
        gt = load_ground_truth(scenario)
        if not gt:
            record(f'{scenario} 转写准确率', False, '缺少标注文件，请先运行 make_ground_truth_audio.py')
            continue

        wav = os.path.join(AUDIO_DIR, f'{scenario}.wav')
        handle, processed = tempfile.mkstemp(suffix='_processed.wav')
        os.close(handle)

        started = time.time()
        try:
            preprocess_audio(wav, processed)
            result = asr_engine.transcribe(processed, language='zh')
        finally:
            if os.path.exists(processed):
                os.remove(processed)
        elapsed = time.time() - started

        cer = character_error_rate(gt['full_text'], result['text'])
        duration = result.get('duration') or gt['duration']
        rtf = elapsed / duration if duration else 0
        record(
            f'{scenario} 字错误率 CER={cer:.2%} (<{CER_TARGET:.0%}, RTF={rtf:.3f}, 引擎={result["engine"]})',
            cer < CER_TARGET,
        )


def test_diarization_accuracy():
    """说话人混淆率与人数估计（需求混淆率 < 5%）。"""
    print('\n[2] 说话人分离准确率')
    from modules import speaker_diarization

    for scenario in SCENARIOS:
        gt = load_ground_truth(scenario)
        if not gt:
            record(f'{scenario} 说话人分离', False, '缺少标注文件')
            continue

        wav = os.path.join(AUDIO_DIR, f'{scenario}.wav')
        hypothesis = speaker_diarization.diarize(wav)
        metrics = speaker_diarization.evaluate(ground_truth_turns(gt), hypothesis)

        record(
            f'{scenario} 混淆率={metrics["confusion_rate"]:.2%} (<{CONFUSION_TARGET:.0%})，'
            f'人数 {metrics["hyp_speakers"]}/{metrics["ref_speakers"]}',
            metrics['confusion_rate'] < CONFUSION_TARGET and metrics['speaker_count_error'] == 0,
        )


def test_speaker_attribution():
    """「谁说了什么」——按句归属说话人的正确率。"""
    print('\n[3] 说话人归属（谁说了什么）')
    from modules import asr_engine, speaker_diarization

    gt = load_ground_truth('gt_meeting')
    if not gt:
        record('说话人归属', False, '缺少标注文件')
        return

    wav = os.path.join(AUDIO_DIR, 'gt_meeting.wav')
    asr = asr_engine.transcribe(wav, language='zh')
    diar = speaker_diarization.diarize(wav)

    segments = [{
        'speaker': 'SPEAKER_00',
        'text': s['text'],
        'start_time': s['start'],
        'end_time': s['end'],
    } for s in asr['segments']]
    labelled = speaker_diarization.assign_speakers_to_segments(segments, diar)

    # 用真实分句中心时间落在哪个真实说话人区间来判断归属是否正确
    correct = 0
    checked = 0
    for segment in labelled:
        center = (segment['start_time'] + segment['end_time']) / 2
        truth = None
        for turn in gt['turns']:
            if turn['start'] <= center <= turn['end']:
                truth = turn['speaker']
                break
        if truth is None:
            continue
        checked += 1
        if segment['speaker'] == truth:
            correct += 1

    if checked == 0:
        record('说话人归属', False, '没有可比对的分句')
        return

    accuracy = correct / checked
    record(f'说话人归属正确率 {accuracy:.1%}（{correct}/{checked} 句）', accuracy >= 0.9)


# ---------------------------------------------------------------- 合规比对

class _FakeItem:
    """模拟知识库 ORM 对象（属性访问 + JSON 字符串字段）。"""

    def __init__(self, title, content, item_type, keywords=None, required_points=None, status='active'):
        self.title = title
        self.content = content
        self.item_type = item_type
        self.keywords = json.dumps(keywords or [], ensure_ascii=False)
        self.required_points = json.dumps(required_points or [], ensure_ascii=False)
        self.status = status


def test_point_coverage():
    """必传要点遗漏检测：只覆盖部分要点时必须报出遗漏。"""
    print('\n[4] 必传要点覆盖与遗漏检测')
    from modules.compliance_checker import calculate_compliance_score

    item = _FakeItem(
        '理财销售必传要点',
        '理财产品销售必须覆盖的合规要点。',
        'key_points',
        keywords=['风险测评', '适当性', '风险告知'],
        required_points=['投资者风险测评', '风险等级匹配', '风险告知义务'],
    )
    # 文本只明确提到了「风险告知义务」这一条要点
    text = '销售过程中必须履行风险告知义务，向客户说明产品风险。'
    result = calculate_compliance_score(
        text, [item], transcription_segments=[
            {'speaker': 'SPEAKER_00', 'text': text, 'start_time': 0, 'end_time': 5}
        ]
    )

    covered = result['covered_points']
    missing = result['missing_points']
    record(
        f'覆盖 {len(covered)} 条 / 遗漏 {len(missing)} 条（期望覆盖1条、遗漏2条）',
        len(covered) == 1 and len(missing) == 2,
        f'covered={covered} missing={missing}',
    )


def test_point_coverage_offtopic():
    """
    与模板无关的会议不应被套用模板、也不应虚报"遗漏全部要点"。

    回归：知识库里只有「项目例会要点」时，一通保险销售电话被判成
    「不合格 + 遗漏进度汇报/问题讨论/下周计划/风险说明」。
    现在按「转写文本 ↔ 模板」的语义相关性门控（阈值 0.50），
    不相关就明确说明「未匹配到适用的模板」并提示补充模板。
    """
    print('\n[4b] 无关会议不应虚报遗漏要点')
    from modules.compliance_checker import calculate_compliance_score

    policy = _FakeItem(
        '理财产品销售合规管理办法',
        '理财产品销售必须遵守合规要求，包括投资者适当性管理、风险测评、风险告知，销售过程需录音录像。',
        'policy',
        keywords=['风险', '销售', '投资者', '合规'],
    )
    points = _FakeItem(
        '理财销售必传要点',
        '理财产品销售必须覆盖的合规要点。',
        'key_points',
        keywords=['风险测评', '适当性', '风险告知'],
        required_points=['投资者风险测评', '风险等级匹配', '风险告知义务'],
    )

    # 与合规完全无关的会议内容
    text = '今天讨论了办公室装修方案，墙面颜色选浅灰，家具下周采购。'
    result = calculate_compliance_score(text, [policy, points])

    record(f'无关会议不虚报遗漏（得到 {len(result["missing_points"])} 条，期望 0）',
           len(result['missing_points']) == 0, str(result['missing_points']))
    record(f'模板被正确判定为不适用（rejected={result.get("point_template_rejected")}）',
           result.get('point_template_rejected') is True)
    record('建议中说明了未套用模板并提示补充',
           any('未套用' in s or '模板' in s for s in result.get('suggestions', [])),
           str(result.get('suggestions', [])[:1]))

    # 对照：内容与模板相关时仍要正常核查（不能因为门控而漏检）
    related = '销售过程中必须履行风险告知义务，向客户说明产品风险，并进行投资者风险测评。'
    result2 = calculate_compliance_score(related, [policy, points])
    record(f'相关会议仍正常核查（覆盖 {len(result2["covered_points"])} 条）',
           result2.get('point_template_rejected') is False
           and len(result2['covered_points']) >= 1,
           f'covered={result2["covered_points"]} missing={result2["missing_points"]}')


def test_summary_not_falsely_rejected():
    """
    正常改写不应被判为幻觉。

    回归：一段 166 字的通话，Qwen 生成的摘要完全正确，但因为摘要里
    「3 字以上实体」只有 3 个且都不在原文中，novel_ratio 达到 1.0，
    被误判为幻觉并退回粗糙的抽取式拼接。
    """
    print('\n[9e] 摘要幻觉判定（小样本不应误杀）')
    from modules.text_analyzer import _is_hallucinated

    source = ('唉，李总，你好。我是中国人寿的小王。现在之前王总啊给我推荐联系您啊。'
              '他之前在我这边配置了那个全家医疗险和重疾险。觉得咱们这个方案还是比较实在的。'
              '通过他也是了解到您平常很看重家人的健康。资产托稳稳定。')
    good_summary = ('小王向李总介绍自己，并表示之前王总推荐其与李总对接家庭保险事宜。'
                    '李总对家庭成员健康非常重视，希望获得更全面的保障。'
                    '小王提议进行免费的家庭保单体检，以确保现有保险方案的完善性。')
    record('正确的改写式摘要不被判为幻觉', _is_hallucinated(good_summary, source) is False)

    real_hallucination = ('据记者报道，央行于3月15日发布《新规》，'
                          '证监会消息人士称将调整政策，网络版全文已上线。')
    record('新闻式幻觉仍被拦住', _is_hallucinated(real_hallucination, source) is True)

    record('空摘要视为无效', _is_hallucinated('', source) is True)


def test_decision_excludes_contextual_tongguo():
    """
    「通过」在中文里多数是「经由」之意，不能当决议动词。

    回归：保险通话里「通过他也是了解到您平常很看重家人的健康」
    被误判为决议结论。
    """
    print('\n[9f] 决议动词歧义（「通过」）')
    from modules.text_analyzer import extract_decisions

    text = '通过他也是了解到您平常很看重家人的健康。我考虑一下再决定。'
    decisions = extract_decisions(text)
    record(f'「经由」义的「通过」不算决议（得到 {len(decisions)} 条）',
           len(decisions) == 0, str(decisions))

    real = '会议通过了新的销售管理办法。会议决定下周组织合规培训。'
    decisions2 = extract_decisions(real)
    record(f'真实决议仍能提取（得到 {len(decisions2)} 条）',
           len(decisions2) >= 1, str(decisions2))


def test_domain_terms_not_split():
    """行业词汇应注册进 jieba，不被切成无意义碎词。"""
    print('\n[9g] 行业词汇分词')
    from modules.text_analyzer import _ensure_domain_terms

    _ensure_domain_terms()
    import jieba

    words = list(jieba.lcut('我是中国人寿的小王，配置了全家医疗险和重疾险'))
    record('「中国人寿」不被切碎', '中国人寿' in words, str(words))
    record('「医疗险」「重疾险」保持整词',
           '医疗险' in words and '重疾险' in words)

    from modules.text_analyzer import extract_keywords

    kws = [k['word'] for k in extract_keywords(
        '唉，李总，你好。我是中国人寿的小王。他之前在我这边配置了那个全家医疗险和重疾险。', top_n=8)]
    record('关键词里不再出现被切碎的「中国」', '中国' not in kws, str(kws))


def test_risk_deduplication():
    """同一风险句式重复出现不应重复计数。"""
    print('\n[5] 风险检测去重')
    from modules.compliance_checker import detect_semantic_risks

    text = '浪费时间。浪费时间。浪费时间。'
    risks = detect_semantic_risks(text, [{'text': text, 'start_time': 0, 'end_time': 9}])
    record(f'重复 3 次的风险句式产生 {len(risks)} 条记录（期望 1 条）', len(risks) == 1)

    # 无分句时也不应重复
    risks_no_seg = detect_semantic_risks(text)
    record(f'无分句时产生 {len(risks_no_seg)} 条记录（期望 1 条）', len(risks_no_seg) == 1)


def test_negation_context():
    """禁止语境中的关键词不应被判为风险。"""
    print('\n[6] 否定/禁止语境识别')
    from modules.compliance_checker import is_negated_context

    cases = [
        ('我们禁止使用保本保收益这类表述', '保本保收益', True),
        ('不得承诺保本收益', '保本收益', True),
        ('Do Not Use LIBOR', 'libor', True),
        ('这款产品保本保收益，绝对安全', '保本保收益', False),
    ]
    passed = True
    for text, keyword, expected in cases:
        actual = is_negated_context(text, keyword)
        if actual != expected:
            passed = False
            print(f'      ✗ {text!r} + {keyword!r}: 期望 {expected}，实际 {actual}')
    record('禁止语境判定（含大小写归一化）', passed)


def test_malformed_knowledge_data():
    """知识库字段为 null / 字符串时不应崩溃。"""
    print('\n[7] 异常知识库数据健壮性')
    from modules.compliance_checker import calculate_compliance_score

    bad_null = _FakeItem('空字段', '内容', 'key_points', keywords=None, required_points=None)
    bad_null.keywords = 'null'
    bad_null.required_points = 'null'

    bad_string = _FakeItem('字符串字段', '内容', 'policy')
    bad_string.keywords = '"保本,收益"'
    bad_string.required_points = '"要点一"'

    ok = True
    try:
        calculate_compliance_score('测试文本', [bad_null, bad_string])
    except Exception as exc:
        ok = False
        print(f'      ✗ 抛出异常: {type(exc).__name__}: {exc}')

    record('字段为 null / 字符串时不崩溃', ok)


def test_evaluation_metrics():
    """说话人混淆率评估函数本身的正确性。"""
    print('\n[8] 混淆率评估函数')
    from modules import speaker_diarization

    reference = [
        {'start': 0, 'end': 10, 'speaker': 'SPEAKER_00'},
        {'start': 10, 'end': 20, 'speaker': 'SPEAKER_01'},
    ]
    perfect = list(reference)
    m_perfect = speaker_diarization.evaluate(reference, perfect)
    record(f'完全一致时混淆率={m_perfect["confusion_rate"]:.2%}', m_perfect['confusion_rate'] == 0.0)

    # 说话人标签互换不应算错（标签是任意的，靠最优映射对齐）
    swapped = [
        {'start': 0, 'end': 10, 'speaker': 'SPEAKER_07'},
        {'start': 10, 'end': 20, 'speaker': 'SPEAKER_03'},
    ]
    m_swapped = speaker_diarization.evaluate(reference, swapped)
    record(f'标签互换时混淆率={m_swapped["confusion_rate"]:.2%}', m_swapped['confusion_rate'] == 0.0)

    wrong = [
        {'start': 0, 'end': 20, 'speaker': 'SPEAKER_00'},
    ]
    m_wrong = speaker_diarization.evaluate(reference, wrong)
    record(f'全部判为同一人时混淆率={m_wrong["confusion_rate"]:.2%}（应显著大于0）',
           m_wrong['confusion_rate'] > 0.3)


# ---------------------------------------------------------------- 文本结构化

def test_action_items_and_decisions():
    """待办事项与决议结论提取。"""
    print('\n[9] 结构化纪要（待办事项 / 决议结论）')
    from modules.text_analyzer import extract_action_items, extract_decisions

    text = (
        '今天我们讨论产品销售合规问题。'
        '会议决定下周组织一次全员合规培训。'
        '我们一致同意加强销售过程的录音录像管理。'
        '请各部门必须在本周五之前提交整改报告。'
        '销售团队需要落实风险告知书的书面确认流程。'
    )
    actions = extract_action_items(text)
    decisions = extract_decisions(text)

    record(f'待办事项提取 {len(actions)} 条', len(actions) >= 2, str(actions[:2]))
    record(f'决议结论提取 {len(decisions)} 条', len(decisions) >= 2, str(decisions[:2]))

    # 提取结果不应互为片段（重叠匹配产生的同义碎片要合并）
    has_fragment = any(a in b for a in actions for b in actions if a is not b)
    record('待办事项之间无互为片段的重复条目', not has_fragment, str(actions))


def test_noise_segment_filter():
    """
    低能量且极短的伪语音片段应被丢弃，避免产生幻觉文本。

    回归：真实电话录音里 VAD 把 1 秒的呼吸/碰麦声当成语音，
    识别出「有二十。」「I.」这类不存在的句子。
    """
    print('\n[9c] 伪语音片段过滤')
    import numpy as np

    from modules.asr_engine import _drop_noise_regions

    sr = 16000
    # 4 段「语音」（等幅噪声，能量相当） + 1 段极短且极低能量的伪语音
    audio = np.zeros(sr * 10, dtype=np.float32)
    regions = []
    rng = np.random.default_rng(0)
    for i in range(4):
        start = 0.5 + i * 2.0
        audio[int(start * sr):int((start + 1.5) * sr)] = rng.normal(0, 0.1, int(1.5 * sr))
        regions.append((start, start + 1.5))
    # 伪语音：1 秒、能量只有语音的约 1/25
    audio[int(9.0 * sr):int(9.5 * sr)] = rng.normal(0, 0.008, int(0.5 * sr))
    regions.append((9.0, 9.5))

    kept, dropped = _drop_noise_regions(audio, sr, regions)
    record(f'丢弃伪语音片段 {len(dropped)} 个（期望 1），保留 {len(kept)} 个（期望 4）',
           len(dropped) == 1 and len(kept) == 4)

    # 片段太少时不做判断（中位数不可靠）
    kept2, dropped2 = _drop_noise_regions(audio, sr, regions[:2])
    record(f'片段过少时不误判（保留 {len(kept2)} 个，丢弃 {len(dropped2)} 个）',
           len(kept2) == 2 and len(dropped2) == 0)


def test_speaker_fragment_merge():
    """
    碎片簇（≤2 窗口 / <2.5s / 单连续段）应被归并，避免单说话人被拆成多人。

    回归：真实单侧通话录音被判成 2 个说话人，多出来的簇只有 2 个窗口、1.7 秒。
    """
    print('\n[9d] 说话人碎片簇归并')
    import numpy as np

    from modules import speaker_diarization as sd

    # 10 个窗口：前 9 个属于同一说话人，最后 1 个是边界抖动的碎片（1 窗口、1s）
    base = np.random.default_rng(1).normal(0, 0.02, (9, 8)).astype(np.float32)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    frag = base[0:1] + np.random.default_rng(2).normal(0, 0.004, (1, 8)).astype(np.float32)
    frag /= np.linalg.norm(frag, axis=1, keepdims=True)
    embeddings = np.vstack([base, frag])
    windows = [(i * 2.0, i * 2.0 + 1.5) for i in range(9)] + [(18.0, 19.0)]
    labels = np.array([0] * 9 + [1], dtype=int)

    merged = sd._merge_fragment_clusters(embeddings, labels, windows)
    record(f'碎片簇被归并（合并后簇数 {len(set(merged))}，期望 1）', len(set(merged)) == 1)

    # 两个真正的说话人（各 5 个窗口，跨度长）不应被归并
    left = base[:5]
    right = base[5:]
    labels2 = np.array([0] * 5 + [1] * 4, dtype=int)
    merged2 = sd._merge_fragment_clusters(np.vstack([left, right]), labels2, windows[:9])
    record(f'正常说话人簇不被误并（簇数 {len(set(merged2))}，期望 2）', len(set(merged2)) == 2)


def test_extraction_filters_questions():
    """
    提问与不确定表述不应被当成待办/决议。

    回归：旧实现用单字字符类当引导词（如 [计划预计安排]），会在词中间命中，
    抽出「行卡，线上填写问卷就可以」这类碎片；且「我考虑一下再决定」
    会被误判为决议结论。
    """
    print('\n[9b] 待办/决议的疑问句过滤')
    from modules.text_analyzer import extract_action_items, extract_decisions

    questions = '这个需要多长时间？我们需要准备什么材料吗？你们决定了吗？'
    actions = extract_action_items(questions)
    decisions = extract_decisions(questions)
    record(f'纯提问不产生待办（得到 {len(actions)} 条）', len(actions) == 0, str(actions))
    record(f'纯提问不产生决议（得到 {len(decisions)} 条）', len(decisions) == 0, str(decisions))

    real_call = (
        '你好，我想咨询一下这个理财产品的风险等级。'
        '风险测评大概需要多长时间，需要准备什么材料？'
        '只需要身份证和银行卡，线上填写问卷就可以。'
        '明白了，那收益是怎么计算的，有没有保本承诺。'
        '好的，谢谢你的说明，我考虑一下再决定。'
        '不客气，风险告知书需要您本人签字确认。'
    )
    actions2 = extract_action_items(real_call)
    decisions2 = extract_decisions(real_call)
    record(f'真实通话：决议结论不含「我考虑一下再决定」',
           all('考虑一下' not in d for d in decisions2), str(decisions2))
    # 不应出现从词中间切出来的碎片
    fragments = [a for a in actions2 if a.startswith(('行', '计', '算', '下', '实'))]
    record(f'真实通话：无词中间截断的碎片', not fragments, str(fragments))


def test_sentiment_neutral():
    """中性会议叙述不应被判为负面情绪。"""
    print('\n[10] 情绪分析（中性文本）')
    from modules.text_analyzer import analyze_sentiment

    neutral_texts = [
        '会议开始，请大家汇报一下本周的工作。',
        '今天主要讨论项目进度和下周的工作安排。',
    ]
    positive_text = '本周工作非常顺利，项目进展很好，大家都完成得很出色。'
    negative_text = '这个问题很麻烦，进度延迟了，客户也不满意，大家都有些抱怨。'

    ok = True
    for text in neutral_texts:
        result = analyze_sentiment(text)
        if result['sentiment'] != 'neutral':
            ok = False
            print(f'      ✗ {text!r} -> {result["sentiment"]}（期望 neutral）')

    pos = analyze_sentiment(positive_text)
    neg = analyze_sentiment(negative_text)
    if pos['sentiment'] != 'positive':
        ok = False
        print(f'      ✗ 正面文本 -> {pos["sentiment"]}（期望 positive）')
    if neg['sentiment'] != 'negative':
        ok = False
        print(f'      ✗ 负面文本 -> {neg["sentiment"]}（期望 negative）')

    record('中性/正面/负面三分类正确', ok)


def test_summary_generation():
    """摘要生成可用且不超出长度上限。"""
    print('\n[11] 摘要生成')
    from modules.text_analyzer import generate_summary

    text = (
        '各位同事大家好，今天我们召开产品销售合规会议。'
        '首先需要说明风险等级，我们销售的产品属于中等风险等级。'
        '同时要说明投资限制，本产品起投金额为五万元。'
        '关于业绩提醒义务，请注意提醒过往业绩不代表未来。'
        '向客户介绍时必须充分揭示风险，不得有任何遗漏。'
        '会议决定下周组织一次全员合规培训。'
        '请各部门在本周五之前提交整改报告。'
    )
    summary = generate_summary(text, max_length=300)
    record(f'摘要长度 {len(summary)} 字符，非空且不超限',
           bool(summary) and 0 < len(summary) <= 320, summary[:60])


# ---------------------------------------------------------------- 报告与接口

def test_report_generator():
    """报告生成：空分数不崩溃 + HTML 转义。"""
    print('\n[12] 报告生成健壮性')
    from modules.report_generator import generate_compliance_trend_report, generate_report_html

    meetings = [
        {'id': 1, 'title': '<img src=x onerror=alert(1)>', 'date': '2026-10-01T10:00:00',
         'duration': 60, 'total_score': None, 'score_level': None, 'summary': '摘要<script>',
         'compliance_report': {'risk_keywords': '[]', 'missing_points': '[]'}},
        {'id': 2, 'title': '正常会议', 'date': '2026-10-02T10:00:00',
         'duration': 120, 'total_score': 88.5, 'score_level': '优秀', 'summary': '摘要',
         'compliance_report': {'risk_keywords': '["保本"]', 'missing_points': '["风险告知"]'}},
    ]

    ok = True
    try:
        report = generate_compliance_trend_report(meetings)
        html = generate_report_html('compliance_trend', report)
    except Exception as exc:
        ok = False
        print(f'      ✗ 抛出异常: {type(exc).__name__}: {exc}')
        html = ''

    if ok and '<img src=x onerror=alert(1)>' in html:
        ok = False
        print('      ✗ HTML 未转义，存在注入风险')

    record('空分数不崩溃 + HTML 转义', ok)


def test_http_api():
    """HTTP 接口冒烟测试（使用 Flask 测试客户端）。"""
    print('\n[13] HTTP 接口冒烟')
    try:
        from app import app
    except Exception as exc:
        record('导入应用', False, f'{type(exc).__name__}: {exc}')
        return

    client = app.test_client()

    response = client.get('/api/health')
    record(f'GET /api/health -> {response.status_code}', response.status_code == 200)

    response = client.get('/api/meetings?page=1&page_size=5')
    record(f'GET /api/meetings -> {response.status_code}', response.status_code == 200)

    response = client.post('/api/meetings/test-summary',
                           json={'text': '会议决定下周开展合规培训，请各部门落实整改。'})
    ok = response.status_code == 200
    if ok:
        data = response.get_json().get('data', {})
        ok = 'action_items' in data and 'decisions' in data
    record(f'POST /api/meetings/test-summary -> {response.status_code}', ok)

    response = client.post('/api/meetings/test-analyze', json={'text': '本产品保本保收益，绝对安全。'})
    ok = response.status_code == 200
    detail = ''
    if ok:
        payload = response.get_json()
        ok = payload.get('code') == 200
        detail = f"score={payload.get('data', {}).get('total_score') if payload.get('data') else 'no-kb'}"
    else:
        detail = response.get_data(as_text=True)[:120]
    record(f'POST /api/meetings/test-analyze -> {response.status_code}', ok, detail)

    response = client.get('/api/reports/compliance-trend')
    record(f'GET /api/reports/compliance-trend -> {response.status_code}',
           response.status_code in (200, 404))

    wav = os.path.join(AUDIO_DIR, 'gt_2spk.wav')
    if os.path.exists(wav):
        with open(wav, 'rb') as handle:
            response = client.post(
                '/api/v1/transcribe',
                data={'audio': (handle, 'gt_2spk.wav'), 'enable_diarization': 'true',
                      'enable_compliance': 'true'},
                content_type='multipart/form-data',
            )
        ok = response.status_code == 200
        detail = ''
        if ok:
            payload = response.get_json()
            data = payload.get('data') or {}
            speakers = {s.get('speaker') for s in data.get('transcriptions', [])}
            ok = payload.get('code') == 200 and bool(data.get('text'))
            detail = (f"文本{len(data.get('text',''))}字, "
                      f"{len(data.get('transcriptions', []))}句, 说话人{sorted(speakers)}")
        else:
            detail = response.get_data(as_text=True)[:200]
        record(f'POST /api/v1/transcribe -> {response.status_code}', ok, detail)
    else:
        record('POST /api/v1/transcribe', False, '缺少测试音频 gt_2spk.wav')


def test_upload_analysis_pipeline():
    """
    上传音频分析路径端到端（POST /api/meetings）——含合规检查。

    这是一个回归测试：该路径在后台线程里执行，线程内没有 Flask 应用上下文，
    所有数据库访问都必须包在 app_context 内。曾因把「查知识库 / 查评分权重」
    放在上下文外而必然抛 "Working outside of application context."，
    导致页面上传音频后在 80% 处分析失败。
    """
    print('\n[14] 上传分析路径端到端（后台线程 + 应用上下文）')
    import time as _time

    from modules.audio_io import DEFAULT_SR, load_audio, save_wav

    source = os.path.join(AUDIO_DIR, 'gt_2spk.wav')
    if not os.path.exists(source):
        record('上传分析路径', False, '缺少 gt_2spk.wav')
        return

    # 截取前 14 秒，缩短识别与摘要耗时
    trimmed = os.path.join(AUDIO_DIR, 'upload_smoke.wav')
    try:
        audio = load_audio(source, sr=DEFAULT_SR)[: 14 * DEFAULT_SR]
        save_wav(trimmed, audio, DEFAULT_SR)
    except Exception as exc:
        record('上传分析路径', False, f'准备测试音频失败: {exc}')
        return

    try:
        from app import app
    except Exception as exc:
        record('上传分析路径', False, f'导入应用失败: {exc}')
        return

    client = app.test_client()
    meeting_id = None
    try:
        with open(trimmed, 'rb') as handle:
            response = client.post(
                '/api/meetings',
                data={
                    'audio_file': (handle, 'upload_smoke.wav'),
                    'meeting_title': '上传路径回归测试',
                    'enable_diarization': 'true',
                    'enable_compliance': 'true',
                },
                content_type='multipart/form-data',
            )
        ok = response.status_code == 200
        payload = response.get_json() or {}
        meeting_id = payload.get('meeting_id')
        record(f'POST /api/meetings -> {response.status_code}, meeting_id={meeting_id}',
               ok and bool(meeting_id))
        if not meeting_id:
            return

        # 轮询后台线程进度
        final = None
        deadline = _time.time() + 300
        while _time.time() < deadline:
            progress = (client.get(f'/api/meetings/{meeting_id}/progress').get_json() or {}).get('data', {})
            if progress.get('progress') == 100:
                final = progress
                break
            if progress.get('progress') == -1:
                record('后台分析完成', False, f"失败于 {progress.get('message')}")
                return
            _time.sleep(2)

        if final is None:
            record('后台分析完成', False, '等待超时（300s）')
            return
        record('后台分析完成（未抛应用上下文异常）', True, final.get('message', ''))

        detail = (client.get(f'/api/meetings/{meeting_id}').get_json() or {}).get('data', {})
        transcriptions = detail.get('transcriptions') or []
        speakers = {t.get('speaker') for t in transcriptions}
        record(f'转写入库 {len(transcriptions)} 句，说话人 {sorted(speakers)}',
               len(transcriptions) > 0 and len(speakers) >= 2)

        # 说话人分离必须用原始音频：曾误用预处理后的音频，导致 2 人被并成 1 人
        record(f'说话人分离未因预处理退化（识别到 {len(speakers)} 人，期望 ≥2）',
               len(speakers) >= 2)

        report = detail.get('compliance_report')
        ok = bool(report) and 'total_score' in report and 'missing_points' in report
        record(f'合规报告入库 score={report.get("total_score") if report else None}',
               ok)

        record('待办事项与决议结论字段存在',
               'action_items' in detail and 'decisions' in detail)

        # 音频质量报告应真正落库（此前只在局部变量里算完就丢，界面永远显示"暂无"）
        quality = detail.get('audio_quality')
        ok = isinstance(quality, dict) and all(
            k in quality for k in ('noise_reduction', 'snr_before', 'snr_after', 'improvement'))
        record(f'音频质量报告已落库并在详情返回 {quality if ok else quality}', ok)
    finally:
        # 清理本次测试产生的数据，避免污染演示库
        if meeting_id:
            try:
                client.delete(f'/api/meetings/{meeting_id}')
            except Exception:
                pass
        if os.path.exists(trimmed):
            try:
                os.remove(trimmed)
            except OSError:
                pass


def main():
    quick = '--quick' in sys.argv
    print('=' * 78)
    print('语音链路量化评测')
    print('=' * 78)

    heavy = [
        ('转写准确率', test_transcription_accuracy),
        ('说话人分离', test_diarization_accuracy),
        ('说话人归属', test_speaker_attribution),
    ]
    light = [
        ('要点覆盖', test_point_coverage),
        ('离题遗漏', test_point_coverage_offtopic),
        ('风险去重', test_risk_deduplication),
        ('否定语境', test_negation_context),
        ('异常数据', test_malformed_knowledge_data),
        ('混淆率评估', test_evaluation_metrics),
        ('结构化纪要', test_action_items_and_decisions),
        ('疑问句过滤', test_extraction_filters_questions),
        ('伪语音过滤', test_noise_segment_filter),
        ('碎片簇归并', test_speaker_fragment_merge),
        ('摘要幻觉判定', test_summary_not_falsely_rejected),
        ('决议动词歧义', test_decision_excludes_contextual_tongguo),
        ('行业词汇分词', test_domain_terms_not_split),
        ('情绪分析', test_sentiment_neutral),
        ('摘要生成', test_summary_generation),
        ('报告生成', test_report_generator),
        ('HTTP 接口', test_http_api),
        ('上传分析路径', test_upload_analysis_pipeline),
    ]

    tests = light if quick else (heavy + light)

    for name, func in tests:
        try:
            func()
        except Exception as exc:
            record(name, False, f'用例异常 {type(exc).__name__}: {exc}')
            traceback.print_exc()

    print('\n' + '=' * 78)
    print('评测汇总')
    print('=' * 78)
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    for name, ok, detail in RESULTS:
        print(f'  {"✅" if ok else "❌"} {name}' + (f'  ({detail})' if detail else ''))
    print(f'\n通过 {passed}/{total} ({passed / total * 100:.1f}%)')

    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())
