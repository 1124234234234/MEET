"""
测试三个改进：
1. 会议摘要质量
2. 合规分析误判修复
3. 说话人分离改进
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("测试 1: 合规分析误判修复")
print("=" * 60)

from modules.compliance_checker import is_negated_context, detect_semantic_risks

test_cases = [
    ("禁止承诺保本保收益，这是合规红线", True, "禁止 + 保本保收益 - 应该被识别为否定语境"),
    ("我们不能使用零风险这种误导性表述", True, "不能 + 零风险 - 应该被识别为否定语境"),
    ("这款产品绝对安全，大家放心购买", False, "直接使用绝对安全 - 应该是风险"),
    ("我们承诺保本保收益，收益很高", False, "直接承诺保本保收益 - 应该是风险"),
    ("严禁夸大产品收益，必须如实告知", True, "严禁 + 夸大收益 - 应该被识别为否定语境"),
]

all_pass = True
for text, expected_negated, desc in test_cases:
    result = is_negated_context(text, "保本保收益" if "保本" in text else "零风险" if "零风险" in text else "绝对安全" if "绝对安全" in text else "夸大")
    # 简单测试：取文本中的第一个风险词
    import re
    risk_words = ['保本保收益', '零风险', '无风险', '绝对安全', '夸大', '稳赚', '一定赚']
    kw = None
    for rw in risk_words:
        if rw in text:
            kw = rw
            break
    if kw:
        result = is_negated_context(text, kw)
        status = "✓" if result == expected_negated else "✗"
        if result != expected_negated:
            all_pass = False
        print(f"  {status} {desc}")
        print(f"    文本: {text}")
        print(f"    关键词: {kw}, 否定语境: {result}, 预期: {expected_negated}")
    print()

# 测试语义风险检测
print("  测试语义风险检测（合规会议内容不应误判）：")
compliant_text = """
本次会议强调合规销售的重要性。禁止承诺保本保收益，禁止使用零风险等误导性表述。
必须如实告知客户产品的风险等级，不得夸大产品收益。
所有销售人员必须持证上岗，销售过程必须录音录像。
"""
risks = detect_semantic_risks(compliant_text)
print(f"  合规会议文本中检测到的风险数: {len(risks)} (预期: 0)")
if len(risks) > 0:
    print("  检测到的风险:")
    for r in risks:
        print(f"    - {r['keyword']} ({r['category']})")
    all_pass = False
else:
    print("  ✓ 合规会议未被误判")
print()

print("  测试真正的风险内容：")
risky_text = """
这款产品绝对安全，保本保收益，大家放心推荐给客户。
我跟你说，这个肯定赚，稳赚不赔，收益很高的。
"""
risks = detect_semantic_risks(risky_text)
print(f"  风险会议文本中检测到的风险数: {len(risks)} (预期: >0)")
if len(risks) > 0:
    print("  ✓ 正确检测到风险内容:")
    for r in risks:
        print(f"    - {r['keyword']} ({r['category']})")
else:
    print("  ✗ 未检测到风险内容")
    all_pass = False
print()

print("=" * 60)
print("测试 2: 会议摘要生成")
print("=" * 60)

from modules.text_analyzer import generate_summary

test_text = """
各位同事大家好，今天我们召开理财产品销售合规培训会议。
首先，我想强调一下合规销售的重要性。合规是我们公司经营的生命线。
接下来，我们讨论投资者适当性管理。根据监管要求，我们必须在销售前对客户进行风险测评。
要确保客户的风险承受能力与产品风险等级相匹配。
这款产品属于中高风险等级，投资者需要承担一定的投资风险。
关于风险告知义务，我们必须明确向客户说明投资风险。
特别注意，禁止承诺保本保收益，禁止使用零风险等误导性表述。
所有销售人员必须持证上岗，销售过程必须录音录像。
必须向客户书面确认风险告知书，不能口头代替。
我们要提醒客户过往业绩不代表未来表现。
建议下周组织一次全员销售合规培训。
如果客户坚持购买超出其风险承受能力的产品，需要进行特别风险提示。
好的，今天的培训就到这里，谢谢大家。
"""

summary = generate_summary(test_text, max_length=300)
print(f"  输入文本长度: {len(test_text)} 字")
print(f"  摘要长度: {len(summary)} 字")
print(f"  摘要内容:")
print(f"    {summary}")
print()

# 简单质量检查
has_first_person = any(w in summary for w in ['我', '我们', '你', '你们', '他', '他们'])
print(f"  包含第一人称: {has_first_person} (越少越好)")
print(f"  以'本次会议'开头: {'本次会议' in summary[:10]}")
print()

print("=" * 60)
print("测试 3: 说话人分离")
print("=" * 60)

audio_file = "test_compliance_meeting_3min.mp3"
if os.path.exists(audio_file):
    from modules.speaker_diarization import speaker_diarization_simple
    import time
    
    print(f"  测试音频: {audio_file}")
    print(f"  正在进行说话人分离（自动估计说话人数）...")
    t0 = time.time()
    segments = speaker_diarization_simple(audio_file, num_speakers=None, timeout=120)
    elapsed = time.time() - t0
    
    print(f"  耗时: {elapsed:.1f} 秒")
    print(f"  分离出的段落数: {len(segments)}")
    
    if segments:
        speakers = set(s['speaker'] for s in segments)
        print(f"  识别出的说话人数: {len(speakers)} (预期: 3人)")
        print(f"  说话人列表: {', '.join(sorted(speakers))}")
        
        for sp in sorted(speakers):
            sp_segs = [s for s in segments if s['speaker'] == sp]
            total_dur = sum(s['end'] - s['start'] for s in sp_segs)
            print(f"    {sp}: {len(sp_segs)} 段, 总时长 {total_dur:.1f}s")
    else:
        print("  说话人分离失败或无结果")
else:
    print(f"  测试音频文件不存在: {audio_file}")
    print("  跳过说话人分离测试")

print()
print("=" * 60)
print(f"测试完成! 整体结果: {'全部通过' if all_pass else '部分失败'}")
print("=" * 60)