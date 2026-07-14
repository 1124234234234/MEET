"""
完整端到端验证测试
验证三大核心改进：合规分析误判修复、说话人分离、摘要大模型
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ========== 测试1：合规分析误判修复 ==========
print("=" * 60)
print("测试1：合规分析 - 否定语境识别")
print("=" * 60)

from modules.compliance_checker import is_negated_context

test_cases = [
    ("禁止承诺保本保收益", "保本保收益", True, "禁止承诺的应该被豁免"),
    ("不得使用零风险等表述", "零风险", True, "不得使用的应该被豁免"),
    ("严禁保证收益", "保证收益", True, "严禁的应该被豁免"),
    ("产品有保本保收益的特点", "保本保收益", False, "肯定语境应该检测到风险"),
    ("不能承诺保本", "保本", True, "不能承诺的应该被豁免"),
    ("投资者需要承担投资风险", "投资风险", False, "正常表述不应豁免"),
    ("不允许使用误导性词汇", "误导性", True, "不允许应该被豁免"),
]

all_pass = True
for text, keyword, expected, desc in test_cases:
    result = is_negated_context(text, keyword)
    status = "PASS" if result == expected else "FAIL"
    if result != expected:
        all_pass = False
    print(f"  [{status}] '{text}' + '{keyword}' => 豁免={result}, 预期={expected} | {desc}")

print(f"\n  合规分析测试: {'全部通过' if all_pass else '有失败'}\n")


# ========== 测试2：说话人分离 ==========
print("=" * 60)
print("测试2：说话人分离 - 3人识别")
print("=" * 60)

audio_file = "test_compliance_meeting_3min.mp3"
if os.path.exists(audio_file):
    from modules.speaker_diarization import speaker_diarization_simple
    t0 = time.time()
    segments = speaker_diarization_simple(audio_file, num_speakers=3, timeout=120)
    elapsed = time.time() - t0

    if segments:
        speakers = set(s['speaker'] for s in segments)
        print(f"  耗时: {elapsed:.1f}s")
        print(f"  段落数: {len(segments)}")
        print(f"  说话人数: {len(speakers)} (指定: 3)")
        for sp in sorted(speakers):
            sp_segs = [s for s in segments if s['speaker'] == sp]
            dur = sum(s['end'] - s['start'] for s in sp_segs)
            print(f"    {sp}: {len(sp_segs)}段, {dur:.1f}s")
    else:
        print("  说话人分离失败")
else:
    print(f"  测试音频不存在: {audio_file}")


# ========== 测试3：摘要生成（大模型+TextRank） ==========
print()
print("=" * 60)
print("测试3：摘要生成 - mT5大模型+TextRank")
print("=" * 60)

os.environ['HF_HUB_OFFLINE'] = '1'
from modules.text_analyzer import generate_summary

test_text = """
各位同事大家好，今天我们召开理财产品销售合规培训会议。
首先，我想强调一下合规销售的重要性。合规是我们公司经营的生命线。
接下来，我们讨论投资者适当性管理。根据监管要求，必须在销售前对客户进行风险测评。
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

print("  [3a] mT5大模型模式:")
t0 = time.time()
summary_model = generate_summary(test_text, max_length=400, use_model=True)
t1 = time.time()
print(f"  耗时: {t1-t0:.1f}s")
print(f"  长度: {len(summary_model)}字")
print(f"  内容: {summary_model[:200]}")

print()
print("  [3b] 纯TextRank模式:")
summary_tr = generate_summary(test_text, max_length=400, use_model=False)
print(f"  长度: {len(summary_tr)}字")
print(f"  内容: {summary_tr[:200]}")


# ========== 测试4：合规评分全流程 ==========
print()
print("=" * 60)
print("测试4：合规评分全流程")
print("=" * 60)

from modules.compliance_checker import calculate_compliance_score

# 模拟知识库
class MockItem:
    def __init__(self, title, content, status='active', item_type='required',
                 required_points=None, keywords=None, risk_keywords=None):
        self.title = title
        self.content = content
        self.status = status
        self.item_type = item_type
        self.required_points = json.dumps(required_points or [])
        self.keywords = json.dumps(keywords or [])
        self.risk_keywords = json.dumps(risk_keywords or [])

import json

kb_items = [
    MockItem(
        "理财产品销售合规要求",
        "理财产品销售必须遵守合规要求，包括风险测评和风险告知",
        risk_keywords=["保本保收益", "零风险", "绝对安全", "稳赚不赔"]
    ),
]

result = calculate_compliance_score(test_text, kb_items)
print(f"  合规总分: {result['total_score']:.1f}")
print(f"  风险词检测: {result.get('risk_keywords_found', [])}")
print(f"  说明: '禁止承诺保本保收益'中的'保本保收益'应该被豁免，不应出现在风险词中")

# 检查"保本保收益"是否被错误标记为风险
risk_words = result.get('risk_keywords_found', [])
if '保本保收益' in risk_words:
    print("  [FAIL] '保本保收益'仍被误判为风险词！")
elif len(risk_words) == 0:
    print("  [PASS] 无误判风险词")
else:
    print(f"  [INFO] 检测到的风险词: {risk_words}")


print()
print("=" * 60)
print("全部测试完成！")
print("=" * 60)
