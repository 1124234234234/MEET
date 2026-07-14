"""
快速测试改进后的摘要和说话人分离
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("测试 1: 会议摘要生成")
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
print(f"  摘要长度: {len(summary)} 字")
print(f"  摘要内容:")
print(f"    {summary}")
print()

print("=" * 60)
print("测试 2: 说话人分离")
print("=" * 60)

audio_file = "test_compliance_meeting_3min.mp3"
if os.path.exists(audio_file):
    from modules.speaker_diarization import speaker_diarization_simple
    import time
    
    print(f"  测试音频: {audio_file}")
    print(f"  正在进行说话人分离（自动估计说话人数）...")
    t0 = time.time()
    segments = speaker_diarization_simple(audio_file, num_speakers=3, timeout=120)
    elapsed = time.time() - t0
    
    print(f"  耗时: {elapsed:.1f} 秒")
    print(f"  分离出的段落数: {len(segments)}")
    
    if segments:
        speakers = set(s['speaker'] for s in segments)
        print(f"  识别出的说话人数: {len(speakers)} (指定: 3人)")
        print(f"  说话人列表: {', '.join(sorted(speakers))}")
        
        for sp in sorted(speakers):
            sp_segs = [s for s in segments if s['speaker'] == sp]
            total_dur = sum(s['end'] - s['start'] for s in sp_segs)
            print(f"    {sp}: {len(sp_segs)} 段, 总时长 {total_dur:.1f}s")
    else:
        print("  说话人分离失败或无结果")
else:
    print(f"  测试音频文件不存在: {audio_file}")

print()
print("=" * 60)
print("测试完成!")
print("=" * 60)