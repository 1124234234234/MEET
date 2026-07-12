import edge_tts
import asyncio
import os
import time

# 3个说话人：主持人(晓晓-女)、经理1(云扬-男)、经理2(云健-男)
dialogues = [
    # 第一轮：开场介绍
    ("zh-CN-XiaoxiaoNeural", "各位同事大家好，今天我们召开这次会议，主要讨论新推出的理财产品销售方案，以及相关的合规要求。"),
    ("zh-CN-YunyangNeural", "好的，我先来介绍一下这款理财产品的基本情况。这款产品是一款中高风险的理财产品，预期年化收益率可以达到百分之八以上，在目前市场上算是比较有竞争力的。"),
    ("zh-CN-YunjianNeural", "那这个产品的风险等级是怎么定的？我们需要向客户怎么说明？"),

    # 第二轮：风险讨论
    ("zh-CN-YunyangNeural", "这款产品的风险等级是中高风险，投资者需要承担一定的投资风险。我们必须向客户明确说明这一点，不能含糊其辞。"),
    ("zh-CN-XiaoxiaoNeural", "我补充一下，公司在宣传材料中提到可以承诺保本保收益，绝对安全，大家可以放心推荐给客户。这个表述是不对的。"),
    ("zh-CN-YunjianNeural", "对，这个必须纠正。我们禁止承诺保本保收益，也禁止夸大产品收益。这是合规红线，任何人都不允许越过。"),

    # 第三轮：销售培训
    ("zh-CN-XiaoxiaoNeural", "我觉得我们需要加强对销售人员的培训，确保每个销售人员都了解产品的风险等级和合规要求。"),
    ("zh-CN-YunyangNeural", "我同意，建议下周组织一次全员培训，重点讲解产品的风险特征和销售话术规范。"),
    ("zh-CN-YunjianNeural", "培训内容应该包括以下几点：第一，必须如实告知客户投资风险；第二，不得使用任何可能误导客户的表述；第三，提醒客户过往业绩不代表未来表现。"),

    # 第四轮：合规要求
    ("zh-CN-XiaoxiaoNeural", "另外，我们要求所有销售人员在销售过程中，必须向客户书面确认风险知悉书，不能口头说一下就完了。"),
    ("zh-CN-YunyangNeural", "这个必须落实到位。如果发现有没有签署风险知悉书就销售的情况，要严肃处理。"),
    ("zh-CN-YunjianNeural", "我强调一下，不得向客户承诺固定收益，不得使用保本、零风险等误导性词语。违反规定的，一经发现立即停职处理。"),

    # 第五轮：总结
    ("zh-CN-XiaoxiaoNeural", "好的，我们总结一下今天会议的要求。第一，禁止承诺保本保收益，禁止夸大产品收益。"),
    ("zh-CN-YunyangNeural", "第二，必须如实告知客户投资风险，签署风险知悉书。第三，下周组织全员销售合规培训。"),
    ("zh-CN-YunjianNeural", "第四，提醒客户过往业绩不代表未来表现。第五，违反合规规定的严肃处理。"),
    ("zh-CN-XiaoxiaoNeural", "好的，大家都清楚了吧？有什么问题随时沟通。今天就到这里，谢谢大家。"),
]

async def generate_audio_with_retry(text, voice, output_file, retries=3):
    for attempt in range(retries):
        try:
            communicate = edge_tts.Communicate(text, voice, rate="-15%")
            await communicate.save(output_file)
            return True
        except Exception as e:
            print(f"  尝试 {attempt+1} 失败: {e}")
            if attempt < retries - 1:
                await asyncio.sleep(2)
    return False

async def generate_multi_speaker_audio():
    temp_files = []
    voices = {"zh-CN-XiaoxiaoNeural": "主持人", "zh-CN-YunyangNeural": "经理A", "zh-CN-YunjianNeural": "经理B"}
    
    for i, (voice, text) in enumerate(dialogues):
        temp_file = f"temp_part_{i}.mp3"
        speaker = voices.get(voice, "未知")
        print(f"生成片段 {i+1}/{len(dialogues)} [{speaker}]: {text[:20]}...")
        
        success = await generate_audio_with_retry(text, voice, temp_file)
        if success:
            temp_files.append(temp_file)
        else:
            print(f"  跳过片段 {i+1}")
        await asyncio.sleep(0.5)
    
    # 合并音频
    output_file = "d:/games/新建文件夹 (7)/voice-reco/test_meeting_3min.mp3"
    
    # 用二进制方式合并（mp3可以直接拼接）
    combined = b""
    for tf in temp_files:
        with open(tf, "rb") as f:
            data = f.read()
            combined += data
            # 片段之间加一点静音（重复前几帧作为间隔）
            combined += data[:200]
    
    with open(output_file, "wb") as f:
        f.write(combined)
    
    # 清理临时文件
    for tf in temp_files:
        try:
            os.remove(tf)
        except:
            pass
    
    file_size = os.path.getsize(output_file)
    print(f"\n音频生成完成: test_meeting_3min.mp3")
    print(f"文件大小: {file_size / 1024:.1f} KB")
    print(f"对话轮数: {len(temp_files)} 段")
    print(f"说话人数: 3人（主持人、经理A、经理B）")

asyncio.run(generate_multi_speaker_audio())
