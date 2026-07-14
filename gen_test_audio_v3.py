import edge_tts
import asyncio
import os
import tempfile
from pydub import AudioSegment

OUTPUT_FILE = "d:/games/新建文件夹 (7)/voice-reco/test_compliance_meeting_3min.mp3"
TARGET_SAMPLE_RATE = 16000

dialogues = [
    ("zh-CN-XiaoxiaoNeural", "各位同事大家好，欢迎参加今天的理财产品销售合规培训会议。"),
    ("zh-CN-YunyangNeural", "好的，主持人。我们都已经准备好了，请开始吧。"),
    ("zh-CN-YunjianNeural", "是的，我们非常重视这次培训，希望能够更好地理解合规要求。"),
    
    ("zh-CN-XiaoxiaoNeural", "首先，我想强调一下合规销售的重要性。合规是我们公司经营的生命线，任何违规行为都将受到严肃处理。"),
    ("zh-CN-YunyangNeural", "明白，合规确实是第一位的，我们必须严格遵守各项规定。"),
    ("zh-CN-YunjianNeural", "没错，只有合规经营，公司才能长久发展。"),
    
    ("zh-CN-XiaoxiaoNeural", "接下来，我们讨论投资者适当性管理。根据监管要求，我们必须在销售前对客户进行风险测评。"),
    ("zh-CN-YunyangNeural", "这个我知道，要确保客户的风险承受能力与产品风险等级相匹配。"),
    ("zh-CN-YunjianNeural", "对，不能把高风险产品卖给风险承受能力低的客户。"),
    
    ("zh-CN-XiaoxiaoNeural", "非常正确。而且，我们必须如实告知客户产品的风险等级。这款产品属于中高风险等级，投资者需要承担一定的投资风险。"),
    ("zh-CN-YunyangNeural", "明白了，不能隐瞒风险，必须向客户充分说明可能的损失。"),
    ("zh-CN-YunjianNeural", "是的，要让客户清楚了解投资风险，做出明智的投资决策。"),
    
    ("zh-CN-XiaoxiaoNeural", "第二点，关于风险告知义务。我们必须明确向客户说明投资风险，包括市场风险、信用风险和流动性风险。"),
    ("zh-CN-YunyangNeural", "市场风险是指市场波动可能导致产品净值下降，对吧？"),
    ("zh-CN-YunjianNeural", "信用风险是指发行人可能无法按时支付利息或本金。"),
    
    ("zh-CN-XiaoxiaoNeural", "说得很对。特别注意，禁止承诺保本保收益，禁止使用零风险等误导性表述。"),
    ("zh-CN-YunyangNeural", "这个是重点，绝对不能承诺保本保收益，这是合规红线。"),
    ("zh-CN-YunjianNeural", "是的，一旦违反，后果很严重。"),
    
    ("zh-CN-XiaoxiaoNeural", "第三点，关于销售行为规范。所有销售人员必须持证上岗，销售过程必须录音录像。"),
    ("zh-CN-YunyangNeural", "持证上岗是基本要求，没有销售资格证是不能开展销售工作的。"),
    ("zh-CN-YunjianNeural", "录音录像是为了保留证据，确保销售过程合规透明。"),
    
    ("zh-CN-XiaoxiaoNeural", "必须向客户书面确认风险告知书，不能口头代替。"),
    ("zh-CN-YunyangNeural", "这个必须落实到位，客户签字确认才能进行下一步。"),
    ("zh-CN-YunjianNeural", "对，风险告知书是重要的法律文件。"),
    
    ("zh-CN-XiaoxiaoNeural", "另外，请各位记住，要提醒客户过往业绩不代表未来表现。"),
    ("zh-CN-YunyangNeural", "很多客户可能会认为过去表现好未来也会好，这点必须明确说明。"),
    ("zh-CN-YunjianNeural", "是的，投资有风险，历史业绩仅供参考。"),
    
    ("zh-CN-XiaoxiaoNeural", "关于产品宣传，我们必须使用公司统一的宣传材料，不得自行制作或修改宣传内容。"),
    ("zh-CN-YunyangNeural", "明白，统一宣传材料可以确保信息的准确性和合规性。"),
    ("zh-CN-YunjianNeural", "不能为了销售而夸大产品收益或隐瞒风险。"),
    
    ("zh-CN-XiaoxiaoNeural", "最后，请大家务必遵守公司的合规要求，确保每一笔销售都合规合法。"),
    ("zh-CN-YunyangNeural", "我们一定会严格遵守，绝不触碰合规红线。"),
    ("zh-CN-YunjianNeural", "如果发现有违规行为，我们会及时上报，共同维护公司的合规环境。"),
    
    ("zh-CN-XiaoxiaoNeural", "好的，今天的培训内容就这些。接下来有什么问题大家可以提问。"),
    ("zh-CN-YunyangNeural", "主持人，请问如果客户坚持要购买超出其风险承受能力的产品怎么办？"),
    ("zh-CN-XiaoxiaoNeural", "如果客户坚持购买，我们需要进行特别风险提示，并要求客户签署书面确认。"),
    
    ("zh-CN-YunjianNeural", "明白了，这样既保护了客户，也保护了我们自己。"),
    ("zh-CN-XiaoxiaoNeural", "是的。还有其他问题吗？"),
    ("zh-CN-YunyangNeural", "暂时没有了，谢谢主持人。"),
    
    ("zh-CN-XiaoxiaoNeural", "好的，今天的培训就到这里，谢谢大家的参与。"),
    ("zh-CN-YunyangNeural", "感谢主持人的讲解，我们会认真落实这些要求。"),
    ("zh-CN-YunjianNeural", "有问题随时沟通，确保合规销售。"),
]

voices = {
    "zh-CN-XiaoxiaoNeural": "主持人（女）",
    "zh-CN-YunyangNeural": "经理A（男）",
    "zh-CN-YunjianNeural": "经理B（男）"
}

async def generate_segment(text, voice, output_file):
    try:
        communicate = edge_tts.Communicate(text, voice, rate="-10%")
        await communicate.save(output_file)
        return True
    except Exception as e:
        print(f"  生成失败: {e}")
        return False

async def main():
    temp_files = []
    
    print("=" * 60)
    print("生成合规会议测试音频（3分钟版）")
    print(f"目标采样率: {TARGET_SAMPLE_RATE} Hz")
    print(f"说话人数: 3人")
    print("=" * 60 + "\n")

    for i, (voice, text) in enumerate(dialogues):
        temp_file = os.path.join(tempfile.gettempdir(), f"audio_seg_{i}.mp3")
        speaker = voices.get(voice, "未知")
        print(f"[{i+1}/{len(dialogues)}] {speaker}: {text[:25]}...")
        
        success = await generate_segment(text, voice, temp_file)
        if success:
            temp_files.append(temp_file)
        await asyncio.sleep(0.3)

    print(f"\n合并音频片段...")
    combined = AudioSegment.empty()
    
    for tf in temp_files:
        seg = AudioSegment.from_mp3(tf)
        seg = seg.set_frame_rate(TARGET_SAMPLE_RATE).set_channels(1)
        combined += seg
        combined += AudioSegment.silent(duration=400)

    combined.export(OUTPUT_FILE, format="mp3", bitrate="128k")

    for tf in temp_files:
        try:
            os.remove(tf)
        except:
            pass

    file_size = os.path.getsize(OUTPUT_FILE)
    duration = len(combined) / 1000
    
    print("\n" + "=" * 60)
    print(f"音频生成完成: {OUTPUT_FILE}")
    print(f"文件大小: {file_size / 1024:.1f} KB")
    print(f"时长: {duration:.1f} 秒 ({duration/60:.1f} 分钟)")
    print(f"采样率: {TARGET_SAMPLE_RATE} Hz")
    print(f"对话段数: {len(temp_files)} 段")
    print(f"说话人数: 3人（主持人、经理A、经理B）")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())