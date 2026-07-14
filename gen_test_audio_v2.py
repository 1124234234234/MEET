import edge_tts
import asyncio
import os
import tempfile
from pydub import AudioSegment

OUTPUT_FILE = "d:/games/新建文件夹 (7)/voice-reco/test_compliance_meeting.mp3"
TARGET_SAMPLE_RATE = 16000

dialogues = [
    ("zh-CN-XiaoxiaoNeural", "各位同事大家好，今天我们召开理财产品销售合规培训会议。"),
    ("zh-CN-YunyangNeural", "好的，我先来介绍一下这次培训的主要内容。"),
    ("zh-CN-YunjianNeural", "我们需要确保每位销售人员都能够合规销售理财产品。"),
    ("zh-CN-XiaoxiaoNeural", "首先，关于投资者适当性管理，我们必须在销售前对客户进行风险测评。"),
    ("zh-CN-YunyangNeural", "这个我知道，要确保客户的风险承受能力与产品风险等级匹配。"),
    ("zh-CN-YunjianNeural", "非常正确。而且，我们必须如实告知客户产品的风险等级。"),
    ("zh-CN-XiaoxiaoNeural", "这款产品属于中高风险等级，投资者需要承担一定的投资风险。"),
    ("zh-CN-YunyangNeural", "明白了，不能隐瞒风险，必须向客户充分说明。"),
    ("zh-CN-YunjianNeural", "是的。第二点，关于风险告知义务。"),
    ("zh-CN-XiaoxiaoNeural", "我们必须明确向客户说明投资风险，包括市场风险、信用风险和流动性风险。"),
    ("zh-CN-YunyangNeural", "特别注意，禁止承诺保本保收益，禁止使用零风险等误导性表述。"),
    ("zh-CN-YunjianNeural", "这些都是合规红线，必须严格遵守，任何人都不能越过。"),
    ("zh-CN-XiaoxiaoNeural", "说得对。第三点，关于销售行为规范。"),
    ("zh-CN-YunyangNeural", "所有销售人员必须持证上岗，销售过程必须录音录像。"),
    ("zh-CN-YunjianNeural", "必须向客户书面确认风险告知书，不能口头代替。"),
    ("zh-CN-XiaoxiaoNeural", "另外，请各位记住，要提醒客户过往业绩不代表未来表现。"),
    ("zh-CN-YunyangNeural", "最后，请大家务必遵守公司的合规要求，确保每一笔销售都合规合法。"),
    ("zh-CN-YunjianNeural", "如果发现有违规行为，我们会严肃处理，绝不姑息。"),
    ("zh-CN-XiaoxiaoNeural", "好的，今天的培训就到这里，谢谢大家。"),
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
    print("生成合规会议测试音频")
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