#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成会议测试音频文件
模拟理财销售会议场景，包含合规要点和风险内容
"""

import asyncio
import os
import edge_tts
from edge_tts import Communicate

# 输出目录
OUTPUT_DIR = r"d:\games\新建文件夹 (7)\voice-reco\uploads"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 说话人语音配置
VOICES = {
    "host": "zh-CN-XiaoxiaoNeural",      # 主持人-女声
    "manager": "zh-CN-YunxiNeural",      # 客户经理-男声
    "compliance": "zh-CN-XiaoyiNeural",  # 合规专员-女声（温柔但严肃）
}

# 会议对话脚本（包含合规要点和风险内容）
SCRIPT = [
    {
        "speaker": "host",
        "name": "主持人",
        "text": "各位同事大家好，今天我们来讨论一下新推出的理财产品销售方案。请客户经理小王来介绍一下产品情况。",
        "pause_after": 1.5
    },
    {
        "speaker": "manager",
        "name": "客户经理小王",
        "text": "好的，谢谢大家。这款理财产品年化收益率可以达到百分之八以上，而且我们公司承诺保本保收益，绝对安全，大家可以放心推荐给客户。",
        "pause_after": 1.5
    },
    {
        "speaker": "compliance",
        "name": "合规专员李姐",
        "text": "小王，等一下。根据合规要求，我们必须向客户说明产品的风险等级是中高风险。另外，投资限制是不能用于短期周转。还有，大家必须提醒客户，过往业绩不代表未来表现。",
        "pause_after": 2.0
    },
    {
        "speaker": "manager",
        "name": "客户经理小王",
        "text": "哦对，我会注意的。不过如果客户问起来，我可以说这款产品基本不会亏钱的吧？毕竟历史数据来看确实没亏过。",
        "pause_after": 1.5
    },
    {
        "speaker": "compliance",
        "name": "合规专员李姐",
        "text": "不行，绝对不能这么说。我们明确禁止承诺保本保收益，也禁止夸大产品收益。销售人员必须如实告知客户投资风险，不得使用任何可能误导客户的表述。",
        "pause_after": 1.0
    },
    {
        "speaker": "host",
        "name": "主持人",
        "text": "好的，今天的会议就到这里。大家务必遵守合规要求，做好风险揭示工作。散会。",
        "pause_after": 0
    }
]

async def generate_segment(text, voice, output_path):
    """生成单段语音"""
    communicate = Communicate(text, voice)
    await communicate.save(output_path)
    print(f"  生成: {os.path.basename(output_path)}")

async def generate_silence(duration_sec, output_path):
    """生成静音片段"""
    # 使用 ffmpeg 生成静音 mp3
    cmd = f'ffmpeg -f lavfi -i anullsrc=r=24000:cl=mono -t {duration_sec} -acodec libmp3lame -q:a 4 "{output_path}" -y'
    os.system(cmd)
    print(f"  生成静音: {duration_sec}秒")

async def main():
    print("=" * 60)
    print("正在生成会议测试音频...")
    print("=" * 60)
    
    segment_files = []
    temp_dir = os.path.join(OUTPUT_DIR, "temp_segments")
    os.makedirs(temp_dir, exist_ok=True)
    
    # 生成每段语音和静音间隔
    for i, item in enumerate(SCRIPT):
        print(f"\n段落 {i+1}/{len(SCRIPT)} - {item['name']}:")
        print(f"  内容: {item['text'][:40]}...")
        
        # 生成语音
        voice = VOICES[item["speaker"]]
        seg_path = os.path.join(temp_dir, f"seg_{i:02d}.mp3")
        await generate_segment(item["text"], voice, seg_path)
        segment_files.append(seg_path)
        
        # 生成静音间隔
        if item["pause_after"] > 0:
            silence_path = os.path.join(temp_dir, f"silence_{i:02d}.mp3")
            await generate_silence(item["pause_after"], silence_path)
            segment_files.append(silence_path)
    
    # 创建 ffmpeg concat 列表文件
    concat_list = os.path.join(temp_dir, "concat_list.txt")
    with open(concat_list, "w", encoding="utf-8") as f:
        for seg_file in segment_files:
            # ffmpeg concat 需要单引号包裹路径（Windows路径含特殊字符）
            f.write(f"file '{seg_file}'\n")
    
    # 合并所有片段
    final_output = os.path.join(OUTPUT_DIR, "meeting_test_audio.mp3")
    print(f"\n正在合并音频...")
    
    # 使用 ffmpeg concat 合并
    concat_cmd = f'ffmpeg -f concat -safe 0 -i "{concat_list}" -acodec libmp3lame -q:a 2 "{final_output}" -y'
    result = os.system(concat_cmd)
    
    if result == 0 and os.path.exists(final_output):
        file_size = os.path.getsize(final_output) / 1024  # KB
        print(f"\n✓ 音频生成成功!")
        print(f"  文件: {final_output}")
        print(f"  大小: {file_size:.1f} KB")
        
        # 获取音频时长
        duration_cmd = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{final_output}"'
        duration = os.popen(duration_cmd).read().strip()
        if duration:
            print(f"  时长: {float(duration):.1f} 秒")
    else:
        print(f"\n✗ 音频合并失败")
    
    # 清理临时文件
    print(f"\n清理临时文件...")
    for f in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, f))
    os.rmdir(temp_dir)
    
    print("\n" + "=" * 60)
    print("音频内容预览:")
    print("=" * 60)
    for i, item in enumerate(SCRIPT):
        marker = "⚠ 风险内容" if "保本" in item["text"] or "保收益" in item["text"] or "基本不会亏钱" in item["text"] else ""
        marker += " ✓ 合规要点" if "风险等级" in item["text"] or "投资限制" in item["text"] or "过往业绩" in item["text"] else ""
        print(f"\n[{item['name']}] {marker}")
        print(f"  {item['text']}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    asyncio.run(main())