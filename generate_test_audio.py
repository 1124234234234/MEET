from pydub import AudioSegment
from pydub.generators import Square
import numpy as np

dialogues = [
    "各位同事大家好，今天我们召开理财产品销售培训会议。",
    "首先，请允许我介绍一下本次培训的主要内容。",
    "好的，我们认真听讲。",
    "本次培训的核心是确保每位销售人员都能够合规销售理财产品。",
    "首先，关于投资者适当性管理，我们必须在销售前对客户进行风险测评。",
    "这个我知道，要确保客户的风险承受能力与产品风险等级匹配。",
    "非常正确。而且，我们必须如实告知客户产品的风险等级。",
    "这款产品属于中高风险等级，投资者需要承担一定的投资风险。",
    "明白了，不能隐瞒风险。",
    "是的。第二点，关于风险告知义务。",
    "我们必须明确向客户说明投资风险，包括市场风险、信用风险和流动性风险。",
    "特别注意，禁止承诺保本保收益，禁止使用零风险等误导性表述。",
    "这些都是合规红线，必须严格遵守。",
    "说得对。第三点，关于销售行为规范。",
    "所有销售人员必须持证上岗，销售过程必须录音录像。",
    "必须向客户书面确认风险告知书，不能口头代替。",
    "另外，请各位记住，要提醒客户过往业绩不代表未来表现。",
    "最后，请大家务必遵守公司的合规要求，确保每一笔销售都合规合法。",
    "好的，今天的培训就到这里，谢谢大家。",
]

def text_to_audio(text, speaker_freq=400):
    audio = AudioSegment.empty()
    
    for char in text:
        if char in '，。！？、':
            pause = AudioSegment.silent(duration=300)
            audio += pause
        else:
            char_code = ord(char)
            freq = speaker_freq + (char_code % 100)
            duration = 120
            
            samples = np.random.randn(int(duration * 44.1)) * 0.5
            samples = samples.astype(np.float32)
            
            audio_segment = AudioSegment(
                samples.tobytes(),
                frame_rate=44100,
                sample_width=4,
                channels=1
            )
            
            audio_segment = audio_segment - 15
            audio += audio_segment
        
        audio += AudioSegment.silent(duration=30)
    
    return audio

combined = AudioSegment.empty()

for i, text in enumerate(dialogues):
    freq = 400 if i % 2 == 0 else 500
    segment = text_to_audio(text, freq)
    combined += segment
    combined += AudioSegment.silent(duration=500)

output_path = 'compliant_meeting_test.mp3'
combined.export(output_path, format='mp3', bitrate='128k')

import os
print(f'音频文件已生成: {output_path}')
print(f'文件大小: {os.path.getsize(output_path) / 1024:.2f} KB')
print(f'时长: {len(combined) / 1000:.2f} 秒')