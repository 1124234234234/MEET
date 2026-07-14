import pyttsx3
import os

output_file = r'd:\games\新建文件夹 (7)\voice-reco\test_meeting_3speakers.mp3'

engine = pyttsx3.init()
voices = engine.getProperty('voices')

print(f'Available voices: {len(voices)}')
for i, v in enumerate(voices):
    print(f'  {i}: {v.name} ({v.languages})')

if len(voices) >= 3:
    v0 = voices[0]
    v1 = voices[1]
    v2 = voices[2]
elif len(voices) == 2:
    v0 = voices[0]
    v1 = voices[1]
    v2 = voices[0]
else:
    v0 = voices[0]
    v1 = voices[0]
    v2 = voices[0]

dialogues = [
    (v0, 180, "大家好，今天我们开个短会，主要讨论一下新推出的理财产品销售方案。最近有客户反映，我们的销售人员在介绍产品时，有些表述不太准确。"),
    (v1, 160, "是的，我也注意到了这个问题。上周有个客户投诉，说我们的理财经理承诺保本保收益，这明显违反了监管规定。"),
    (v2, 200, "我补充一下，公司在宣传材料中提到，可以承诺保本保收益，绝对安全。大家可以放心推荐给客户。"),
    (v1, 160, "这个表述是不对的。我们禁止承诺保本保收益，也禁止夸大产品收益。这是合规红线，任何人都不允许越过。"),
    (v0, 180, "对，这个必须纠正。我觉得我们需要加强对销售人员的培训，全员的培训，确保每个销售人员都了解产品的风险等级和合规要求。"),
    (v2, 200, "我同意。建议下周组织一次全员培训，重点讲解产品的风险特征和销售话术规范。培训内容应该包括以下几点。第一，必须如实告知客户投资风险。第二，不得使用任何可能误导客户的表述。第三，提醒客户过往业绩不代表未来表现。"),
    (v1, 160, "另外，我们要求所有销售人员在销售过程中，必须向客户书面确认风险告知书，不能口头说一下就完了。这个必须落实到位。如果发现有不签署风险告知书就销售的情况，要严肃处理。"),
    (v0, 180, "我强调一下，不得向客户承诺固定收益，不得使用保本、零风险等误导性词语。违反规定的，已经发现立即停止处理。"),
    (v2, 200, "好的，我们总结一下今天会议的要求。第一，禁止承诺保本保收益，禁止夸大产品收益。第二，必须如实告知客户投资风险，签署风险告知书。第三，下周组织全员销售合规培训。第四，提醒客户过往业绩不代表未来表现。第五，违反合规规定的，严肃处理。"),
    (v0, 180, "好的，大家都清楚了吧？有什么问题随时沟通。今天就到这里。"),
    (v1, 160, "谢谢大家。"),
]

segments = []
from pydub import AudioSegment
import tempfile

temp_files = []
for i, (voice, rate, text) in enumerate(dialogues):
    engine.setProperty('voice', voice.id)
    engine.setProperty('rate', rate)
    temp_path = os.path.join(tempfile.gettempdir(), f'seg_{i}.wav')
    engine.save_to_file(text, temp_path)
    engine.runAndWait()
    temp_files.append(temp_path)

combined = AudioSegment.empty()
for tf in temp_files:
    seg = AudioSegment.from_wav(tf)
    combined += seg
    combined += AudioSegment.silent(duration=300)
    try:
        os.remove(tf)
    except:
        pass

combined.export(output_file, format='mp3', bitrate='64k')
print(f'Generated: {output_file}')
print(f'Duration: {len(combined)/1000:.1f}s')
print(f'File size: {os.path.getsize(output_file)/1024:.1f} KB')
