"""用 Whisper 转写测试音频，验证能否检测到风险词"""
import whisper
import os

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_audio_files')
test_files = ['compliant_meeting.wav', 'risky_meeting.wav', 'multi_topic.wav']

model = None

for f in test_files:
    path = os.path.join(base, f)
    print(f"\n{'='*60}")
    print(f"转写: {f}")
    print('='*60)

    if model is None:
        print("加载 Whisper 模型 (base)...")
        model = whisper.load_model("base")

    print("转写中...")
    result = model.transcribe(path, language='zh', initial_prompt="以下是普通话的句子。")
    text = result['text']

    print(f"\n转写结果:\n{text}")
    print(f"\n段落数: {len(result['segments'])}")

    # 检查关键词
    risk_keywords = ['保本', '保收益', '夸大']
    required_keywords = ['风险等级', '投资限制', '过往业绩', '未来表现']

    found_risk = [kw for kw in risk_keywords if kw in text]
    found_required = [kw for kw in required_keywords if kw in text]

    print(f"\n风险词命中: {found_risk if found_risk else '无'}")
    print(f"必传要点命中: {found_required if found_required else '无'}")

print("\n" + "="*60)
print("测试完成")
