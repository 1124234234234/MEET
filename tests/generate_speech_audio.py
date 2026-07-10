"""
生成真实中文语音测试音频
使用 Windows SAPI (Microsoft Huihui Desktop 中文女声)
通过 subprocess 调用 PowerShell 避免编码问题

测试音频内容基于知识库的必传要点和风险词：
- 风险说明义务 (required): 风险等级、投资限制
- 禁止保本承诺 (forbidden): 保本、保收益、夸大
- 业绩提醒义务 (required): 过往业绩、未来表现
"""
import os
import subprocess
import sys

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_audio_files")


# 测试音频内容
AUDIO_SCRIPTS = {
    "compliant_meeting.wav": {
        "description": "合规会议音频 - 完整覆盖必传要点，无风险词",
        "sentences": [
            "各位同事大家好，今天我们召开产品销售培训会议。",
            "首先需要说明风险等级，我们销售的产品属于中等风险等级，请向客户明确说明。",
            "同时要说明投资限制，本产品起投金额为五万元，投资期限不低于一年。",
            "关于业绩提醒义务，请注意提醒过往业绩不代表未来。",
            "我们展示的过往业绩数据仅为历史表现，不代表未来表现。",
            "向客户介绍时必须充分揭示风险，不得有任何遗漏。",
            "希望大家严格遵守合规要求，规范销售行为。",
        ],
    },
    "risky_meeting.wav": {
        "description": "风险会议音频 - 包含禁止词（保本、保收益、夸大）",
        "sentences": [
            "今天我们讨论如何向客户推销这款理财产品。",
            "这个产品很好卖，你可以跟客户说这个产品保本，绝对安全。",
            "而且我们还可以承诺保收益，每年至少百分之五的收益。",
            "大家销售的时候可以适当夸大一下产品的收益。",
            "不要跟客户提什么风险，风险等级也不用说，免得客户不敢买。",
            "投资限制也不要提，让客户多买一点。",
            "过往业绩就直接说未来肯定能做到，这样客户才会相信。",
        ],
    },
    "multi_topic.wav": {
        "description": "多主题讨论音频 - 涉及项目讨论、决策制定、计划安排",
        "sentences": [
            "今天我们召开项目进度讨论会。",
            "首先汇报一下上周项目进展，开发任务已经完成了百分之八十。",
            "关于技术方案，我们决定采用微服务架构，这样系统性能更好。",
            "接下来需要安排下一阶段的计划，预计下周完成测试工作。",
            "测试团队需要负责跟进测试用例的编写和执行。",
            "关于市场推广方案，营销部门建议下个月启动线上宣传。",
            "最后，我们需要在下周三之前提交项目预算审批。",
            "请各位同事按照分工落实执行，有问题及时反馈。",
        ],
    },
}


def generate_audio_via_powershell(output_path, text):
    """
    通过 PowerShell 调用 SAPI 生成中文语音
    Python subprocess 在 Windows 上会用 ANSI 编码传递参数，中文能正确处理
    """
    # PowerShell 脚本：使用 here-string 避免转义问题
    ps_script = f'''
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SelectVoice("Microsoft Huihui Desktop")
$synth.Rate = 0
$synth.Volume = 100
$synth.SetOutputToWaveFile("{output_path}")
$synth.Speak(@'
{text}
'@)
$synth.Dispose()
Write-Output "OK"
'''

    # 用 UTF-8 编码写入临时文件
    script_file = output_path + ".ps1"
    with open(script_file, "w", encoding="utf-8-sig") as f:  # BOM 让 PowerShell 正确识别
        f.write(ps_script)

    # 调用 PowerShell 执行
    result = subprocess.run(
        ["powershell", "-ExecutionPolicy", "Bypass", "-File", script_file],
        capture_output=True,
        text=True,
        encoding="gbk",
    )

    # 删除临时脚本
    if os.path.exists(script_file):
        os.remove(script_file)

    if result.returncode != 0:
        print(f"PowerShell Error: {result.stderr}")
        return False
    return True


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("生成真实中文语音测试音频")
    print("使用 Windows SAPI - Microsoft Huihui Desktop (中文女声)")
    print("=" * 60 + "\n")

    success_count = 0
    for filename, config in AUDIO_SCRIPTS.items():
        output_path = os.path.join(OUTPUT_DIR, filename)
        text = "。".join(config["sentences"])

        print(f"\n[{filename}]")
        print(f"  场景: {config['description']}")
        print(f"  句子数: {len(config['sentences'])}")
        print(f"  字数: {len(text)}")

        if generate_audio_via_powershell(output_path, text):
            size_kb = os.path.getsize(output_path) / 1024
            print(f"  生成成功: {size_kb:.1f} KB")
            success_count += 1
        else:
            print(f"  生成失败!")

    print("\n" + "=" * 60)
    print(f"完成: {success_count}/{len(AUDIO_SCRIPTS)} 个音频生成成功")
    print("=" * 60)

    if success_count > 0:
        print("\n文件列表:")
        for f in sorted(os.listdir(OUTPUT_DIR)):
            if f.endswith(".wav"):
                size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
                print(f"  {f}: {size/1024:.1f} KB")


if __name__ == "__main__":
    main()
