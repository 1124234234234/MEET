"""
生成带精确标注的多人会议测试音频（用于量化评测说话人分离与转写准确率）。

做法：用 Windows SAPI 逐个合成对话句子，对每个"说话人"应用不同的
语速与音高变换构造可区分的声纹，按已知时间轴拼接，并输出逐句的
说话人与起止时间标注（ground truth）。

产物：
  tests/test_audio_files/gt_meeting.wav    测试音频
  tests/test_audio_files/gt_meeting.json   标注（含每句话的说话人与时间）

用法：python tests/make_ground_truth_audio.py
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.audio_io import DEFAULT_SR, load_audio, save_wav

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_audio_files')
WAV_PATH = os.path.join(OUTPUT_DIR, 'gt_meeting.wav')
JSON_PATH = os.path.join(OUTPUT_DIR, 'gt_meeting.json')

VOICE = 'Microsoft Huihui Desktop'
SILENCE_BETWEEN = 0.45

# (说话人序号, 语速, 音高半音, 文本)
DIALOGUE_3SPK = [
    (0, -1, 0.0, '各位同事大家好，今天我们召开产品销售合规会议'),
    (1, 2, 4.0, '好的，我先汇报一下上周的整改进展情况'),
    (2, -3, -3.0, '我这边补充一点，客户投诉主要集中在收益表述上'),
    (0, -1, 0.0, '这个问题必须引起重视，我们要重新梳理销售话术'),
    (1, 2, 4.0, '我们建议下周组织一次全员合规培训，覆盖风险告知要求'),
    (2, -3, -3.0, '同意这个安排，同时需要落实销售过程的录音录像'),
    (0, -1, 0.0, '那么请相关部门在本周五之前提交整改报告'),
    (1, 2, 4.0, '明白，我们会同步更新投资者适当性管理的材料'),
    (2, -3, -3.0, '另外提醒大家，禁止使用保本保收益这类表述'),
    (0, -1, 0.0, '好的，今天的会议就到这里，谢谢大家'),
]

# 两人对话场景：用于验证算法不会把 2 人会议强行切成 3 人
DIALOGUE_2SPK = [
    (0, -1, 0.0, '你好，我想咨询一下这个理财产品的风险等级'),
    (1, 3, 4.5, '这款产品属于中等风险，需要先做风险测评'),
    (0, -1, 0.0, '风险测评大概需要多长时间，需要准备什么材料'),
    (1, 3, 4.5, '只需要身份证和银行卡，线上填写问卷就可以'),
    (0, -1, 0.0, '明白了，那收益是怎么计算的，有没有保本承诺'),
    (1, 3, 4.5, '没有保本承诺，过往业绩也不代表未来表现'),
    (0, -1, 0.0, '好的，谢谢你的说明，我考虑一下再决定'),
    (1, 3, 4.5, '不客气，风险告知书需要您本人签字确认'),
]

SCENARIOS = [
    ('gt_meeting', DIALOGUE_3SPK),
    ('gt_2spk', DIALOGUE_2SPK),
]

SPEAKER_LABELS = ['SPEAKER_00', 'SPEAKER_01', 'SPEAKER_02']


def synthesize(text, rate, out_path):
    """用 SAPI 合成单句语音。"""
    ps_script = f'''
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SelectVoice("{VOICE}")
$synth.Rate = {rate}
$synth.Volume = 100
$synth.SetOutputToWaveFile("{out_path}")
$synth.Speak(@'
{text}
'@)
$synth.Dispose()
Write-Output "OK"
'''
    script_file = out_path + '.ps1'
    with open(script_file, 'w', encoding='utf-8-sig') as handle:
        handle.write(ps_script)
    try:
        result = subprocess.run(
            ['powershell', '-ExecutionPolicy', 'Bypass', '-File', script_file],
            capture_output=True, text=True, encoding='gbk',
        )
    finally:
        if os.path.exists(script_file):
            os.remove(script_file)

    if result.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError(f'SAPI 合成失败: {result.stderr}')


def make_voice_variant(path, semitones):
    """用相位声码器改变音高，构造不同的声纹特征。"""
    if abs(semitones) < 0.01:
        return load_audio(path, sr=DEFAULT_SR)

    import librosa

    y = load_audio(path, sr=DEFAULT_SR)
    shifted = librosa.effects.pitch_shift(y, sr=DEFAULT_SR, n_steps=semitones)
    return np.asarray(shifted, dtype=np.float32)


def build_scenario(name, dialogue):
    """按对话脚本合成一个测试音频，返回标注信息。"""
    timeline = []
    annotations = []
    cursor = 0.0
    temp_files = []

    try:
        for index, (speaker_idx, rate, semitones, text) in enumerate(dialogue):
            temp_path = os.path.join(tempfile.gettempdir(), f'{name}_{index}.wav')
            synthesize(text, rate, temp_path)
            temp_files.append(temp_path)

            audio = make_voice_variant(temp_path, semitones)
            start = cursor
            end = start + len(audio) / DEFAULT_SR
            timeline.append(audio)
            annotations.append({
                'speaker': SPEAKER_LABELS[speaker_idx],
                'start': round(start, 3),
                'end': round(end, 3),
                'text': text,
            })
            cursor = end + SILENCE_BETWEEN
            timeline.append(np.zeros(int(SILENCE_BETWEEN * DEFAULT_SR), dtype=np.float32))

        combined = np.concatenate(timeline)
        wav_path = os.path.join(OUTPUT_DIR, f'{name}.wav')
        json_path = os.path.join(OUTPUT_DIR, f'{name}.json')
        save_wav(wav_path, combined, DEFAULT_SR)

        used_speakers = sorted({t['speaker'] for t in annotations})
        payload = {
            'audio': os.path.basename(wav_path),
            'sample_rate': DEFAULT_SR,
            'duration': round(len(combined) / DEFAULT_SR, 3),
            'speakers': used_speakers,
            'turns': annotations,
            'full_text': ''.join(t['text'] + '。' for t in annotations),
        }
        with open(json_path, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

        print(f'{wav_path}')
        print(f'  时长 {payload["duration"]}s，{len(annotations)} 句，'
              f'{len(used_speakers)} 个说话人 -> {json_path}')
    finally:
        for path in temp_files:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for name, dialogue in SCENARIOS:
        build_scenario(name, dialogue)


if __name__ == '__main__':
    main()
