import soundfile as sf
import os

files = ['compliant_meeting.wav', 'risky_meeting.wav', 'multi_topic.wav']
base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_audio_files')
for f in files:
    path = os.path.join(base, f)
    info = sf.info(path)
    size_kb = os.path.getsize(path) / 1024
    print(f"{f}: {info.duration:.1f}s | SR={info.samplerate} | CH={info.channels} | {size_kb:.1f} KB")
