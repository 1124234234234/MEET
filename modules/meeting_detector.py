"""
参会人员统计模块。

依据说话人分离结果统计参会人数与每人发言占比，供会议详情页展示。

说明：原先这里还有一个 MeetingDetector 类（按音频活跃度判断会议启停），
但全项目没有任何地方实例化它；且「人数阈值 + 持续时长判定启停」属于整体
项目里视频分析/边缘设备的职责，音频侧靠单路拾音无法可靠判定。为避免留下
从未被调用的死代码，已移除该类。
"""


def count_participants(speaker_segments):
    """根据说话人分离结果统计参会人员数量。"""
    if not speaker_segments:
        return 0

    speakers = set()
    for seg in speaker_segments:
        # speaker 键存在但值可能为 None（未做说话人分离），需按 unknown 处理
        speaker = seg.get('speaker') or 'unknown'
        speakers.add(speaker)

    return len(speakers)


def analyze_participation_distribution(speaker_segments):
    """
    分析每位参会人的发言分布。

    返回 {speaker: {'duration','count','percentage','segments'}}。
    """
    if not speaker_segments:
        return {}

    speaker_stats = {}
    for seg in speaker_segments:
        speaker = seg.get('speaker') or 'unknown'
        # 转写记录使用 start_time/end_time 键名，兼容旧的 start/end 键名
        duration = seg.get('end_time', seg.get('end', 0)) - seg.get('start_time', seg.get('start', 0))

        if speaker not in speaker_stats:
            speaker_stats[speaker] = {
                'duration': 0,
                'count': 0,
                'segments': [],
            }

        speaker_stats[speaker]['duration'] += duration
        speaker_stats[speaker]['count'] += 1
        speaker_stats[speaker]['segments'].append(seg)

    total_duration = sum(stats['duration'] for stats in speaker_stats.values())
    for stats in speaker_stats.values():
        stats['percentage'] = (stats['duration'] / total_duration * 100) if total_duration > 0 else 0
        stats['duration'] = round(stats['duration'], 2)
        stats['percentage'] = round(stats['percentage'], 2)

    return speaker_stats
