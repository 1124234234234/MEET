# 会议室智能合规分析系统 · API 接口文档

## 基础信息

- **服务地址**：`http://{host}:{port}`
  - 默认端口取 `config.py` 的 `PORT`（默认 `5001`），也可用环境变量 `PORT` 覆盖
  - 通过 `启动.bat` / `launcher.py` 启动时会自动选一个空闲端口，**以控制台打印的地址为准**
- **数据格式**：JSON（报告类接口返回 HTML）
- **字符编码**：UTF-8
- **跨域**：已启用 CORS；可用环境变量 `CORS_ORIGINS`（逗号分隔）限制来源

统一响应约定：JSON 接口都在响应体里带 `code`、`message`，业务数据放在 `data`。
`code` 与 HTTP 状态码一致（200 成功 / 400 参数错误 / 404 资源不存在 / 413 文件过大 / 500 服务异常）。

---

## 目录

1. [健康检查](#1-健康检查)
2. [语音转写分析（第三方集成）](#2-语音转写分析第三方集成)
3. [会议管理](#3-会议管理)
4. [实时转写（Socket.IO）](#4-实时转写socketio)
5. [知识库管理](#5-知识库管理)
6. [合规检查与自测](#6-合规检查与自测)
7. [评分权重](#7-评分权重)
8. [系统信息](#8-系统信息)
9. [报告生成](#9-报告生成)

---

## 1. 健康检查

### 1.1 服务状态

```
GET /api/health
```

```json
{
  "status": "ok",
  "message": "API service is running",
  "models": {
    "funasr_asr": "D:\\...\\models\\funasr\\iic--speech_paraformer-large...",
    "funasr_vad": "D:\\...\\models\\funasr\\iic--speech_fsmn_vad...",
    "funasr_punc": "D:\\...\\models\\funasr\\iic--punc_ct-transformer...",
    "embedding_model": true,
    "summary_model": true,
    "ready": true
  }
}
```

### 1.2 版本与时间

```
GET /api/v1/health
```

```json
{
  "code": 200,
  "message": "服务正常运行",
  "data": { "timestamp": "2026-10-08T18:00:00", "version": "1.0.0" }
}
```

---

## 2. 语音转写分析（第三方集成）

### 2.1 音频转写与分析（核心接口）

```
POST /api/v1/transcribe
```

**Content-Type**：`multipart/form-data` 或 `application/json`

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `audio` | File | 二选一 | - | 音频文件（表单字段名固定为 `audio`） |
| `audio_base64` | String | 二选一 | - | Base64 编码音频；也支持 `data:audio/wav;base64,...` 前缀 |
| `language` | String | 否 | `zh` | 支持 `zh` `en` `ja` `ko` `fr` `de` `es` `ru` `ar` `pt` |
| `enable_compliance` | Boolean | 否 | `true` | 是否做合规比对（需要知识库中已有启用条目） |
| `enable_diarization` | Boolean | 否 | `false` | 是否做说话人分离（开启后耗时增加，但可得到「谁说了什么」） |

布尔值接受 `true/false`、`1/0`、`yes/no`、`on/off`。
支持格式：`wav` `mp3` `ogg` `flac` `m4a` `webm` `mp4` `aac`；文件小于 1KB 会被拒绝。

**示例**

```bash
# 文件上传
curl -X POST http://127.0.0.1:5001/api/v1/transcribe \
  -F "audio=@meeting.mp3" \
  -F "enable_diarization=true"

# Base64 方式
curl -X POST http://127.0.0.1:5001/api/v1/transcribe \
  -H "Content-Type: application/json" \
  -d '{"audio_base64":"<base64>","language":"zh","enable_compliance":true}'
```

```python
import requests

with open('meeting.mp3', 'rb') as fh:
    resp = requests.post(
        'http://127.0.0.1:5001/api/v1/transcribe',
        files={'audio': ('meeting.mp3', fh, 'audio/mpeg')},
        data={'enable_diarization': 'true'},
        timeout=1800,
    )
print(resp.json())
```

**成功响应**（`code=200`）：

```json
{
  "code": 200,
  "message": "成功",
  "data": {
    "text": "各位同事，大家好。今天我们召开产品销售培训会议。",
    "transcriptions": [
      {
        "speaker": "SPEAKER_00",
        "text": "各位同事，大家好。",
        "start_time": 0.0,
        "end_time": 2.06,
        "confidence": 1.0
      }
    ],
    "segments": [
      { "text": "各位同事，大家好。", "start": 0.0, "end": 2.06, "confidence": 1.0 }
    ],
    "keywords": [
      { "word": "风险等级", "frequency": 2, "score": 0.2292 }
    ],
    "topics": [
      { "topic": "合规审查", "score": 0.72, "semantic_score": 0.68, "keyword_score": 0.9 }
    ],
    "summary": "本次会议围绕合规审查展开讨论，涉及风险等级、投资限制等核心内容。……",
    "sentiment": {
      "sentiment": "neutral",
      "score": 0.5,
      "positive_score": 0.5,
      "negative_score": 0.5
    },
    "action_items": ["必须在本周五之前提交整改报告"],
    "decisions": ["会议决定下周组织一次全员合规培训"],
    "audio_quality": {
      "noise_reduction": 44.51,
      "snr_before": 34.43,
      "snr_after": 67.23,
      "improvement": 32.8
    },
    "compliance_report": {
      "total_score": 66.47,
      "score_level": "合格",
      "components": {
        "semantic_similarity": 21.2,
        "point_coverage": 12.72,
        "risk_detection": 14.0,
        "keyword_matching": 0.0
      },
      "covered_points": ["风险告知义务"],
      "missing_points": ["投资者风险测评", "风险等级匹配"],
      "risk_keywords_found": ["保本保收益"],
      "risk_time_markers": [
        {
          "keyword": "保本保收益",
          "category": "金融销售风险关键词",
          "start_time": 12.48,
          "end_time": 14.83,
          "text": "请向客户明确说明。",
          "severity": "low"
        }
      ],
      "point_time_markers": [
        {
          "point": "风险告知义务",
          "keyword": "风险告知",
          "source": "理财销售必传要点",
          "start_time": 23.57,
          "end_time": 25.84,
          "text": "关于业绩提醒义务。"
        }
      ],
      "matched_keywords": ["风险"],
      "suggestions": ["以下必传要点未覆盖：投资者风险测评, 风险等级匹配"]
    },
    "speaker_segments": [
      { "speaker": "SPEAKER_00", "start": 0.0, "end": 6.09 },
      { "speaker": "SPEAKER_01", "start": 6.79, "end": 10.91 }
    ],
    "duration": 47.72,
    "language": "zh"
  }
}
```

**字段说明**

| 字段 | 说明 |
| --- | --- |
| `text` | 全文转写（已恢复标点） |
| `transcriptions` | 分句结果，含说话人与起止时间；未开启说话人分离时 `speaker` 统一为 `SPEAKER_00` |
| `segments` | 与 `transcriptions` 对应的分句时间轴（无说话人字段） |
| `compliance_report` | 合规比对结果；未开启合规检查或知识库为空时为 `null` |
| `risk_time_markers` | 风险内容出现的**音视频时间节点**，用于回溯核查 |
| `point_time_markers` | 必传要点被覆盖的时间节点 |
| `action_items` / `decisions` | 待办事项 / 决议结论 |

**常见错误响应**

```json
{ "code": 400, "message": "请提供音频文件或Base64编码的音频数据" }
{ "code": 400, "message": "不支持的音频格式，支持：aac, flac, m4a, mp3, mp4, ogg, wav, webm" }
{ "code": 400, "message": "audio_base64 不是有效的 Base64 数据" }
{ "code": 413, "message": "上传文件过大，最大允许 256MB" }
{ "code": 500, "message": "转写失败，请查看服务器日志" }
```

---

## 3. 会议管理

### 3.1 获取会议列表

```
GET /api/meetings
```

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `page` | Integer | 否 | 1 | 页码 |
| `page_size` | Integer | 否 | 10 | 每页数量（上限 100） |
| `status` | String | 否 | - | `processing` / `finished` / `failed` |
| `min_score` | Float | 否 | - | 合规评分下限 |
| `max_score` | Float | 否 | - | 合规评分上限 |
| `start_date` | String | 否 | - | 起始日期 `YYYY-MM-DD` |
| `end_date` | String | 否 | - | 结束日期 `YYYY-MM-DD`（含当天） |
| `include_compliance` | Boolean | 否 | `false` | 是否附带合规摘要 `compliance_summary`（报表页用） |

```json
{
  "code": 200,
  "data": [
    {
      "id": 85, "title": "未命名会议", "date": "2026-10-08T17:20:00",
      "duration": 70, "status": "finished", "audio_path": "uploads/xxx_original.m4a",
      "total_score": 71.55, "score_level": "合格",
      "summary": "……", "keywords": [], "topics": [],
      "sentiment": {}, "action_items": [], "decisions": [],
      "audio_quality": { "noise_reduction": 44.5, "snr_before": 34.4,
                         "snr_after": 67.2, "improvement": 32.8 },
      "compliance_summary": {
        "missing_points_count": 4, "risk_keywords_count": 1,
        "suggestions_count": 2, "first_suggestion": "会议内容与政策要求差距较大，建议重新传达"
      },
      "created_at": "2026-10-08T17:20:00"
    }
  ],
  "total": 70, "page": 1, "page_size": 10
}
```

`audio_quality` 与 `compliance_summary` 均可能为 `null`
（未做预处理 / 该会议没有合规报告）。`compliance_summary` 仅在
`include_compliance=true` 时出现，一次查询带出整页的合规摘要，避免前端逐条请求。

### 3.2 上传音频并分析

```
POST /api/meetings
Content-Type: multipart/form-data
```

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `audio_file` | File | 是 | - | 音频文件（表单字段名固定为 `audio_file`） |
| `meeting_title` | String | 否 | 未命名会议 | 会议标题 |
| `enable_diarization` | Boolean | 否 | `true` | 是否做说话人分离 |
| `enable_compliance` | Boolean | 否 | `true` | 是否做合规比对 |

```json
{ "code": 200, "message": "分析已开始", "meeting_id": 86 }
```

分析在后台异步执行，用 3.3 查进度，或用 Socket.IO 的 `analysis_progress` 事件。

### 3.3 查询分析进度

```
GET /api/meetings/{meeting_id}/progress
```

```json
{ "code": 200, "data": { "progress": 55, "message": "正在提取关键词..." } }
```

`progress` 为 `-1` 表示分析失败，`message` 含失败原因。

### 3.4 获取会议详情

```
GET /api/meetings/{meeting_id}
```

```json
{
  "code": 200,
  "data": {
    "id": 86, "title": "未命名会议", "duration": 44,
    "status": "finished", "total_score": 66.47, "score_level": "合格",
    "summary": "……", "keywords": [], "topics": [], "sentiment": {},
    "action_items": ["必须在本周五之前提交整改报告"],
    "decisions": ["会议决定下周组织一次全员合规培训"],
    "transcriptions": [
      { "id": 1, "speaker": "SPEAKER_00", "text": "……",
        "start_time": 0.0, "end_time": 2.06, "confidence": 1.0, "language": "zh" }
    ],
    "compliance_report": { "total_score": 66.47, "score_level": "合格", "detailed_scores": {}, "……": "……" }
  }
}
```

### 3.5 更新会议信息

```
PUT /api/meetings/{meeting_id}
Content-Type: application/json
```

可传 `title`、`summary`、`total_score`。请求体不是 JSON 对象时返回 400。

### 3.6 删除会议

```
DELETE /api/meetings/{meeting_id}
```

删除会议记录、转写、合规报告，并清理上传目录中的原始与预处理音频文件。

### 3.7 参会人数与发言分布

```
GET /api/meetings/{meeting_id}/participants
```

```json
{
  "code": 200,
  "data": {
    "participant_count": 2,
    "distribution": {
      "SPEAKER_00": { "duration": 24.1, "count": 9, "percentage": 54.8, "segments": [] },
      "SPEAKER_01": { "duration": 19.9, "count": 9, "percentage": 45.2, "segments": [] }
    }
  }
}
```

---

## 4. 实时转写

内置 Web 页面使用 **HTTP 分块上传 + 轮询**，不依赖 socket.io 的浏览器客户端
（客户端脚本只能从 CDN 加载，离线环境会加载失败；服务端仍保留 Socket.IO 事件，
见 4.3，供已有第三方客户端使用）。

### 4.1 HTTP 接口（页面使用，推荐）

音频要求：**16bit 单声道 PCM**，Base64 编码。采样率由调用方在 `start` 时上报，
后端内部重采样到 16kHz 识别，因此 44.1k/48k 采集也能正确工作。

**开始会话**

```
POST /api/realtime/start
Content-Type: application/json

{ "language": "zh", "sample_rate": 16000, "meeting_title": "实时会议",
  "enable_compliance": true, "hotwords": ["风险等级", "录音录像"] }
```

```json
{ "code": 200, "message": "会话已开始",
  "data": { "session_id": "…", "language": "zh", "sample_rate": 16000,
            "meeting_title": "实时会议", "compliance_enabled": true } }
```

**推送音频块**

```
POST /api/realtime/chunk
Content-Type: application/json

{ "session_id": "…", "audio": "<base64 PCM>" }
```

```json
{ "code": 200, "data": { "text": "各位同事，大家好。",
    "segments": [{ "text": "各位同事，大家好。", "start": 0.0, "end": 2.1, "confidence": 1.0 }],
    "compliance": [ { "has_risk": true, "risk_keywords": ["保本保收益"],
                      "risk_items": [ { "keyword": "保本保收益", "severity": "medium",
                                        "start_time": 12.4, "end_time": 14.8, "text": "…" } ],
                      "covered_points": [ { "point": "风险告知义务", "keyword": "风险告知",
                                            "start_time": 23.5, "end_time": 25.8 } ],
                      "start_time": 12.4, "end_time": 14.8, "alerts": [] } ],
    "is_final": false } }
```

建议每 0.3~1 秒推送一块（页面实现为每 500ms 汇总一次）。
`data` 可能为 `null`（这一段还没有识别结果）。
会话不存在时返回 404。

**停止并转分析**

```
POST /api/realtime/stop
{ "session_id": "…" }
```

```json
{ "code": 200, "message": "已停止，正在分析",
  "data": { "audio_file": "uploads/…_realtime.wav", "audio_duration": 44.02,
            "text": "…", "transcriptions": [ { "speaker": "SPEAKER_00", "text": "…",
            "start_time": 0.0, "end_time": 2.06 } ], "meeting_title": "实时会议" } }
```

**轮询结果**

```
GET /api/realtime/result/{session_id}
```

```json
{ "code": 200, "data": {
    "status": "recording | analyzing | done | failed",
    "progress": 100, "message": "分析完成", "meeting_id": 86,
    "result": { "text": "…", "transcriptions": [], "summary": "…", "keywords": [],
                "topics": [], "sentiment": {}, "action_items": [], "decisions": [],
                "compliance_report": {}, "audio_quality": {} } } }
```

- 全程没有有效语音时 `status` 为 `failed`，`meeting_id` 为 `null`，不会产生空会议记录
- 结果保留 1 小时，过期后返回 404
- `result` 仅在 `status=done` 时非空

**放弃会话（页面关闭时调用）**

```
POST /api/realtime/discard
{ "session_id": "…" }
```

### 4.2 音频块事件（`transcription_result.compliance`）字段

| 字段 | 说明 |
| --- | --- |
| `has_risk` | 本段是否命中风险词 |
| `risk_keywords` | 命中的风险词名（便于直接展示） |
| `risk_items` | 明细：`keyword` / `severity` / `start_time` / `end_time` / `text` |
| `covered_points` | 本段覆盖到的必传要点：`point` / `keyword` / `start_time` / `end_time` |
| `start_time` / `end_time` | 本段文本的时间范围 |

### 4.3 Socket.IO 事件（兼容保留）

| 事件 | 方向 | 载荷 |
| --- | --- | --- |
| `start_transcription` | 客户端 → 服务端 | `{language, enable_compliance, meeting_title, hotwords}` |
| `transcription_started` | 服务端 → 客户端 | `{status, language, compliance_enabled, meeting_title}` |
| `audio_chunk` | 客户端 → 服务端 | `{audio: "<base64>"}` |
| `transcription_result` | 服务端 → 客户端 | `{text, segments, compliance, is_final}` |
| `stop_transcription` | 客户端 → 服务端 | - |
| `transcription_stopped` | 服务端 → 客户端 | `{audio_file, audio_duration, text}` |
| `realtime_analysis_progress` | 服务端 → 客户端 | `{progress, message}`（`-1` 为失败） |
| `transcription_final` | 服务端 → 客户端 | 完整分析结果 + `meeting_id` |
| `error` | 服务端 → 客户端 | `{message}` |

> 注意：Socket.IO 需要客户端库，离线环境下请使用上面的 HTTP 接口。

---

## 5. 知识库管理

### 5.1 获取列表

```
GET /api/knowledge-base?page=1&page_size=10&item_type=policy
```

`item_type` 可选 `policy`（政策）、`meeting_spirit`（会议精神）、`key_points` / `required`（必传要点）、`risk_keywords` / `forbidden`（风险词）。

```json
{
  "code": 200,
  "data": [
    { "id": 3, "title": "理财销售必传要点", "content": "……", "item_type": "key_points",
      "keywords": ["风险测评", "适当性"], "required_points": ["投资者风险测评"],
      "status": "active", "created_at": "2026-07-09T16:00:00", "updated_at": null }
  ],
  "total": 8
}
```

### 5.2 获取单条

```
GET /api/knowledge-base/{item_id}
```

### 5.3 新增条目

```
POST /api/knowledge-base
Content-Type: application/json
```

| 参数名 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `title` | String | 是 | 标题 |
| `content` | String | 是 | 内容 |
| `item_type` | String | 否 | 默认 `policy` |
| `keywords` | Array | 否 | 关键词列表 |
| `required_points` | Array | 否 | 必传要点列表 |

### 5.4 上传政策文件自动解析

```
POST /api/knowledge-base
Content-Type: multipart/form-data
```

| 参数名 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `file` | File | 是 | 支持 `.txt` `.pdf` `.docx` `.doc` |
| `item_type` | String | 否 | 默认 `policy` |

```json
{
  "code": 200, "message": "上传成功",
  "data": { "id": 9, "title": "……" },
  "parsed": { "title": "……", "keywords": [], "required_points": [], "content_length": 1200 }
}
```

### 5.5 更新 / 删除 / 搜索

```
PUT    /api/knowledge-base/{item_id}     # 可改 title/content/item_type/keywords/required_points/status
DELETE /api/knowledge-base/{item_id}
GET    /api/knowledge-base/search?q=风险
```

---

## 6. 合规检查与自测

### 6.1 获取会议合规报告

```
GET /api/meetings/{meeting_id}/compliance
```

```json
{
  "code": 200,
  "data": {
    "id": 12, "meeting_id": 86, "total_score": 66.47, "score_level": "合格",
    "detailed_scores": { "semantic_similarity": 21.2, "point_coverage": 12.72,
                         "risk_detection": 14.0, "keyword_matching": 0.0 },
    "missing_points": ["投资者风险测评", "风险等级匹配"],
    "risk_keywords": ["保本保收益"],
    "risk_time_markers": [], "point_time_markers": [],
    "matched_keywords": [], "suggestions": [],
    "created_at": "2026-10-08T18:00:00"
  }
}
```

不存在时返回 404。

### 6.2 文本合规自测（调试知识库配置用）

```
POST /api/meetings/test-analyze
Content-Type: application/json
```

```json
{ "text": "本产品保本保收益，绝对安全。" }
```

可选传 `score_weights` 覆盖本次权重。返回与 `compliance_report` 相同的字段。

### 6.3 文本摘要自测

```
POST /api/meetings/test-summary
Content-Type: application/json
```

```json
{ "text": "会议决定下周开展合规培训，请各部门落实整改。" }
```

```json
{
  "code": 200,
  "data": {
    "summary": "……",
    "keywords": [], "topics": [], "sentiment": {},
    "action_items": ["请各部门落实整改"],
    "decisions": ["会议决定下周开展合规培训"]
  }
}
```

---

## 7. 评分权重

### 7.1 获取权重

```
GET /api/score-weights
```

```json
{
  "code": 200,
  "data": [
    { "weight_name": "semantic_similarity", "weight_value": 40, "description": null },
    { "weight_name": "point_coverage", "weight_value": 30, "description": null },
    { "weight_name": "risk_detection", "weight_value": 20, "description": null },
    { "weight_name": "keyword_matching", "weight_value": 10, "description": null }
  ]
}
```

### 7.2 更新权重

```
PUT /api/score-weights
Content-Type: application/json
```

```json
{ "semantic_similarity": 40, "point_coverage": 30, "risk_detection": 20, "keyword_matching": 10 }
```

未提供的项保持原值；不存在的项会被创建。

---

## 8. 系统信息

| 接口 | 说明 |
| --- | --- |
| `GET /api/languages` | 支持的语言（返回 `code -> 名称` 映射） |
| `GET /api/topics` | 主题候选列表 |
| `GET /api/risk-keywords` | 默认风险关键词列表 |
| `GET /api/hardware/status` | 麦克风/扬声器/摄像头检测（无设备或未安装 OpenCV 时字段会标记为不可用） |
| `GET /api/meeting-status` | 采集 5 秒音频，返回当前环境音量（dB）与是否检测到语音 |

---

## 9. 报告生成

### 9.1 会议纪要报告

```
GET /api/reports/meeting-summary/{meeting_id}
```

返回 `text/html`：会议基本信息、参会人数、摘要、关键词、主题、待办事项、决议结论、
合规评分、风险关键词与遗漏要点、改进建议。

### 9.2 合规趋势报告

```
GET /api/reports/compliance-trend
```

返回 `text/html`：会议场次、平均分、分数分布（优秀/良好/合格/不合格）、
风险项统计与改进建议。无已完成的会议时返回 404。

---

## 错误码

| 错误码 | HTTP | 说明 |
| --- | --- | --- |
| 200 | 200 | 成功 |
| 400 | 400 | 请求参数错误（缺参数、格式不支持、Base64 非法、日期格式错误） |
| 404 | 404 | 资源不存在（会议/知识库条目/合规报告） |
| 413 | 413 | 上传文件超过 `MAX_CONTENT_LENGTH`（默认 256MB） |
| 500 | 500 | 服务内部错误，详见服务器日志 |

---

## 注意事项

1. **音频限制**：单文件最大 256MB；小于 1KB 的文件会被拒绝。
2. **处理耗时**：CPU 推理，识别实时率约 0.1（10 秒音频约 1 秒）；说话人分离与摘要生成会额外增加时间。
3. **模型离线加载**：所有模型从项目 `models/` 目录本地加载，**无需联网**，也不会自动下载；
   模型缺失时识别自动降级到 Whisper，其它分析降级到规则方法。
4. **合规比对依赖知识库**：知识库中没有启用条目时 `compliance_report` 为 `null`。
5. **时间节点可用于回溯**：`risk_time_markers` / `point_time_markers` 中的 `start_time`、`end_time`
   可直接用于定位音视频片段。
