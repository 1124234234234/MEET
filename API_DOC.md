# 语音转写分析系统 API 接口文档

## 概述

本系统提供语音转写和会议分析的 RESTful API 接口，供第三方软件调用。第三方软件可将音频数据发送到接口，系统完成语音转写、说话人分离、合规检查等处理后返回结果。

## 基础信息

- **服务地址**: `http://localhost:5000`
- **API版本**: v1
- **Content-Type**: `multipart/form-data` 或 `application/json`
- **字符编码**: UTF-8

---

## 接口列表

| 接口 | 方法 | 描述 |
|------|------|------|
| `/api/v1/transcribe` | POST | 语音转写与分析（核心接口） |
| `/api/v1/health` | GET | 健康检查 |

---

## 1. 语音转写与分析接口

### 请求

**URL**: `POST /api/v1/transcribe`

**方式一：multipart/form-data（推荐上传文件）**

```bash
curl -X POST http://localhost:5000/api/v1/transcribe \
  -F "audio=@meeting.mp3" \
  -F "language=zh" \
  -F "enable_compliance=true" \
  -F "enable_diarization=true"
```

**方式二：application/json（Base64编码）**

```bash
curl -X POST http://localhost:5000/api/v1/transcribe \
  -H "Content-Type: application/json" \
  -d '{
    "audio_base64": "base64_encoded_audio_data",
    "language": "zh",
    "enable_compliance": true,
    "enable_diarization": true
  }'
```

### 请求参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| audio | File | 否（与audio_base64二选一） | - | 音频文件，支持 mp3、wav、m4a 格式 |
| audio_base64 | String | 否（与audio二选一） | - | Base64编码的音频数据 |
| language | String | 否 | zh | 语言，支持 `zh`（中文）、`en`（英文） |
| enable_compliance | Boolean | 否 | true | 是否进行合规检查 |
| enable_diarization | Boolean | 否 | true | 是否进行说话人分离 |

### 成功响应

**HTTP状态码**: 200

```json
{
    "code": 200,
    "message": "成功",
    "data": {
        "text": "各位同事大家好，今天我们召开理财产品销售合规培训会议...",
        "transcriptions": [
            {
                "speaker": "SPEAKER_00",
                "text": "各位同事大家好",
                "start_time": 0.0,
                "end_time": 2.5,
                "confidence": 0.98
            }
        ],
        "keywords": ["合规", "理财产品", "销售", "培训"],
        "topics": [
            {"topic": "合规培训", "weight": 0.85},
            {"topic": "产品销售", "weight": 0.72}
        ],
        "summary": "本次会议主要讨论理财产品销售的合规要求，包括风险告知、投资者适当性管理等内容。",
        "sentiment": {
            "positive": 0.75,
            "negative": 0.05,
            "neutral": 0.20,
            "overall": "positive"
        },
        "compliance_report": {
            "total_score": 92.5,
            "score_level": "优秀",
            "components": {
                "risk_keywords": 100,
                "required_points": 85,
                "sentiment": 95
            },
            "missing_points": [],
            "risk_keywords_found": [],
            "matched_keywords": ["合规", "风险告知"],
            "suggestions": ["会议内容合规，继续保持"]
        },
        "speaker_segments": [
            {
                "speaker": "SPEAKER_00",
                "start": 0.0,
                "end": 15.3,
                "text": "..."
            }
        ]
    }
}
```

### 失败响应

**HTTP状态码**: 400 / 500

```json
{
    "code": 400,
    "message": "请提供音频文件或Base64编码的音频数据"
}
```

```json
{
    "code": 500,
    "message": "转写失败: 具体错误信息"
}
```

---

## 2. 健康检查接口

### 请求

**URL**: `GET /api/v1/health`

```bash
curl http://localhost:5000/api/v1/health
```

### 成功响应

**HTTP状态码**: 200

```json
{
    "code": 200,
    "message": "服务正常运行",
    "data": {
        "timestamp": "2026-07-22T20:30:00",
        "version": "1.0.0"
    }
}
```

---

## 响应字段说明

### data.text

完整的转写文本内容。

### data.transcriptions

转写分段列表，每段包含：

| 字段 | 类型 | 说明 |
|------|------|------|
| speaker | String | 说话人标识（如 SPEAKER_00） |
| text | String | 该段文本内容 |
| start_time | Float | 开始时间（秒） |
| end_time | Float | 结束时间（秒） |
| confidence | Float | 置信度（0-1） |

### data.keywords

提取的关键词列表。

### data.topics

主题分析结果，每项包含：

| 字段 | 类型 | 说明 |
|------|------|------|
| topic | String | 主题名称 |
| weight | Float | 权重（0-1） |

### data.summary

会议摘要文本。

### data.sentiment

情绪分析结果：

| 字段 | 类型 | 说明 |
|------|------|------|
| positive | Float | 正面情绪占比 |
| negative | Float | 负面情绪占比 |
| neutral | Float | 中性情绪占比 |
| overall | String | 总体情绪（positive/negative/neutral） |

### data.compliance_report

合规检查报告：

| 字段 | 类型 | 说明 |
|------|------|------|
| total_score | Float | 综合合规评分（0-100） |
| score_level | String | 评分等级（优秀/合格/不合格） |
| components | Object | 各维度得分 |
| missing_points | Array | 遗漏的合规要点 |
| risk_keywords_found | Array | 发现的风险关键词 |
| matched_keywords | Array | 匹配的合规关键词 |
| suggestions | Array | 改进建议 |

### data.speaker_segments

说话人分离结果：

| 字段 | 类型 | 说明 |
|------|------|------|
| speaker | String | 说话人标识 |
| start | Float | 开始时间（秒） |
| end | Float | 结束时间（秒） |
| text | String | 该说话人所说内容 |

---

## 错误码说明

| 错误码 | 说明 |
|--------|------|
| 200 | 请求成功 |
| 400 | 请求参数错误 |
| 500 | 服务器内部错误 |

---

## 调用示例

### Python 示例

```python
import requests

# 方式一：上传文件
url = "http://localhost:5000/api/v1/transcribe"
files = {"audio": open("meeting.mp3", "rb")}
data = {
    "language": "zh",
    "enable_compliance": "true",
    "enable_diarization": "true"
}
response = requests.post(url, files=files, data=data)
result = response.json()
print(result)

# 方式二：Base64编码
import base64
with open("meeting.mp3", "rb") as f:
    audio_base64 = base64.b64encode(f.read()).decode()

data = {
    "audio_base64": audio_base64,
    "language": "zh",
    "enable_compliance": True,
    "enable_diarization": True
}
response = requests.post(url, json=data)
result = response.json()
print(result)
```

### JavaScript 示例

```javascript
// 方式一：上传文件
const formData = new FormData();
formData.append('audio', fileInput.files[0]);
formData.append('language', 'zh');
formData.append('enable_compliance', 'true');
formData.append('enable_diarization', 'true');

fetch('http://localhost:5000/api/v1/transcribe', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => console.log(data));

// 方式二：Base64编码
fetch('meeting.mp3')
.then(response => response.arrayBuffer())
.then(buffer => {
    const audioBase64 = btoa(String.fromCharCode(...new Uint8Array(buffer)));
    return fetch('http://localhost:5000/api/v1/transcribe', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            audio_base64: audioBase64,
            language: 'zh',
            enable_compliance: true,
            enable_diarization: true
        })
    });
})
.then(response => response.json())
.then(data => console.log(data));
```

---

## 注意事项

1. **音频格式**: 支持 mp3、wav、m4a 格式，推荐使用 wav 格式以获得最佳转写效果
2. **音频采样率**: 建议使用 16kHz 采样率
3. **响应时间**: 5分钟音频的分析时间约为 5-10 分钟（取决于是否启用说话人分离和合规检查）
4. **并发限制**: 建议同时处理的请求不超过 1 个，避免内存和 CPU 资源不足
5. **CORS**: 服务已启用 CORS，支持跨域调用
