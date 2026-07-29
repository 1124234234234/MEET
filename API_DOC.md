# 会议室智能合规分析系统 - API接口文档

## 基础信息

- **服务地址**: `http://localhost:5000`
- **版本**: v1.0.0
- **编码**: UTF-8
- **跨域**: 已启用CORS，支持所有域名

---

## 目录

1. [健康检查](#1-健康检查)
2. [语音转写分析](#2-语音转写分析)
3. [会议管理](#3-会议管理)
4. [知识库管理](#4-知识库管理)
5. [合规检查](#5-合规检查)
6. [评分权重](#6-评分权重)
7. [系统配置](#7-系统配置)
8. [报告生成](#8-报告生成)

---

## 1. 健康检查

### 1.1 检查服务状态

**请求**:
```
GET /api/v1/health
```

**响应**:
```json
{
    "code": 200,
    "message": "服务正常运行",
    "data": {
        "timestamp": "2026-07-23T10:00:00",
        "version": "1.0.0"
    }
}
```

---

## 2. 语音转写分析

### 2.1 音频转写与分析（核心接口）

**请求**:
```
POST /api/v1/transcribe
```

**Content-Type**: `multipart/form-data` 或 `application/json`

**参数**:

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| audio | File | 二选一 | - | 音频文件，支持 mp3, wav, m4a, ogg, flac |
| audio_base64 | String | 二选一 | - | Base64编码的音频数据 |
| language | String | 否 | zh | 语言，支持 zh（中文）、en（英文） |
| enable_compliance | Boolean | 否 | true | 是否进行合规检查 |
| enable_diarization | Boolean | 否 | true | 是否进行说话人分离 |

**示例（curl）**:
```bash
# 文件上传方式
curl -X POST http://localhost:5000/api/v1/transcribe \
  -F "audio=@meeting.mp3" \
  -F "language=zh" \
  -F "enable_compliance=true"

# Base64方式
curl -X POST http://localhost:5000/api/v1/transcribe \
  -H "Content-Type: application/json" \
  -d '{
    "audio_base64": "base64_encoded_audio_data",
    "language": "zh",
    "enable_compliance": true
  }'
```

**示例（Python）**:
```python
import requests

# 文件上传
response = requests.post(
    'http://localhost:5000/api/v1/transcribe',
    files={'audio': open('meeting.mp3', 'rb')},
    data={'language': 'zh', 'enable_compliance': 'true'}
)
print(response.json())
```

**成功响应** (code=200):
```json
{
    "code": 200,
    "message": "成功",
    "data": {
        "text": "会议完整转写文本内容...",
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 5.2,
                "text": "第一段转写内容",
                "speaker": "Speaker 1"
            }
        ],
        "keywords": ["关键词1", "关键词2", "关键词3"],
        "topics": [
            {"name": "工作汇报", "score": 0.85},
            {"name": "项目讨论", "score": 0.32}
        ],
        "summary": "会议摘要内容...",
        "sentiment": {
            "positive": 0.75,
            "neutral": 0.20,
            "negative": 0.05,
            "overall": "积极"
        },
        "compliance_report": {
            "score": 92.5,
            "level": "优秀",
            "missing_points": [],
            "risk_content": [],
            "suggestions": []
        },
        "speaker_segments": [
            {
                "speaker": "Speaker 1",
                "start": 0.0,
                "end": 10.5,
                "text": "说话人1的发言内容"
            }
        ]
    }
}
```

**失败响应** (code=400):
```json
{
    "code": 400,
    "message": "请提供音频文件或Base64编码的音频数据"
}
```

---

## 3. 会议管理

### 3.1 获取会议列表

**请求**:
```
GET /api/meetings
```

**参数**:

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| page | Integer | 否 | 1 | 页码 |
| per_page | Integer | 否 | 20 | 每页数量 |

**响应**:
```json
{
    "code": 200,
    "data": {
        "meetings": [
            {
                "id": 1,
                "title": "未命名会议",
                "date": "2026-07-23T10:00:00",
                "duration": 300,
                "status": "completed",
                "compliance_score": 92.5,
                "transcription_count": 15
            }
        ],
        "total": 100,
        "page": 1,
        "per_page": 20
    }
}
```

### 3.2 获取单个会议详情

**请求**:
```
GET /api/meetings/{meeting_id}
```

**响应**:
```json
{
    "code": 200,
    "data": {
        "id": 1,
        "title": "未命名会议",
        "date": "2026-07-23T10:00:00",
        "duration": 300,
        "status": "completed",
        "compliance_score": 92.5,
        "transcriptions": [...],
        "summary": "会议摘要",
        "keywords": [...],
        "topics": [...],
        "sentiment": {...},
        "compliance_report": {...}
    }
}
```

### 3.3 创建会议（上传分析）

**请求**:
```
POST /api/meetings
```

**Content-Type**: `multipart/form-data`

**参数**:

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| audio | File | 是 | - | 音频文件 |
| title | String | 否 | 未命名会议 | 会议标题 |
| language | String | 否 | zh | 语言 |
| enable_diarization | Boolean | 否 | true | 启用说话人分离 |
| enable_compliance | Boolean | 否 | true | 启用合规检查 |

**响应**:
```json
{
    "code": 200,
    "message": "会议分析任务已创建",
    "data": {
        "meeting_id": 1,
        "status": "processing"
    }
}
```

### 3.4 更新会议信息

**请求**:
```
PUT /api/meetings/{meeting_id}
```

**Content-Type**: `application/json`

**参数**:

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| title | String | 否 | 会议标题 |

**响应**:
```json
{
    "code": 200,
    "message": "会议信息更新成功"
}
```

### 3.5 删除会议

**请求**:
```
DELETE /api/meetings/{meeting_id}
```

**响应**:
```json
{
    "code": 200,
    "message": "会议删除成功"
}
```

### 3.6 获取会议分析进度

**请求**:
```
GET /api/meetings/{meeting_id}/progress
```

**响应**:
```json
{
    "code": 200,
    "data": {
        "meeting_id": 1,
        "progress": 50,
        "status": "processing",
        "message": "正在进行说话人分离..."
    }
}
```

---

## 4. 知识库管理

### 4.1 获取知识库列表

**请求**:
```
GET /api/knowledge-base
```

**响应**:
```json
{
    "code": 200,
    "data": [
        {
            "id": 1,
            "title": "合规要点1",
            "content": "合规要点详细内容...",
            "keywords": ["关键词1", "关键词2"],
            "status": "active",
            "created_at": "2026-07-23T10:00:00"
        }
    ]
}
```

### 4.2 获取单个知识库条目

**请求**:
```
GET /api/knowledge-base/{item_id}
```

**响应**:
```json
{
    "code": 200,
    "data": {
        "id": 1,
        "title": "合规要点1",
        "content": "合规要点详细内容...",
        "keywords": ["关键词1", "关键词2"],
        "status": "active",
        "created_at": "2026-07-23T10:00:00"
    }
}
```

### 4.3 创建知识库条目

**请求**:
```
POST /api/knowledge-base
```

**Content-Type**: `application/json`

**参数**:

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| title | String | 是 | 条目标题 |
| content | String | 是 | 条目内容 |
| keywords | Array | 否 | 关键词列表 |

**响应**:
```json
{
    "code": 200,
    "message": "知识库条目创建成功",
    "data": {
        "id": 1
    }
}
```

### 4.4 更新知识库条目

**请求**:
```
PUT /api/knowledge-base/{item_id}
```

**Content-Type**: `application/json`

**参数**:

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| title | String | 否 | 条目标题 |
| content | String | 否 | 条目内容 |
| keywords | Array | 否 | 关键词列表 |
| status | String | 否 | 状态（active/inactive） |

**响应**:
```json
{
    "code": 200,
    "message": "知识库条目更新成功"
}
```

### 4.5 删除知识库条目

**请求**:
```
DELETE /api/knowledge-base/{item_id}
```

**响应**:
```json
{
    "code": 200,
    "message": "知识库条目删除成功"
}
```

### 4.6 搜索知识库

**请求**:
```
GET /api/knowledge-base/search?q={关键词}
```

**响应**:
```json
{
    "code": 200,
    "data": [
        {
            "id": 1,
            "title": "合规要点1",
            "content": "合规要点详细内容...",
            "similarity": 0.85
        }
    ]
}
```

---

## 5. 合规检查

### 5.1 获取会议合规报告

**请求**:
```
GET /api/meetings/{meeting_id}/compliance
```

**响应**:
```json
{
    "code": 200,
    "data": {
        "meeting_id": 1,
        "score": 92.5,
        "level": "优秀",
        "missing_points": [],
        "risk_content": [],
        "suggestions": [],
        "analysis_details": {
            "semantic_similarity": 95,
            "point_coverage": 90,
            "risk_detection": 100,
            "keyword_matching": 88
        }
    }
}
```

---

## 6. 评分权重

### 6.1 获取评分权重配置

**请求**:
```
GET /api/score-weights
```

**响应**:
```json
{
    "code": 200,
    "data": {
        "semantic_similarity": 40,
        "point_coverage": 30,
        "risk_detection": 20,
        "keyword_matching": 10
    }
}
```

### 6.2 更新评分权重配置

**请求**:
```
PUT /api/score-weights
```

**Content-Type**: `application/json`

**参数**:

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| semantic_similarity | Integer | 否 | 语义相似度权重（0-100） |
| point_coverage | Integer | 否 | 要点覆盖权重（0-100） |
| risk_detection | Integer | 否 | 风险检测权重（0-100） |
| keyword_matching | Integer | 否 | 关键词匹配权重（0-100） |

**响应**:
```json
{
    "code": 200,
    "message": "评分权重更新成功"
}
```

---

## 7. 系统配置

### 7.1 获取支持的语言列表

**请求**:
```
GET /api/languages
```

**响应**:
```json
{
    "code": 200,
    "data": [
        {"code": "zh", "name": "中文"},
        {"code": "en", "name": "英文"}
    ]
}
```

### 7.2 获取主题列表

**请求**:
```
GET /api/topics
```

**响应**:
```json
{
    "code": 200,
    "data": [
        "工作汇报", "项目讨论", "问题解决", "决策制定",
        "进度跟进", "计划安排", "意见交流", "培训学习"
    ]
}
```

### 7.3 获取风险关键词

**请求**:
```
GET /api/risk-keywords
```

**响应**:
```json
{
    "code": 200,
    "data": ["消极", "反对", "抵制", "抱怨", "不满", "拒绝", "不行", "不可能", "做不到"]
}
```

### 7.4 获取硬件状态

**请求**:
```
GET /api/hardware/status
```

**响应**:
```json
{
    "code": 200,
    "data": {
        "cpu_usage": 35,
        "memory_usage": 45,
        "disk_usage": 60
    }
}
```

---

## 8. 报告生成

### 8.1 生成会议摘要报告

**请求**:
```
GET /api/reports/meeting-summary/{meeting_id}
```

**响应**: 返回HTML格式的会议摘要报告

### 8.2 生成合规趋势报告

**请求**:
```
GET /api/reports/compliance-trend
```

**参数**:

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| start_date | String | 否 | - | 开始日期（YYYY-MM-DD） |
| end_date | String | 否 | - | 结束日期（YYYY-MM-DD） |

**响应**: 返回HTML格式的合规趋势报告

---

## 错误码说明

| 错误码 | 说明 |
|--------|------|
| 200 | 成功 |
| 400 | 请求参数错误 |
| 404 | 资源不存在 |
| 500 | 服务器内部错误 |

---

## 注意事项

1. **音频文件限制**: 最大支持256MB
2. **分析时间**: 5分钟音频分析约需5-10分钟（首次运行需下载模型）
3. **模型下载**: 首次运行会自动下载所需模型，需保持网络畅通
4. **文件格式**: 支持 mp3, wav, m4a, ogg, flac 格式