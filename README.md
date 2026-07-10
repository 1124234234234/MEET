# AI 语音识别与翻译系统

基于 OpenAI Whisper 的语音识别 API 服务，支持自动语言检测，并可将识别结果翻译成目标语言。

## 功能特性

- 自动检测音频语言（支持中、英、日、韩、法、德、西、俄、阿、葡等 10+ 种语言）
- 自动翻译识别结果（内置中↔英翻译）
- RESTful API，方便集成到其他系统
- 内置 Web Demo 页面，上传音频即可试用
- 支持 Docker 一键部署

## 项目结构

```
voice-reco/
├── Dockerfile          # Docker 镜像构建文件
├── app.py              # Flask API 服务主程序
├── index.html          # Web Demo 页面
├── requirements.txt    # Python 依赖
├── start.sh            # Linux/Mac 启动脚本
└── start.bat           # Windows 启动脚本
```

## Docker 部署

### Windows 系统

1. 确保已安装 Docker Desktop
2. 双击运行 `start.bat` 脚本

### Linux/Mac 系统

1. 确保已安装 Docker
2. 运行 `chmod +x start.sh && ./start.sh`

### 手动部署

```bash
# 构建镜像
docker build -t voice-recognition:latest .

# 启动容器
docker run -d --name voice-api -p 5000:5000 --memory=6g voice-recognition:latest

# 查看日志
docker logs voice-api

# 健康检查
curl http://localhost:5000/health
```

## 使用 Web Demo

API 启动后，直接用浏览器打开 http://localhost:5000 即可上传音频进行测试。

## API 说明

### 健康检查
```
GET /health
```

### 语音识别
```
POST /transcribe
Content-Type: multipart/form-data

参数：
- audio: 音频文件（必填）
- target_language: 目标语言代码（可选）

响应：
{
  "transcribed_text": "转录文本",
  "detected_language": "zh",
  "language_name": "Chinese",
  "translated_text": "翻译文本（如有）
}
```

### 支持的语言
```
GET /languages
```

## 配置

Whisper 模型大小可在 app.py 中修改（tiny / base / small / medium / large-v3-turbo），模型越大准确度越高，但耗时和内存占用也越大。
