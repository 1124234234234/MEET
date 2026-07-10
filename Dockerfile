FROM python:3.10-slim
WORKDIR /app
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y ffmpeg git \
    && rm -rf /var/lib/apt/lists/*
RUN pip install openai-whisper flask flask-cors transformers torch -i https://pypi.tuna.tsinghua.edu.cn/simple
COPY app.py .
COPY index.html .
EXPOSE 5000
CMD ["python", "app.py"]
