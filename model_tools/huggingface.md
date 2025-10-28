# Download from huggingface

ref link: https://blog.frognew.com/2024/06/using-huggingface-cli-to-download-models.html

install huggingface-cli

```
pip install 'huggingface_hub[cli]' --index-url=https://mirrors.aliyun.com/pypi/simple
# 设置国内的镜像站
export HF_ENDPOINT=https://hf-mirror.com
# 下载模型到指定目录
huggingface-cli download Qwen/Qwen2-7B-Instruct --local-dir ~/models/models--Qwen--Qwen2-7B-Instruct
```

