FROM python:3.10-slim

WORKDIR /app

# 设置 pip 重试和超时
ENV PIP_DEFAULT_TIMEOUT=100
ENV PIP_RETRIES=3

# 先 COPY requirements.txt 再安装
COPY requirements.txt .

# 升级 pip
RUN pip install --no-cache-dir --upgrade pip

# 安装 PyTorch GPU 包（如果你用 GPU）
RUN pip install torch==2.9.1+cu130 torchvision==0.15.2+cu130 torchaudio==2.0.2+cu130 \
    -f https://download.pytorch.org/whl/torch_stable.html

# 安装其他依赖
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple/ \
    --timeout 300

# COPY 项目代码
COPY . .

# >docker build -t timer_xudianchi:v1.0 .

# 默认运行命令
# CMD ["python", "your_main_script.py"]
