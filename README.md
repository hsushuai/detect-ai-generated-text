## 介绍

🕵️‍♂️ 微调 DeBERTa-v3 进行 AI 生成文章检测

任务来自 Kaggle 竞赛 **LLM - Detect AI Generated Text** [[Link]](https://www.kaggle.com/competitions/llm-detect-ai-generated-text)

数据集使用 “🔍📝🕵️🤖” 队公开的混合数据 [link](https://www.kaggle.com/datasets/conjuring92/ai-bin7-mix-v1)

## Section 1: 准备

### 1.1 实验环境

- Pytorch 2.1.2
- Python 3.10
- CUDA 12.1
- Ubuntu 22.04

GPU 使用 Nvidia A10 * 1（资源有限，项目支持多 GPU 训练）

### 1.2 依赖

克隆仓库并安装 requirements

```
git clone https://github.com/rbiswasfc/llm-detect-ai.git
cd llm-detect-ai
pip install -r requirements.txt
```

### 1.3 数据集和模型

下载数据集，从头开始微调模型 🥊

```bash
mkdir data

wget -P data https://github.com/hsushuai/detect-ai-generated-text/releases/download/dataset/detect-ai-generated-text-mix.zip
wget -P data  https://github.com/hsushuai/detect-ai-generated-text/releases/download/dataset/llm-detect-ai-generated-text.zip

unzip data/detect-ai-generated-text-mix.zip -d data
unzip data/llm-detect-ai-generated-text.zip -d data
```

下载微调好的模型，直接用于部署推理 🚀

```bash
mkdir models

wget -P models https://github.com/hsushuai/detect-ai-generated-text/releases/download/models/finetuned-deberta-v3-small-best.pth.tar
wget -P models https://github.com/hsushuai/detect-ai-generated-text/releases/download/models/finetuned-deberta-v3-small-last.pth.tar
```

## Section 2: 训练

### 2.1 （可选）修改 Hugging Face 源

如果你的网络环境无法直接连接到抱抱脸 🫣 官方地址，可以修改 Hugging Face 源

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

训练代码和配置分别在 `code` 和 `conf` 文件夹中。 我们使用 HF `accelerate` 来进行训练。多 GPU 时，采用 DDP 策略。


### 2.2 微调 DeBERTa

可以修改 `--config-name deberta-v3-large` 去微调 deberta-v3-large 模型

```bash
accelerate launch ./src/run_train.py \
--config-name deberta-v3-small \
use_wandb=false
```

可以设置 `use_wandb=true` 来使用 wandb 记录训练过程，前提是需要通过 `wandb login` 来配置个人密钥 🗝️ ，详情见 [wandb 官方文档](https://docs.wandb.ai/)。

## Section 3：部署

使用 Flask 实现简易的 web 服务 🤖

```bash
python deployment/app.py
```

### 🐳 Docker 部署

使用 Dockerfile 构建镜像，并运行容器

```bash
docker build -t detect_ai_generated_text .
docker run -d -p 5000:5000 detect_ai_generated_text
```