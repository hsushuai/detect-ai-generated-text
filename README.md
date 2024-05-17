## 介绍

🕵️‍♂️ 微调 DeBERTa-v3 进行 AI 生成文章检测

任务来自 Kaggle 竞赛 **LLM - Detect AI Generated Text** [[Link]](https://www.kaggle.com/competitions/llm-detect-ai-generated-text)

数据集使用



## Section 1: 准备

### 1.1 实验环境

- Pytorch 2.1.2
- Python 3.10
- CUDA 12.1
- Ubuntu 22.04

GPU 使用 Nvidia A10 * 1（资源有限，项目支持多 GPU 训练）

### 1.3 依赖

克隆仓库并安装 requirements

```
git clone https://github.com/rbiswasfc/llm-detect-ai.git
cd llm-detect-ai
pip install -r requirements.txt
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