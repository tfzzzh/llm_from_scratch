# Large Language Model From Scratch
This repository implement a Llamma2 like Large Language Model (LLM) from scratch using only basic functionality of Python and PyTorch. This project serves as an educational implementation to understand the core components of modern transformer-based language models.

## Implemented Components
+ **BPE Tokenizer** - Byte Pair Encoding tokenizer for text preprocessing - see [`tokenizer.py`](llm/tokenizer.py)
+ **Llama2-like Transformer Model** - Transformer architecture - see [`layers.py`](llm/layers.py)
+ **Flash Attention 2** - Flash Attention via Triton - see [`flash_attention.py`](llm/flash_attention.py)
+ **AdamW Optimizer** - Weight decay regularized Adam optimizer - see [`optimizer.py`](llm/optimizer.py)
+ **Cosine Annealed Learning Rate Scheduler** - Learning rate scheduling - see [`lr_scheduler.py`](llm/lr_scheduler.py)
+ **Training Framework** - Complete training loop with logging and checkpoint management - see [`trainer.py`](llm/trainer.py)
+ **Data Utilities** - DataLoader - see [`utility.py`](llm/utility.py)

## Prerequisites
- Python 3.12
- PyTorch
- All the code is tested in a machine with Ubuntu Operating System and GTX 1080 GPU

## Quick Start

### Installation and Testing
1. **Clone this repository** 
```bash
git clone <repository-url>
cd llm_from_scratch
```
2. **Install uv package manager**
```sh
pip install uv
```
3. **Run tests to verify installation**
```sh
uv run pytest
```

## Dataset Preparation
### Download TinyStories Dataset
``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

cd ..
```
### Train Tokenizer and Preprocess Data
```python
uv run python scripts/prepare_dataset.py
```
**Note:** Depending on your hardware, this script may take approximately 30 minutes to complete. Upon completion, you'll have:
- A trained BPE tokenizer checkpoint
- Tokenized training and validation datasets
## Training the Model
### Start Training
```python
uv run python scripts/train_llm.py
```
## Monitor Training Process
Launch TensorBoard to visualize training metrics in real-time:
```sh
uv run tensorboard --logdir ./outputs/tensorboard/
```
Then navigate to http://localhost:6006 in your browser to view:
- loss curves
- Learning rate schedules
- gradient norms during the training


## Training Results
![Training Loss](./figures/training_loss.png)
*Training loss decreases steadily over epochs, showing effective learning*
![Learning Rates](./figures/learning_rate.png)
*Cosine annealing schedule provides smooth learning rate decay*

## References and Acknowledge
- [Stanford CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/spring2025/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)
- [TinyStories Dataset](https://huggingface.co/datasets/roneneldan/TinyStories)
- [Flash Attention2](https://arxiv.org/abs/2307.08691)