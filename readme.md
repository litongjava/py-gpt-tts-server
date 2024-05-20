# py-gpt-tts-server
本项目是[GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)的服务端,专注与服务端推理,仅包含必要的python包.并做了大量的优化工作也确保稳定和高性能

## 本项目优势
- 前后端分离
- 工程化
- 高性能

## GPT-SoVITS简介
GPT-SoVITS是一款由RVC创始人RVC-Boss和AI声音转换技术专家Rcell联合开发的跨语言TTS克隆项目，被誉为“最强大的中文声音克隆项目”。

### GPT-SoVITS 简介

GPT-So-ViTS（General Purpose Text-to-Speech with ViTS-Based Style Embeddings）主要是一个文本到语音（Text-to-Speech, TTS）模型，
并不直接基于GPT（Generative Pre-trained Transformer）或BERT（Bidirectional Encoder Representations from Transformers）。
它利用了Transformer架构的一些特性，但是与GPT或BERT解决的自然语言处理问题不同，GPT-So-ViTS专注于生成语音。

- GPT 生成式
- So "Style-optimized”在这里意味着模型不仅仅生成语音，而是能够捕捉并模仿给定语音样本的特定风格，如音调、节奏和情感表达，使生成的语音不仅准确还具有高度的自然感和个性化特征。这种风格的优化是通过对模型的特定组成部分（如ViTS）进行训练，使其能够从声音样本中提取风格信息并有效地应用到合成语音中。
- VITS "Vision Transformer for Style"：用于处理样式嵌入，但在本项目中，它被用来从参考语音中提取样式特征。

### GPT 和 BERT
GPT（Generative Pre-trained Transformer）和BERT（Bidirectional Encoder Representations from Transformers）都是基于Transformer架构的深度学习模型，但它们在设计和用途上有重要的区别：

1. **架构方向性**：
   - **BERT**：是一个双向的模型，这意味着它在处理输入时考虑上下文中的前后信息。BERT通过掩码语言模型（Masked Language Model, MLM）任务进行预训练，其中随机选取的词被掩码（隐藏），模型需要预测这些被掩码的词。
   - **GPT**：是一个单向的模型，主要用于生成任务，它在预训练时使用了一个更直接的语言建模任务，即根据之前的词预测下一个词。

2. **预训练任务**：
   - **BERT**：使用了MLM和下一句预测（Next Sentence Prediction, NSP）的任务来处理和理解语言中的双向上下文关系。
   - **GPT**：使用传统的语言模型任务，即基于前面的文本序列生成下一个词，主要用于文本生成。

3. **用途**：
   - **BERT**：由于其对双向上下文的敏感性，特别适用于需要理解整个输入句子的任务，如情感分析、命名实体识别、问题回答等。
   - **GPT**：更擅长于生成连贯的文本，因此常用于聊天机器人、文本生成、文本摘要等任务。

### BERT和和GPT基于的深度学习框架
GPT和BERT这两个模型都不局限于使用特定的深度学习框架。它们最初是基于不同框架开发的，但现在都可以在多个流行的框架中实现和使用，包括TensorFlow和PyTorch。

- **BERT** 最初是由Google的研究者在2018年发布的，他们使用的是TensorFlow框架。但现在，BERT模型也可以在PyTorch等其他框架中使用，例如通过Hugging Face的Transformers库，该库支持TensorFlow和PyTorch。

- **GPT**（特别是OpenAI的GPT系列，如GPT-2和GPT-3）最初是使用PyTorch开发的。同样，通过像Hugging Face这样的库，GPT模型也可以在TensorFlow中使用。

总的来说，无论是BERT还是GPT，都不依赖于一个特定的深度学习框架。根据开发者的需要和偏好，这些模型可以在多个主流框架中被加载和运行。

### 预训练模型 和 微调模型
进行推理时，你通常需要两部分：预训练模型和微调（或训练）模型。
1. **预训练模型**：预训练模型是指已经使用大量数据训练过的模型，这些模型可以直接使用或用作特定任务的初始设置。
通常，预训练模型是公开可用的，你可以直接下载使用（如从Hugging Face的模型库）。
2. **微调模型**：在机器学习中，把预训练模型经过进一步训练以适应特定任务时而产生的模型，通常被称为“微调模型”或“微调后的模型”。这个过程，即在预训练模型的基础上进行额外训练以提高在特定任务上的表现，通常被称为“微调”（Fine-tuning）。

### 推理需要的模型
在推理过程中需要用到两种模型 预训练模型和微调模型

### 本项目的预训练模型
[下载地址](https://huggingface.co/lj1995/GPT-SoVITS/tree/main)
1. **chinese-hubert-base**
- **HuBERT**（Hidden Unit BERT）通常用于任务。这是一个基于BERT的自监督学习方法，专门设计用于处理语音数据。"chinese-hubert-base"很是一个针对中文语音数据训练的基本版本的HuBERT模型。

2. **chinese-roberta-wwm-ext-large**：
   - **RoBERTa** 是一个基于BERT的改进版，它优化了BERT的训练方法，比如移除了NSP（下一句预测）任务，并使用更大的批次进行训练。"chinese-roberta-wwm-ext-large"是一个扩展的、大型的RoBERTa模型，专为中文文本处理优化。"wwm"代表"Whole Word Masking"（全词遮蔽），这是一种在训练时只遮蔽整个词而非单个字符的技术。

3. **s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt**
   - `s1bert`：这部分名称表明模型是基于BERT架构的一种变体或者特殊应用，`s1`表示模型的一个特定版本或者配置。
   - `25hz-2kh`：这部分表示模型在处理数据时的某些参数，例如频率为25Hz和/或最高频率限制为2kHz，这样的参数与音频或信号处理相关。
   - `longer`：意味着模型使用了更长的训练周期或数据序列。
   - `epoch=68`：表示模型训练了68个训练周期。
   - `step=50232`：表示模型在训练过程中达到了50,232个训练步骤。
   - `ckpt`：文件后缀`.ckpt`代表“checkpoint”，这是PyTorch和TensorFlow等深度学习框架常用的模型保存格式之一，主要用于保存训练过程中的模型状态。

4. **s2D488k.pth**：
   - 文件名中的".pth"表明这是一个PyTorch模型文件。"s2D488k"中的"s2D"和"488k"指模型的具体配置或训练过程中的一些参数，但具体意义不明确。

5. **s2G488k.pth**：
   - 与上一个类似，这也是一个PyTorch模型文件。文件名中的"s2G"和"488k"同样是某种配置或参数的指示，但具体意义不明确。

### 本项目的微调模型
openai_alloy 是以openai的alloy发音人为蓝本微调后的模型
1. **openai_alloy-e15.ckpt** GPT模型：这个部分通常是基于Transformer的，用于处理文本输入。它的任务是理解文本的语义内容，并将文本转换成相应的声学特征，这些特征将被用来生成语音。这一步骤包括语言理解、语调模式的生成等，以确保生成的语音既自然又准确地反映了文本的意图和情感。
2. **openai_alloy_e8_s112.pth** vits模型:这个部分用于存储训练好的网络权重和参数。在GPT-So-ViTS的上下文中，这是用于处理语音样式的部分，如基于ViT的样式嵌入。这个模型负责从参考语音样本中提取风格特征，并将这些特征应用到生成的语音中，以确保语音的自然性和表达性。这部分的作用是捕捉并再现语音的非语言属性，如音色、情感和其他声学细节。


## run
```
conda create --name py-gpt-tts-server python=3.9 -y
conda activate py-gpt-tts-server
git clone https://github.com/litongjava/py-gpt-tts-server.git
pip install -r requirements.txt
python main.py
```





