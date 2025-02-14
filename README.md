<div align="center">
<h1>LLaVA-Steering </h1>
<h3>Visual Instruction Tuning with 500x Fewer Parameters through Modality Linear Representation-Steering</h3>
<div>
    <h4 align="center">
        <a href=''><img src=''></a>
    </h4>
</div>
</div>

## News
* **` Feb. 15th, 2025`**: The core codes have been released! More models and details will be released！
* **` Dec. 16th, 2024`**: We release paper for LLaVA-Steering.
* Code will be released asap after internal review.

## Abstract
Multimodal Large Language Models (MLLMs) have significantly advanced visual tasks by integrating visual representations into large language models (LLMs). The textual modality, inherited from LLMs, equips MLLMs with abilities like instruction following and in-context learning. In contrast, the visual modality enhances performance in downstream tasks by leveraging rich semantic content, spatial information, and grounding capabilities. These intrinsic modalities work synergistically across various visual tasks.
Our research initially reveals a persistent imbalance between these modalities, with text often dominating output generation during visual instruction tuning. This imbalance occurs when using both full fine-tuning and parameter-efficient fine-tuning (PEFT) methods. We then found that re-balancing these modalities can significantly reduce the number of trainable parameters required, inspiring a direction for further optimizing visual instruction tuning. Hence, in this paper, we introduce Modality Linear Representation-Steering (MoReS) to achieve the goal. MoReS effectively re-balances the intrinsic modalities throughout the model, where the key idea is to steer visual representations through linear transformations in the visual subspace across each model layer. 
To validate our solution, we composed LLaVA Steering, a suite of models integrated with the proposed MoReS method. Evaluation results show that the composed LLaVA Steering models require, on average, 500 times fewer trainable parameters than LoRA needs while still achieving comparable performance across three visual benchmarks and eight visual question-answering tasks.
Last, we present the LLaVA Steering Factory, an in-house developed platform that enables researchers to quickly customize various MLLMs with component-based architecture for seamlessly integrating state-of-the-art models, and evaluate their intrinsic modality imbalance. This open-source project enriches the research community to gain a deeper understanding of MLLMs.
<p align="center">
  <img src="./figs/modelarch.png" width="800" />
</p>


## Installation

**Step 1: Clone LLaVA-Steering repository:**

```bash
git clone https://github.com/bibisbar/LLaVA-Steering.git
cd LLaVA-Steering
```

**Step 2: Environment Setup:**

***Create and activate a new conda environment***

```Shell
conda create -n llava_steer python=3.10 -y
conda activate llava_steer
```
***Install Dependencies***


```bash
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install flash-attn==2.5.6
pip install --no-deps pyreft==0.0.6 pyvene==0.1.2 evaluate==0.4.2 
```


## Quick Start

### 1. Datasets Preparation
- Please refer to the [Data Preparation](https://tinyllava-factory.readthedocs.io/en/latest/Prepare%20Datasets.html) section in tinyllava's [Documenation](https://tinyllava-factory.readthedocs.io/en/latest/).

- Please refer to the [Evaluation](https://tinyllava-factory.readthedocs.io/en/latest/Evaluation.html) section in TinyLlaVA's [Documenation](https://tinyllava-factory.readthedocs.io/en/latest/Evaluation.html).

### 2. Train
#### Pretraining
- Check pretrain script [HERE](scripts/exp/exp_1st_stage_pretrain)
#### Visual Instruction Tuning
- Check 2nd-stage finetune script [HERE](scripts/exp/exp_2nd_stage_finetune)
- Check 3rd-stage finetune script [HERE](scripts/exp/exp_3rd_stage_finetune)
### 3. Evaluation
- Download the checkpoint from our huggingface or you have trained your own model.
- Specify the ckpt-dir path and data-path in the scripts [HERE](scripts/eval).
- Simply run!
## Custom Your MLLM!
In the LLaVA Steering Factory, we establish standardized training and evaluation pipelines, along with flexible data preprocessing and model configurations. Our framework allows researchers to easily customize their models with various training strategies without the need for additional coding. We implement all mainstream LLMs and vision encoders, including multiple PEFT methods and our proposed MoReS technique. Furthermore, we support a wide range of benchmarks and integrate our intrinsic modality imbalance evaluation. The goal of the LLaVA Steering Factory is to facilitate research in MLLMs, particularly in addressing intrinsic modality imbalance to optimize visual instruction tuning.
<p align="center">
  <img src="./figs/factory.png" width="800" />
</p>

### Model Zoo
Currently, there's one 2-stage trained model of MoReS. More will be released! Stay tuned!

- [MoReS-llava-steering-tinyllava_phi2](https://huggingface.co/BroJun/MoReS-llava-steering)


## Contact
If you have any questions, feel free to either initiate an *Issue* or contact us by email (*bijinhe@outlook.com*).

## ❤️ Community efforts
* Our codebase is built upon the [LLaVA](https://github.com/haotian-liu/LLaVA) and [TinyLLaVA_Factory](https://github.com/TinyLLaVA/TinyLLaVA_Factory?tab=readme-ov-file). Great work!
