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
* **` Dec. 16rd, 2024`**: We release paper for LLaVA-Steering.
* Code will be released asap after internal review.

## Abstract
Multimodal Large Language Models (MLLMs) have significantly advanced visual tasks by integrating visual representations into large language models (LLMs). The textual modality, inherited from LLMs, equips MLLMs with abilities like instruction following and in-context learning. In contrast, the visual modality enhances performance in downstream tasks by leveraging rich semantic content, spatial information, and grounding capabilities. These intrinsic modalities work synergistically across various visual tasks.
Our research initially reveals a persistent imbalance between these modalities, with text often dominating output generation during visual instruction tuning. This imbalance occurs when using both full fine-tuning and parameter-efficient fine-tuning (PEFT) methods. We then found that re-balancing these modalities can significantly reduce the number of trainable parameters required, inspiring a direction for further optimizing visual instruction tuning. Hence, in this paper, we introduce Modality Linear Representation-Steering (MoReS) to achieve the goal. MoReS effectively re-balances the intrinsic modalities throughout the model, where the key idea is to steer visual representations through linear transformations in the visual subspace across each model layer. 
To validate our solution, we composed LLaVA Steering, a suite of models integrated with the proposed MoReS method. Evaluation results show that the composed LLaVA Steering models require, on average, 500 times fewer trainable parameters than LoRA needs while still achieving comparable performance across three visual benchmarks and eight visual question-answering tasks.
Last, we present the LLaVA Steering Factory, an in-house developed platform that enables researchers to quickly customize various MLLMs with component-based architecture for seamlessly integrating state-of-the-art models, and evaluate their intrinsic modality imbalance. This open-source project enriches the research community to gain a deeper understanding of MLLMs.
<p align="center">
  <img src="Topics/LLaVA-Steering/figs/modelarch.pdf" width="800" />
</p>

## Getting Started
### Installation

**Step 1: Clone LLaVA-Steering repository:**

```bash
git clone https://github.com/bibisbar/LLaVA-Steering.git
cd LLaVA-Steering
```

**Step 2: Environment Setup:**

***Create and activate a new conda environment***

```bash
conda create -n LLaVASteering
conda activate LLaVASteering
```

***Install Dependencies***


```bash
pip install -r requirements.txt
```


### Quick Start

#### Datasets Preparation

Please follow the [MMSeg data preparation document](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md) to download and pre-process the datasets including PASCAL VOC, PASCAL Context, Cityscapes, ADE20k, COCO Object and COCO-Stuff164k.
We provide some dataset processing scripts in the `process_dataset.sh`.