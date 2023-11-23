# GBSWT5

Huggingface Transformers-compatible GBST+T5 implementation(as CharFormer(Tay et al., 2022)) for GBST-KEByT5 Model.

Supports following pretrained checkpoints:
  * etri-lirs/gbst-kebyt5-base-preview
  * etri-lirs/gbst-kebyt5-large-preview (not yet)

Copyright (C), 2023- Jong-hun Shin, Electronics and Telecommunications Research Institute. All rights reserved.

## How To Use
Install with pip.
```
pip install git+https://github.com/etri-crossmodal/gbswt5.git
```

### How to load a model
```
import gbswt5
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("etri-lirs/gbst-kebyt5-base-preview")
model = AutoModelForSeq2SeqLM.from_pretrained("etri-lirs/gbst-kebyt5-base-preview")
```

## Dependency
 * pytorch>=1.8.0
 * transformers>=4.27.0
 * einops>=0.6.0

## Acknowledgement

 * This software was supported by the Institute of Information & communication Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT). (No. RS-2022-00187238, Development of Large Korean Language Model Technology for Efficient Pre-training)
 * This software includes lucidrains/charformer-pytorch GitHub project for GBST implementation, which distributed under MIT License. Copyright (c) 2021 Phil Wang. all rights reserved. (Original Code URL: https://github.com/lucidrains/charformer-pytorch)
 * This software includes HuggingFace transformers's T5 implementation for GBST-enabled T5 model, which distributed under Apache 2.0 License. Copyright 2018- The Huggingface team. All rights reserved.
 
We are grateful for their excellent works.
