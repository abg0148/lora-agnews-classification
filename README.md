# lora-agnews-classification
Fine-tuning RoBERTa on the AGNEWS dataset using Low-Rank Adaptation (LoRA) under 1 million trainable parameters — NYU Deep Learning Project, Spring 2025.


# LoRA Fine-Tuning on RoBERTa for AGNEWS Classification under 1M params

**NYU Deep Learning - Spring 2025**  
**Project 2 | Team Members: Abhinav Gupta, Rishi Kaushik, Morgan Waddington**  
**Date: April' 2025**  

---

## Project Overview

This project implements a **parameter-efficient fine-tuning strategy** using **Low-Rank Adaptation (LoRA)** on a frozen **RoBERTa** model for text classification on the [AGNEWS dataset](https://www.kaggle.com/competitions/deep-learning-project-2-spring-2025). The core challenge was to **maximize test accuracy while keeping total trainable parameters under 1 million**.

We achieved a final **test accuracy of _93.48%_** using less than **750k trainable parameters**, while maintaining clarity, reproducibility, and modularity throughout our codebase.

---

## Dataset

The [AGNEWS dataset](https://huggingface.co/datasets/ag_news) is a news categorization dataset with 4 classes:
- **World**
- **Sports**
- **Business**
- **Sci/Tech**

We applied custom preprocessing, filtering out extreme-length examples and removing noise such as URLs, HTML codes, and source annotations.

---

## 🔍 Model Architecture

- **Base Model:** [`roberta-base`](https://huggingface.co/roberta-base)
- **Adaptation Technique:** LoRA (Low-Rank Adaptation)
- **LoRA Config:**
  - `r=8`
  - `alpha=8`
  - `dropout=0.2`
  - `target_modules=["query"]`
- **Trainable Parameters:** ~741k

---

## Training Details

- **Optimizer:** AdamW
- **Learning Rate:** 2e-4
- **Batch Size:** 16 (train), 64 (eval)
- **Epochs:** 3
- **Evaluation Strategy:** Epoch-wise
- **Frameworks Used:** `Transformers`, `Datasets`, `PEFT`, `PyTorch`

---

## Results

| Metric            | Value     |
|-------------------|-----------|
| **Test Accuracy** | 93.48%    |
| **Trainable Params** | ~741k  |
| **LoRA Applied To** | Query weights in attention blocks |

---

## Visualizations

The notebook includes:
- Training vs. Validation Loss curves
- Validation Accuracy across epochs
- Normalized Confusion Matrix
- Label-wise Misclassification Analysis
- Top Misclassified Samples

---

CITATIONS & ACKNOWLEDGEMENTS
----------------------------

This project builds on the work of many researchers, developers, and community contributors. We would like to acknowledge the following sources:

1. LoRA: Low-Rank Adaptation
   Hu, E., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, L., Chen, W. (2021).
   "LoRA: Low-Rank Adaptation of Large Language Models."
   arXiv preprint arXiv:2106.09685.
   https://arxiv.org/abs/2106.09685

2. Kaggle Community Notebooks
   - Nayan Sakhiya. "AG News Text Classification"
     https://www.kaggle.com/code/nayansakhiya/ag-news-text-classification
   - Keith Cooper. "Multi-Class Classification with Transformer Models"
     https://www.kaggle.com/code/keithcooper/multi-class-classification-with-transformer-models

3. Generative Assistance
   We used ChatGPT and Perplexity.ai to assist with idea exploration, debugging, and writing.
   All outputs were reviewed and adapted by team members to meet academic standards.



