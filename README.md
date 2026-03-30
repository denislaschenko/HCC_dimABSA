# HCC_dimABSA
Exploring Aspect-based sentiment analysis with continuous emotion scores (Valence &amp; Arousal) – Bachelor-level implementation of SemEval2026.

## System Architecture
Our system utilizes a modular cascading pipeline, specifically separating continuous regression from generative extraction to avoid LLM numerical hallucinations:

1. **Dimensional Regression (DimASR):** Utilizes `roberta-large` trained with a custom Label Distribution Learning (LDL) and Concordance Correlation Coefficient (CCC) objective.
2. **Generative Extraction (DimASTE):** Instruction-tuning of **`Qwen2.5-7B-Instruct`** using **QLoRA** (4-bit quantization). We employ **Retrieval-Augmented In-Context Learning (RAG-ICL)** via `BAAI/bge-base-en-v1.5` to provide dynamic few-shot prompting and ensure structured outputs.
3. **Metric Categorization (DimASQP):** Supervised Contrastive Learning (SupCon) using `sup-simcse-roberta-large` with nearest-centroid inference.

**Core Tech Stack:** `PyTorch`, `transformers`, `peft` (LoRA), `bitsandbytes`, `sentence-transformers`.

## Official Model Performance (SemEval-2026 Task 3)

We evaluate our modular pipeline across all three DimABSA subtasks on the official CodaBench metrics. We compare our resource-efficient approach (RoBERTa-Large + Qwen-7B) against massive LLM baselines provided by the task organizers.

### Subtask 1: Dimensional Aspect Sentiment Regression (DimASR)
[cite_start]*Metric: Joint Root Mean Square Error (RMSE_VA) - Lower is better.* [cite: 313, 317]

| Domain | Our Model (RoBERTa LDL) | Baseline (GPT-OSS 120B) |
| :--- | :--- | :--- |
| **Laptop** | **1.2942** [cite: 316] | 1.5269 [cite: 320] |
| **Restaurant** | **1.3011** [cite: 316] | 1.4605 [cite: 320] |

### Subtask 2: Dimensional Aspect Sentiment Triplet Extraction (DimASTE)
*Metric: Continuous F1 (cF1) - Higher is better.* [cite: 313, 319]

| Domain | Our Model (Qwen-7B + RAG) | Baseline (GPT-OSS 120B) |
| :--- | :--- | :--- |
| **Laptop** | **0.5635** [cite: 318] | 0.4515 [cite: 323] |
| **Restaurant** | **0.6326** [cite: 318] | 0.5442 [cite: 323] |

### Subtask 3: Dimensional Aspect Sentiment Quad Prediction (DimASQP)
*Metric: Continuous F1 (cF1) - Higher is better.* [cite: 313, 319]

| Domain | Our Model (SupCon + Centroid) | Baseline (Llama-3.3 70B) |
| :--- | :--- | :--- |
| **Laptop** | **0.2512** [cite: 318] | 0.2483 [cite: 326] |
| **Restaurant** | **0.5612** [cite: 318] | 0.5048 [cite: 327] |

*Conclusion: Our decoupled architecture using targeted 7B-parameter models successfully rivals and outperforms 70B-120B brute-force baselines across extraction and regression tasks.* [cite: 329, 330, 341]

## Quickstart & Requirements

Ensure you have a GPU with sufficient VRAM for 4-bit LLM inference.

```bash
# Core dependencies
pip install torch transformers peft bitsandbytes sentence-transformers pandas scikit-learn

# Run the pipeline (Subtask 1)
bash run_domains.sh
