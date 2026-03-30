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
[cite_start]*Metric: Joint Root Mean Square Error (RMSE_VA) - Lower is better.*

| Domain | Our Model (RoBERTa LDL) | Baseline (GPT-OSS 120B) |
| :--- | :--- | :--- |
| **Laptop** | **1.2942** | 1.5269 |
| **Restaurant** | **1.3011** | 1.4605 |

### Subtask 2: Dimensional Aspect Sentiment Triplet Extraction (DimASTE)
*Metric: Continuous F1 (cF1) - Higher is better.* 

| Domain | Our Model (Qwen-7B + RAG) | Baseline (GPT-OSS 120B) |
| :--- | :--- | :--- |
| **Laptop** | **0.5635**  | 0.4515 |
| **Restaurant** | **0.6326**  | 0.5442 |

### Subtask 3: Dimensional Aspect Sentiment Quad Prediction (DimASQP)
*Metric: Continuous F1 (cF1) - Higher is better.*

| Domain | Our Model (SupCon + Centroid) | Baseline (Llama-3.3 70B) |
| :--- | :--- | :--- |
| **Laptop** | **0.2512** | 0.2483 |
| **Restaurant** | **0.5612** | 0.5048 |

*Conclusion: Our decoupled architecture using targeted 7B-parameter models successfully rivals and outperforms 70B-120B brute-force baselines across extraction and regression tasks.*

## Quickstart & Requirements

Ensure you have a GPU with sufficient VRAM for 4-bit LLM inference.

```bash
# Core dependencies
pip install torch transformers peft bitsandbytes sentence-transformers pandas scikit-learn

# Example: Run Subtask 1 (DimASR) after setting domain="restaurant" in config
python scripts/run_subtask_1.py

# Example: Run Subtask 2 (DimASTE) Extraction
python scripts/run_subtask_2.py
```
*Note: The script run_domains.sh is provided as a convenience shortcut to sequentially execute Subtask 3 batch processing across all domains.*
