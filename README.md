# HCC_dimABSA
Exploring Aspect-based sentiment analysis with continuous emotion scores (Valence &amp; Arousal) – Bachelor-level implementation of SemEval2026.

## System Architecture
Our system utilizes a modular cascading pipeline, specifically separating continuous regression from generative extraction to avoid LLM numerical hallucinations:

1. **Dimensional Regression (DimASR):** Utilizes `roberta-large` trained with a custom Label Distribution Learning (LDL) and Concordance Correlation Coefficient (CCC) objective.
2. **Generative Extraction (DimASTE):** Instruction-tuning of **`Qwen2.5-7B-Instruct`** using **QLoRA** (4-bit quantization). We employ **Retrieval-Augmented In-Context Learning (RAG-ICL)** via `BAAI/bge-base-en-v1.5` to provide dynamic few-shot prompting and ensure structured outputs.
3. **Metric Categorization (DimASQP):** Supervised Contrastive Learning (SupCon) using `sup-simcse-roberta-large` with nearest-centroid inference.

**Core Tech Stack:** `PyTorch`, `transformers`, `peft` (LoRA), `bitsandbytes`, `sentence-transformers`.

## Official Model Performance (Subtask 1: DimASR)

The following table reports the official regression performance across domains using our `roberta-large` LDL implementation:

| Domain | PCC_Valence (↑) | PCC_Arousal (↑) | RMSE_VA (↓) |
| :--- | :--- | :--- | :--- |
| **Laptop** | 0.8531 | 0.5379 | 1.2942 |
| **Restaurant** | 0.8870 | 0.6462 | 1.3011 |

*Note: The model significantly outperformed the 120B GPT-OSS baseline provided by the shared task organizers.*

## Quickstart & Requirements

Ensure you have a GPU with sufficient VRAM for 4-bit LLM inference.

```bash
# Core dependencies
pip install torch transformers peft bitsandbytes sentence-transformers pandas scikit-learn

# Run the pipeline (Subtask 1)
bash run_domains.sh
