import json
import os

# CONFIG
PRED_FILE = r"/outputs/subtask_2/predictions/pred_eng_laptop.jsonl"  # Your BEST prediction file
GOLD_FILE = r"/task-dataset/track_a/gold-datasets/eng_laptop_gold_task2.jsonl"
OUTPUT_FILE = r"/task-dataset/track_a/subtask_2/eng/dpo_pairs_laptop.jsonl"


def format_triplets(triplet_list):
    """Converts a list of dicts to the string format the model outputs."""
    # Matches the format: (Aspect, Opinion, VA), (Aspect, Opinion, VA)
    parts = []
    for t in triplet_list:
        parts.append(f"({t['Aspect']}, {t['Opinion']}, {t['VA']})")
    return ", ".join(parts)


def main():
    # 1. Load Gold Data
    gold_map = {}
    with open(GOLD_FILE, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            gold_map[item["ID"]] = item

    # 2. Load Predictions & Generate Pairs
    dpo_data = []
    mistake_count = 0

    # Synchronized Prompt (Matches SFT structure but removes hallucination bait)
    INSTRUCTION = """Below is an instruction describing a task, paired with an input that provides additional context. Your goal is to generate an output that correctly completes the task.

    ### Instruction:
    Given a textual instance [Text], extract all (A, O, VA) triplets, where:
    - A is an Aspect term (a phrase describing an entity mentioned in [Text])
    - O is an Opinion term
    - VA is a Valence–Arousal score in the format (valence#arousal).

    ### Example:
    Input:
    [Text]

    Output:
    [Triplet]

    ### Question:
    Now complete the following example:
    Input:
    """

    with open(PRED_FILE, "r", encoding="utf-8") as f:
        for line in f:
            pred_item = json.loads(line)
            p_id = pred_item["ID"]

            if p_id not in gold_map: continue

            gold_item = gold_map[p_id]

            # Format the "Strings" for comparison
            pred_str = format_triplets(pred_item["Triplet"])
            gold_str = format_triplets(gold_item["Triplet"])

            # CRITICAL: Only create a training example if the model was WRONG.
            # DPO learns from the GAP between "Wrong" and "Right".
            if pred_str != gold_str:
                mistake_count += 1

                # Build the DPO Entry
                # "prompt": The input text sent to the model
                # "chosen": The correct Gold Standard response
                # "rejected": The incorrect prediction the model actually made

                full_prompt = f"{INSTRUCTION}\n[Text] {gold_item['Text']}\n\nOutput:\n"

                dpo_entry = {
                    "prompt": full_prompt,
                    "chosen": gold_str,
                    "rejected": pred_str
                }
                dpo_data.append(dpo_entry)

    # 3. Save DPO Dataset
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for entry in dpo_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Generated {len(dpo_data)} DPO pairs from {mistake_count} mistakes.")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()