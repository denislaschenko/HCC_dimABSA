import json
import os
import re

# Absolute path for your remote workspace
PROJECT_ROOT = "/workspace/HCC_dimABSA_remote/"


def merge_and_harmonize(domain="laptop"):
    data_dir = os.path.join(PROJECT_ROOT, "task-dataset", "track_a")
    train_file = os.path.join(data_dir, "subtask_1", "eng", f"eng_{domain}_train_alltasks.jsonl")

    # Target file for production
    output_file = os.path.join(data_dir, "subtask_1", "eng", f"eng_{domain}_production_train.jsonl")

    # Gold file source (fallback to dev if gold is missing)
    gold_file = os.path.join(data_dir, "gold-datasets", f"eng_{domain}_gold_task3.jsonl")
    fallback_gold = os.path.join(data_dir, "subtask_3", "eng", f"eng_{domain}_dev_task3.jsonl")
    selected_gold = gold_file if os.path.exists(gold_file) else fallback_gold

    print(f"\n--- Harmonizing {domain.upper()} Data ---")

    with open(train_file, 'r', encoding='utf-8') as f:
        raw_train = [json.loads(line) for line in f]

    with open(selected_gold, 'r', encoding='utf-8') as f:
        raw_gold = [json.loads(line) for line in f]

    # Handle ID Continuity
    def get_max_id(data):
        ids = [int(re.search(r"(\d+)$", str(item['ID'])).group(1)) for item in data if
               re.search(r"(\d+)$", str(item['ID']))]
        return max(ids) if ids else 0

    current_max = get_max_id(raw_train)
    combined_raw = raw_train + raw_gold
    final_data = []

    for i, item in enumerate(combined_raw):
        # 1. Update ID (using domain_train_XXXX format)
        if i >= len(raw_train):  # If it's a gold/dev sample
            item['ID'] = f"{domain}_train_{current_max + (i - len(raw_train)) + 1}"

        # 2. Harmonize Structure: Convert "Aspect" list to "Quadruplet"
        if "Quadruplet" not in item:
            if "Aspect" in item and isinstance(item["Aspect"], list):
                # Map Aspects to full Quadruplets with neutral defaults
                item["Quadruplet"] = [
                    {
                        "Aspect": asp,
                        "Opinion": "NULL",
                        "Category": f"{domain.upper()}#GENERAL",
                        "VA": "5.00#5.00"
                    } for asp in item["Aspect"]
                ]
                del item["Aspect"]  # Remove the old mismatched key
            else:
                item["Quadruplet"] = []

        final_data.append(item)

    # Write unified file
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in final_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Success! {output_file} created with {len(final_data)} harmonized samples.")


if __name__ == "__main__":
    merge_and_harmonize("restaurant")