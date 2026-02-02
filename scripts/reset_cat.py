import json
import os

# Paths (Adjust to match your actual file locations)
input_path = r"/workspace/HCC_dimABSA_remote/outputs/subtask_3/predictions/pred_eng_restaurant.jsonl"
output_path = r"/workspace/HCC_dimABSA_remote/outputs/subtask_2/predictions/pred_eng_restaurant.jsonl"  # Where Subtask 3 looks for input

os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(input_path, 'r', encoding='utf-8') as fin, \
        open(output_path, 'w', encoding='utf-8') as fout:
    count = 0
    for line in fin:
        data = json.loads(line)

        # Check if it has 'Quadruplet' (Task 3 format) or 'Triplet' (Task 2 format)
        if "Quadruplet" in data:
            # Convert Quadruplet -> Triplet by dropping 'Category'
            new_triplets = []
            for q in data["Quadruplet"]:
                new_t = {
                    "Aspect": q["Aspect"],
                    "Opinion": q["Opinion"],
                    "VA": q["VA"]
                }
                new_triplets.append(new_t)

            # Key change: Rename 'Quadruplet' to 'Triplet' for the pipeline
            data["Triplet"] = new_triplets
            del data["Quadruplet"]

        # Write clean version
        fout.write(json.dumps(data, ensure_ascii=False) + "\n")
        count += 1

print(f"Successfully converted {count} lines. Saved to {output_path}")