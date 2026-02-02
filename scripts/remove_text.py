import json
import os
import sys

current_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_path, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.shared import config


def main():

    input_path = r"/workspace/HCC_dimABSA_remote/outputs/subtask_3/predictions/pred_eng_restaurant.jsonl"

    print(f"Target Input:  {input_path}")

    if not os.path.exists(input_path):
        print(f"ERROR: The file {input_path} does not exist.")
        print("Check if you need to switch to 'OPTION A' in the script if you want to clean the standard prediction file.")
        return

    processed_count = 0

    temp = []


    try:
        with open(input_path, 'r', encoding='utf-8') as fin:

            for line in fin:
                if not line.strip():
                    continue

                data = json.loads(line)
                temp.append(data)

            fin.close()

        with open(input_path, 'w', encoding='utf-8') as fout:
            for data in temp:
                if "Text" in data:
                    del data["Text"]

                fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                processed_count += 1

            fout.close()

        print(f"\nSUCCESS! Processed {processed_count} lines.")
        print(f"Cleaned file saved to:\n{input_path}")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()