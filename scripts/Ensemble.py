import json
import os
import numpy as np

# --- CONFIGURATION ---
PREDICTION_FILES = [
    "../outputs/subtask_1/predictions/pred_eng_laptop_1.jsonl",
    "../outputs/subtask_1/predictions/pred_eng_laptop_42.jsonl",
    "../outputs/subtask_1/predictions/pred_eng_laptop_100.jsonl"
]

OUTPUT_FILE = "../outputs/subtask_1/predictions/pred_ensemble_final.jsonl"


def load_predictions(filepath):
    preds = {}
    try:
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return None

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                preds[data['ID']] = data
        print(f"✅ Loaded {len(preds)} predictions from: {filepath}")
        return preds
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None


def ensemble_predictions():
    print(f"--- Starting Ensemble of {len(PREDICTION_FILES)} files ---")

    all_preds = []
    for f in PREDICTION_FILES:
        p = load_predictions(f)
        if p:
            all_preds.append(p)

    if not all_preds:
        print("⚠️ No valid prediction files loaded. Exiting.")
        return

    print(f"Successfully loaded {len(all_preds)} files. Averaging...")

    # Use the first loaded file as the template
    base_preds = all_preds[0]
    ensemble_results = []

    # Iterate through IDs
    for item_id, item_data in base_preds.items():
        new_record = {
            "ID": item_id,
            "Aspect_VA": []
        }

        # Iterate through aspects
        for idx, aspect_data in enumerate(item_data['Aspect_VA']):
            aspect_name = aspect_data['Aspect']

            valence_scores = []
            arousal_scores = []

            # Collect scores from all models
            for model_preds in all_preds:
                # Safety checks
                if item_id not in model_preds:
                    continue

                # Assuming strict ordering (safe for same-generated files)
                try:
                    other_aspect = model_preds[item_id]['Aspect_VA'][idx]
                    if other_aspect['Aspect'] != aspect_name:
                        # Fallback: search for aspect by name if order differs
                        found = False
                        for a in model_preds[item_id]['Aspect_VA']:
                            if a['Aspect'] == aspect_name:
                                other_aspect = a
                                found = True
                                break
                        if not found:
                            continue
                except IndexError:
                    continue

                # Parse values
                try:
                    v_str, a_str = other_aspect['VA'].split('#')
                    valence_scores.append(float(v_str))
                    arousal_scores.append(float(a_str))
                except ValueError:
                    continue

            # Average
            if valence_scores and arousal_scores:
                final_v = np.mean(valence_scores)
                final_a = np.mean(arousal_scores)

                new_record['Aspect_VA'].append({
                    "Aspect": aspect_name,
                    "VA": f"{final_v:.2f}#{final_a:.2f}"
                })
            else:
                # Fallback if something went wrong (keep original)
                new_record['Aspect_VA'].append(aspect_data)

        ensemble_results.append(new_record)

    # Save output
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for record in ensemble_results:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"🎉 Ensemble saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    ensemble_predictions()