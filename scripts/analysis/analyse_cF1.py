import json
from collections import Counter, defaultdict


def normalize_quad(q):
    return (q['Aspect'], q['Category'], q['Opinion'])


def analyze_errors(pred_file, gold_file):
    print(f"Analyzing {pred_file} against {gold_file}...")

    with open(pred_file) as f:
        preds = {json.loads(line)['ID']: json.loads(line) for line in f}
    with open(gold_file) as f:
        golds = {json.loads(line)['ID']: json.loads(line) for line in f}

    tp, fp, fn = 0, 0, 0
    cat_confusions = Counter()
    span_errors = []

    for id, gold_item in golds.items():
        if id not in preds: continue

        gold_quads = gold_item.get("Quadruplet", [])
        pred_quads = preds[id].get("Quadruplet", [])

        gold_set = set(normalize_quad(q) for q in gold_quads)
        pred_set = set(normalize_quad(q) for q in pred_quads)

        common = gold_set & pred_set
        tp += len(common)

        for p in pred_set - gold_set:
            fp += 1
            found_aspect_match = False
            for g in gold_set:
                if p[0] == g[0] and p[2] == g[2]:  # Same Aspect & Opinion
                    cat_confusions[(p[1], g[1])] += 1
                    found_aspect_match = True

            if not found_aspect_match:
                for g in gold_set:
                    if p[1] == g[1] and (p[0] in g[0] or g[0] in p[0]):
                        span_errors.append(f"Pred: '{p[0]}' | Gold: '{g[0]}'")

        fn += len(gold_set - pred_set)

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-9)

    print(f"\n--- LOCAL SCORE ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    print(f"\n--- TOP CATEGORY CONFUSIONS (Pred -> Gold) ---")
    for (pred_cat, gold_cat), count in cat_confusions.most_common(10):
        print(f"Predicted '{pred_cat}' but was '{gold_cat}': {count} times")

    print(f"\n--- TOP SPAN ERRORS (Partial Matches) ---")
    for err in span_errors[:10]:
        print(err)


if __name__ == "__main__":
    analyze_errors("pred_local.jsonl", "local_dev_gold.jsonl")