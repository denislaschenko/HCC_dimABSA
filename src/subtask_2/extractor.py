def decode_bio_tags(tokens, labels):
    spans = []
    current = []

    for tok, tag in zip(tokens, labels):
        if tag == "B":
            if current:
                spans.append(" ".join(current))
                current = []
            current.append(tok)
        elif tag == "I" and current:
            current.append(tok)
        else:
            if current:
                spans.append(" ".join(current))
                current = []

    if current:
        spans.append(" ".join(current))

    return spans


def extract_triplets(text, tokenizer, aspect_labels, opinion_labels, va_pred):
    tokens = tokenizer.tokenize(text)

    aspects = decode_bio_tags(tokens, aspect_labels)
    opinions = decode_bio_tags(tokens, opinion_labels)

    # Pairing strategy (simple but effective)
    triplets = []
    for op in opinions:
        va_mean = va_pred.mean(0)
        v = float(va_mean[0])
        a = float(va_mean[1])

        for asp in aspects:
            triplets.append({
                "Aspect": asp,
                "Opinion": op,
                "VA": f"{v:.2f}#{a:.2f}"
            })

    return triplets
