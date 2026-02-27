#!/usr/bin/env python3
"""
Compare usage of 'estimate' between Apollo peak (FY1967-1969) and inflection (FY1976-1977)
"""

import re
from pathlib import Path
import fitz

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        return ""

def is_prose_line(line):
    if len(line) < 40:
        return False
    alpha_count = sum(1 for c in line if c.isalpha())
    non_alpha_count = len(line) - alpha_count
    if non_alpha_count == 0:
        return alpha_count > 40
    return alpha_count > (1.5 * non_alpha_count)

def extract_prose(text):
    lines = text.split('\n')
    prose_lines = [line.strip() for line in lines if is_prose_line(line.strip())]
    return ' '.join(prose_lines)

def tokenize(text):
    return re.findall(r'\b[a-zA-Z]+\b', text.lower())

def find_estimate_contexts(text, limit=10):
    """Find sentences containing 'estimate' variants."""
    contexts = []
    sentences = re.split(r'[.!?]+', text)

    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 30 or len(sentence) > 300:
            continue
        if re.search(r'\bestimate[sd]?\b', sentence, re.IGNORECASE):
            contexts.append(sentence)
            if len(contexts) >= limit:
                break
    return contexts

def analyze_period(base_dir, years, label):
    print(f"\n{'='*70}")
    print(f"{label}")
    print('='*70)

    combined_text = ""
    for year in years:
        files = list(Path(base_dir).glob(f"FY{year}*.pdf"))
        for f in files:
            text = extract_text_from_pdf(f)
            if text:
                combined_text += " " + extract_prose(text)

    tokens = tokenize(combined_text)
    total = len(tokens)

    # Count estimate variants
    estimate_count = sum(1 for t in tokens if t in ['estimate', 'estimates', 'estimated', 'estimating'])
    estimate_rate = (estimate_count / total) * 1000 if total > 0 else 0

    print(f"Total tokens: {total:,}")
    print(f"'estimate' variants: {estimate_count} ({estimate_rate:.2f}/1k)")

    # Get contexts
    print(f"\nSample contexts:")
    print("-" * 60)
    contexts = find_estimate_contexts(combined_text, 8)
    for i, ctx in enumerate(contexts, 1):
        # Highlight the estimate word
        highlighted = re.sub(r'\b(estimate[sd]?)\b', r'**\1**', ctx, flags=re.IGNORECASE)
        print(f"\n{i}. \"{highlighted}\"")

    return {'tokens': total, 'estimate_count': estimate_count, 'rate': estimate_rate}

def main():
    base_dir = Path(__file__).parent

    # Apollo peak years
    apollo = analyze_period(base_dir, [1967, 1968, 1969], "APOLLO PEAK (FY1967-1969)")

    # Inflection point years
    inflection = analyze_period(base_dir, [1976, 1977], "INFLECTION POINT (FY1976-1977)")

    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print(f"Apollo 'estimate' rate:     {apollo['rate']:.2f}/1k")
    print(f"Inflection 'estimate' rate: {inflection['rate']:.2f}/1k")
    print(f"Change: +{((inflection['rate'] - apollo['rate']) / apollo['rate'] * 100):.0f}%")

if __name__ == "__main__":
    main()
