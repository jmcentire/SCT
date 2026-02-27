#!/usr/bin/env python3
"""
Deep analysis of FY1975-1977 inflection point in NASA linguistic data.
Examines which specific hedge words increased and in what context.
"""

import re
import os
from collections import Counter, defaultdict
from pathlib import Path

try:
    import fitz
except ImportError:
    print("ERROR: PyMuPDF not installed")
    exit(1)

# Hedge words grouped by type
HEDGE_CATEGORIES = {
    'epistemic_possibility': ['may', 'might', 'could', 'possibly', 'probable', 'probably', 'likely'],
    'epistemic_expectation': ['would', 'should', 'expect', 'expects', 'expected', 'expecting',
                              'anticipate', 'anticipates', 'anticipated', 'anticipating'],
    'epistemic_appearance': ['appears', 'seems', 'suggest', 'suggests', 'suggested', 'suggesting'],
    'estimation': ['estimate', 'estimates', 'estimated', 'estimating', 'approximately',
                   'projected', 'projecting', 'projection', 'projections',
                   'forecast', 'forecasts', 'forecasted', 'forecasting'],
    'intention': ['believe', 'believes', 'believed', 'believing',
                  'intend', 'intends', 'intended', 'intending',
                  'plan', 'plans', 'planned', 'planning',
                  'target', 'targets', 'targeted', 'targeting']
}

ALL_HEDGES = set()
for words in HEDGE_CATEGORIES.values():
    ALL_HEDGES.update(words)


def extract_text_from_pdf(pdf_path):
    """Extract all text from a PDF file."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"  Error: {e}")
        return ""


def is_prose_line(line):
    """Check if line is prose."""
    if len(line) < 40:
        return False
    alpha_count = sum(1 for c in line if c.isalpha())
    non_alpha_count = len(line) - alpha_count
    if non_alpha_count == 0:
        return alpha_count > 40
    return alpha_count > (1.5 * non_alpha_count)


def extract_prose(text):
    """Extract prose lines from text."""
    lines = text.split('\n')
    prose_lines = [line.strip() for line in lines if is_prose_line(line.strip())]
    return ' '.join(prose_lines)


def tokenize(text):
    """Simple tokenization."""
    return re.findall(r'\b[a-zA-Z]+\b', text.lower())


def find_hedge_contexts(text, window=80):
    """Find sentences/contexts containing hedge words."""
    contexts = defaultdict(list)
    sentences = re.split(r'[.!?]+', text)

    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 20:
            continue
        tokens = tokenize(sentence)
        for token in tokens:
            if token in ALL_HEDGES:
                # Truncate long sentences
                display = sentence[:window*2] + "..." if len(sentence) > window*2 else sentence
                contexts[token].append(display)

    return contexts


def analyze_hedge_distribution(tokens):
    """Analyze hedge word distribution by category."""
    results = {}
    total = len(tokens)

    for category, words in HEDGE_CATEGORIES.items():
        count = sum(1 for t in tokens if t in words)
        rate = (count / total) * 1000 if total > 0 else 0
        word_counts = Counter(t for t in tokens if t in words)
        results[category] = {
            'count': count,
            'rate_per_1k': round(rate, 2),
            'top_words': word_counts.most_common(5)
        }

    return results


def get_fy_files(base_dir, year):
    """Get all PDF files for a fiscal year."""
    pattern = f"FY{year}*.pdf"
    return list(Path(base_dir).glob(pattern))


def analyze_year(base_dir, year):
    """Full analysis of a fiscal year."""
    print(f"\n{'='*60}")
    print(f"FY{year} DETAILED ANALYSIS")
    print('='*60)

    files = get_fy_files(base_dir, year)
    print(f"Files: {len(files)}")
    for f in sorted(files):
        print(f"  - {f.name}")

    # Combine text from all files
    combined_text = ""
    for pdf_file in files:
        text = extract_text_from_pdf(pdf_file)
        if text:
            combined_text += " " + extract_prose(text)

    tokens = tokenize(combined_text)
    total_tokens = len(tokens)
    print(f"\nTotal tokens: {total_tokens:,}")

    # Overall hedge count
    hedge_count = sum(1 for t in tokens if t in ALL_HEDGES)
    hedge_rate = (hedge_count / total_tokens) * 1000 if total_tokens > 0 else 0
    print(f"Total hedges: {hedge_count} ({hedge_rate:.1f}/1k)")

    # By category
    print(f"\nHEDGE BREAKDOWN BY CATEGORY:")
    print("-" * 50)
    dist = analyze_hedge_distribution(tokens)
    for category, data in sorted(dist.items(), key=lambda x: -x[1]['rate_per_1k']):
        print(f"\n{category.upper()}: {data['rate_per_1k']}/1k ({data['count']} occurrences)")
        for word, count in data['top_words']:
            print(f"    {word}: {count}")

    # Sample contexts
    print(f"\nSAMPLE HEDGE CONTEXTS:")
    print("-" * 50)
    contexts = find_hedge_contexts(combined_text)

    # Show top hedges with examples
    all_hedge_counts = Counter(t for t in tokens if t in ALL_HEDGES)
    for word, count in all_hedge_counts.most_common(10):
        print(f"\n'{word}' ({count} uses):")
        for ctx in contexts[word][:2]:  # Show 2 examples
            print(f"  \"{ctx}\"")

    return {
        'year': year,
        'total_tokens': total_tokens,
        'hedge_count': hedge_count,
        'hedge_rate': hedge_rate,
        'distribution': dist,
        'top_hedges': all_hedge_counts.most_common(15)
    }


def compare_years(results):
    """Compare results across years."""
    print("\n" + "="*70)
    print("YEAR-OVER-YEAR COMPARISON")
    print("="*70)

    years = sorted(results.keys())

    # Header
    print(f"\n{'Category':<25}", end="")
    for year in years:
        print(f"FY{year:>8}", end="")
    print(f"{'Change':>12}")
    print("-" * 70)

    # By category
    categories = list(HEDGE_CATEGORIES.keys())
    for cat in categories:
        print(f"{cat:<25}", end="")
        rates = []
        for year in years:
            rate = results[year]['distribution'][cat]['rate_per_1k']
            rates.append(rate)
            print(f"{rate:>8.1f}", end="")

        # Change from first to last
        if len(rates) >= 2:
            change = ((rates[-1] - rates[0]) / rates[0]) * 100 if rates[0] > 0 else 0
            print(f"{change:>+11.0f}%")
        else:
            print()

    # Total
    print("-" * 70)
    print(f"{'TOTAL HEDGING':<25}", end="")
    rates = []
    for year in years:
        rate = results[year]['hedge_rate']
        rates.append(rate)
        print(f"{rate:>8.1f}", end="")
    change = ((rates[-1] - rates[0]) / rates[0]) * 100 if rates[0] > 0 else 0
    print(f"{change:>+11.0f}%")

    # Top word changes
    print("\n\nTOP HEDGE WORD CHANGES:")
    print("-" * 70)

    # Collect all words
    word_rates = defaultdict(dict)
    for year in years:
        tokens = results[year]['total_tokens']
        for word, count in results[year]['top_hedges']:
            word_rates[word][year] = (count / tokens) * 1000

    # Find biggest increases
    changes = []
    for word, rates_dict in word_rates.items():
        if len(rates_dict) >= 2 and min(years) in rates_dict and max(years) in rates_dict:
            first = rates_dict[min(years)]
            last = rates_dict[max(years)]
            if first > 0.1:  # Only consider words with meaningful presence
                pct_change = ((last - first) / first) * 100
                changes.append((word, first, last, pct_change))

    changes.sort(key=lambda x: -x[3])

    print(f"{'Word':<15} {'FY'+ str(min(years)):>10} {'FY'+str(max(years)):>10} {'Change':>12}")
    print("-" * 50)
    for word, first, last, pct in changes[:15]:
        print(f"{word:<15} {first:>10.2f} {last:>10.2f} {pct:>+11.0f}%")


def main():
    base_dir = Path(__file__).parent

    # Analyze FY1975, FY1976, FY1977
    results = {}
    for year in [1975, 1976, 1977]:
        results[year] = analyze_year(base_dir, year)

    # Compare
    compare_years(results)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
