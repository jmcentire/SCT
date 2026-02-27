#!/usr/bin/env python3
"""
NASA Budget Document Linguistic Analyzer
Extracts metrics from Congressional budget justifications (1961-present)
for Variance Compression Thesis analysis.
"""

import re
import os
import csv
import math
import glob
from collections import Counter
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    print("ERROR: PyMuPDF not installed. Run: pip3 install pymupdf")
    exit(1)

# Hedge words (epistemic markers)
HEDGE_WORDS = {
    'may', 'might', 'could', 'would', 'should', 'possibly', 'likely',
    'appears', 'seems', 'suggest', 'suggests', 'suggested', 'suggesting',
    'estimate', 'estimates', 'estimated', 'estimating',
    'approximately', 'anticipate', 'anticipates', 'anticipated', 'anticipating',
    'expect', 'expects', 'expected', 'expecting',
    'believe', 'believes', 'believed', 'believing',
    'intend', 'intends', 'intended', 'intending',
    'plan', 'plans', 'planned', 'planning',
    'target', 'targets', 'targeted', 'targeting',
    'projected', 'projecting', 'projection', 'projections',
    'forecast', 'forecasts', 'forecasted', 'forecasting',
    'probable', 'probably'
}

# Nominalization suffixes
NOMINALIZATION_PATTERN = re.compile(r'\b\w+(tion|sion|ment|ance|ence|ity|ness)\b', re.IGNORECASE)

# Narrative section headers to find
NARRATIVE_HEADERS = re.compile(
    r'^[\s\d.]*\b(MISSION|PROGRAM\s+OBJECTIVES?|JUSTIFICATION|PROJECT\s+DESCRIPTION|'
    r'PURPOSE|BACKGROUND|INTRODUCTION|OVERVIEW|OBJECTIVES?|GOALS?|SUMMARY|'
    r'DESCRIPTION|STRATEGY|RATIONALE|BASIS\s+OF|EXPLANATION)\b',
    re.IGNORECASE | re.MULTILINE
)


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
        print(f"  Error extracting {pdf_path}: {e}")
        return ""


def is_prose_line(line):
    """Check if line is prose (alpha > 1.5× non-alpha, length > 40 chars)."""
    if len(line) < 40:
        return False
    alpha_count = sum(1 for c in line if c.isalpha())
    non_alpha_count = len(line) - alpha_count
    if non_alpha_count == 0:
        return alpha_count > 40
    return alpha_count > (1.5 * non_alpha_count)


def extract_narrative_sections(text):
    """Extract prose from narrative sections."""
    lines = text.split('\n')
    prose_lines = []
    in_narrative = False

    for line in lines:
        stripped = line.strip()

        # Check for narrative header
        if NARRATIVE_HEADERS.match(stripped):
            in_narrative = True
            continue

        # Check for section break (all caps header, table indicators, etc.)
        if stripped and len(stripped) > 3:
            if stripped.isupper() and len(stripped) > 10:
                # Could be new section header
                if not any(h in stripped.upper() for h in ['MISSION', 'OBJECTIVE', 'JUSTIFICATION', 'PURPOSE', 'BACKGROUND', 'DESCRIPTION', 'OVERVIEW']):
                    in_narrative = False

        # If in narrative section and line is prose, include it
        if is_prose_line(stripped):
            prose_lines.append(stripped)

    # If no narrative sections found, fall back to all prose lines
    if len(prose_lines) < 100:
        prose_lines = [line.strip() for line in lines if is_prose_line(line.strip())]

    return ' '.join(prose_lines)


def tokenize(text):
    """Simple tokenization - split on whitespace and punctuation."""
    # Remove punctuation and split
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    return words


def calculate_shannon_entropy(tokens):
    """Calculate Shannon entropy of token distribution."""
    if not tokens:
        return 0.0

    token_counts = Counter(tokens)
    total = len(tokens)
    entropy = 0.0

    for count in token_counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    return round(entropy, 2)


def calculate_metrics(text, sample_size=10000):
    """Calculate all linguistic metrics for a text."""
    tokens = tokenize(text)

    if not tokens:
        return None

    # Sample if too large (for LD comparability)
    if len(tokens) > sample_size:
        # Take evenly distributed sample
        step = len(tokens) // sample_size
        sampled_tokens = tokens[::step][:sample_size]
    else:
        sampled_tokens = tokens

    total_tokens = len(tokens)
    unique_tokens = len(set(sampled_tokens))

    # Lexical Diversity (type-token ratio on sample)
    ld = round(unique_tokens / len(sampled_tokens), 4) if sampled_tokens else 0

    # Shannon Entropy
    se = calculate_shannon_entropy(sampled_tokens)

    # Hedging density (per 1000 tokens)
    hedge_count = sum(1 for t in tokens if t in HEDGE_WORDS)
    hedge_per_1k = round((hedge_count / total_tokens) * 1000, 1) if total_tokens else 0

    # Nominalization density (per 1000 tokens)
    nom_count = len(NOMINALIZATION_PATTERN.findall(text))
    nom_per_1k = round((nom_count / total_tokens) * 1000, 1) if total_tokens else 0

    return {
        'total_tokens': total_tokens,
        'unique_tokens': unique_tokens,
        'lexical_diversity': ld,
        'shannon_entropy': se,
        'hedge_per_1k': hedge_per_1k,
        'nom_per_1k': nom_per_1k
    }


def get_fiscal_year(filename):
    """Extract fiscal year from filename."""
    match = re.search(r'FY(\d{4})', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def get_document_type(filename):
    """Categorize document type from filename."""
    fname = filename.lower()
    if 'volume-1' in fname or 'vol-1' in fname or 'vol1' in fname:
        return 'Vol1'
    elif 'volume-2' in fname or 'vol-2' in fname or 'vol2' in fname:
        return 'Vol2'
    elif 'volume-3' in fname or 'vol-3' in fname or 'vol3' in fname:
        return 'Vol3'
    elif 'request' in fname:
        return 'Request'
    elif 'amendment' in fname:
        return 'Amendment'
    elif 'backup' in fname:
        return 'Backup'
    else:
        return 'General'


def process_fiscal_year(year, pdf_files, base_dir):
    """Process all PDFs for a given fiscal year and combine metrics."""
    year_files = [f for f in pdf_files if get_fiscal_year(os.path.basename(f)) == year]

    if not year_files:
        return None

    # Combine text from all files for this year
    combined_text = ""
    doc_types = set()

    for pdf_file in sorted(year_files):
        fname = os.path.basename(pdf_file)
        print(f"    Processing: {fname}")

        text = extract_text_from_pdf(pdf_file)
        if text:
            narrative = extract_narrative_sections(text)
            combined_text += " " + narrative
            doc_types.add(get_document_type(fname))

    if not combined_text.strip():
        return None

    metrics = calculate_metrics(combined_text)
    if metrics:
        metrics['fiscal_year'] = f"FY{year}"
        metrics['document_type'] = '+'.join(sorted(doc_types))

    return metrics


def main():
    base_dir = Path(__file__).parent

    # Find all PDF files
    pdf_files = list(base_dir.glob("FY*.pdf"))
    print(f"Found {len(pdf_files)} PDF files")

    # Get unique fiscal years
    years = sorted(set(get_fiscal_year(f.name) for f in pdf_files if get_fiscal_year(f.name)))
    print(f"Fiscal years: {min(years)}-{max(years)}")

    # Filter to years we need (FY1977+, since FY1961-1976 already done)
    years_to_process = [y for y in years if y >= 1977]
    print(f"Processing: FY{min(years_to_process)}-FY{max(years_to_process)}")

    # Load existing data
    existing_csv = base_dir.parent / "Downloads" / "nasa_metrics_through_fy1976.csv"
    existing_data = []
    if existing_csv.exists():
        with open(existing_csv, 'r') as f:
            reader = csv.DictReader(f)
            existing_data = list(reader)
        print(f"Loaded {len(existing_data)} existing records from CSV")

    # Process each year
    results = []
    for year in years_to_process:
        print(f"\n  FY{year}...")
        metrics = process_fiscal_year(year, [str(f) for f in pdf_files], base_dir)
        if metrics:
            results.append(metrics)
            print(f"    -> {metrics['total_tokens']} tokens, hedge={metrics['hedge_per_1k']}/1k, nom={metrics['nom_per_1k']}/1k")

    # Combine with existing data
    all_data = existing_data + results

    # Write output CSV
    output_file = base_dir / "nasa_metrics_fy1961_fy2025.csv"
    fieldnames = ['fiscal_year', 'total_tokens', 'unique_tokens', 'lexical_diversity',
                  'shannon_entropy', 'hedge_per_1k', 'nom_per_1k', 'document_type']

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_data:
            writer.writerow(row)

    print(f"\n\nResults written to: {output_file}")
    print(f"Total records: {len(all_data)}")

    # Print summary table
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"{'FY':<8} {'Tokens':>10} {'LD':>8} {'SE':>6} {'Hedge/1k':>10} {'Nom/1k':>10}")
    print("-"*80)
    for row in results:
        print(f"{row['fiscal_year']:<8} {row['total_tokens']:>10} {row['lexical_diversity']:>8} "
              f"{row['shannon_entropy']:>6} {row['hedge_per_1k']:>10} {row['nom_per_1k']:>10}")


if __name__ == "__main__":
    main()
