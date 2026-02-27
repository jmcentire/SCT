#!/usr/bin/env python3
"""Paper 3: Constellation-Indexed Model Composition.

Five experimental passes:
  Pass 1: Index construction (specialist fingerprints)
  Pass 2: Retrieval evaluation (generalist fingerprint → specialist lookup)
  Pass 3: Correlation evaluation (activation signature vs. answer quality)
  Pass 4: Composition evaluation (node-level weighted parameter mixing)
  Pass 5: Cross-domain evaluation (multi-domain queries)

Usage:
  # Train specialists from generalist checkpoint:
  python paper3_constellation.py --train-specialists --device cuda

  # Run all evaluation passes:
  python paper3_constellation.py --run-all --device cuda

  # Run individual passes:
  python paper3_constellation.py --pass 1 --device cpu
  python paper3_constellation.py --pass 2 --device cpu
"""
import os, sys, time, json, gc, argparse, math
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig, AutoModelForCausalLM, AutoTokenizer,
    GPT2LMHeadModel,
)

sys.path.insert(0, os.path.dirname(__file__) or '.')

from src.fingerprint.capture import build_probe_set, capture_at_checkpoint
from src.regime.detect import cosine_similarity_batch
from src.speculative.predictors import compute_val_loss
from src.constellation.compose import (
    compute_correlation_scores, get_activation_signature,
    compute_node_weights, compose_parameters, weight_entropy,
    task_arithmetic_compose,
)

# Try to import extended cross-domain probes (150 probes, 25 per pair)
# Falls back to built-in 30 probes (5 per pair) if not available
try:
    from paper3_extended_probes import EXTENDED_CROSS_DOMAIN_PROBES
    _USE_EXTENDED_PROBES = True
except ImportError:
    _USE_EXTENDED_PROBES = False

# ============================================================
# Constants
# ============================================================

DOMAINS = ['medical', 'legal', 'code', 'science', 'general']
PROBES_PER_DOMAIN = 40
NUM_PROBES = PROBES_PER_DOMAIN * len(DOMAINS)  # 200

SPECIALIST_TRAIN_STEPS = 2000
SPECIALIST_BATCH_SIZE = 4
SPECIALIST_LR = 5e-5
SPECIALIST_MAX_LENGTH = 256

COMPOSE_LAST_N_LAYERS = 4
TOP_K_SPECIALISTS = 3

# ============================================================
# Domain Probe Set
# ============================================================

DOMAIN_PROBES = {
    'medical': [
        "The patient presented with acute respiratory distress syndrome requiring",
        "Metformin is commonly prescribed for type 2 diabetes because it",
        "The mechanism of action of selective serotonin reuptake inhibitors involves",
        "Magnetic resonance imaging revealed a lesion in the left temporal lobe",
        "Treatment of hypertension typically begins with lifestyle modifications including",
        "The pathophysiology of Alzheimer's disease involves accumulation of",
        "Cardiac catheterization showed significant stenosis in the left anterior",
        "Immunoglobulin E plays a central role in allergic reactions by",
        "The Glasgow Coma Scale assesses level of consciousness based on",
        "Antibiotic resistance develops through mechanisms including horizontal gene transfer",
        "Pulmonary function tests indicated an obstructive pattern consistent with",
        "The hypothalamic-pituitary-adrenal axis regulates the stress response through",
        "Surgical intervention was indicated due to bowel obstruction caused by",
        "Pharmacokinetics of warfarin are affected by polymorphisms in the CYP2C9",
        "The five-year survival rate for stage III non-small cell lung cancer",
        "Chronic kidney disease is classified into stages based on glomerular filtration",
        "The blood-brain barrier restricts passage of molecules based on their",
        "Electrocardiogram findings consistent with myocardial infarction include ST-segment",
        "Vaccination works by stimulating the adaptive immune system to produce",
        "The differential diagnosis for chest pain includes cardiac, pulmonary, and",
        "Cerebrospinal fluid analysis showed elevated protein and pleocytosis suggesting",
        "Beta-blockers reduce heart rate by antagonizing the effects of catecholamines",
        "The WHO classification of brain tumors is based on histological and",
        "Hemoglobin A1c reflects average blood glucose levels over the preceding",
        "Deep vein thrombosis risk factors include immobilization, surgery, and",
        "The renin-angiotensin-aldosterone system regulates blood pressure through vasoconstriction",
        "Autoimmune diseases result from the immune system attacking self-antigens in",
        "Pain management in palliative care follows the WHO analgesic ladder from",
        "The phases of clinical trials progress from safety assessment in Phase I",
        "Osteoporosis is diagnosed when bone mineral density T-score falls below",
        "Sepsis is defined as life-threatening organ dysfunction caused by a",
        "The cranial nerves include twelve pairs originating from the brainstem",
        "Insulin resistance is a hallmark of metabolic syndrome characterized by",
        "Genetic testing for hereditary breast cancer includes screening for BRCA1",
        "The complement system consists of plasma proteins that enhance innate",
        "Perioperative management of anticoagulation requires balancing bleeding and thrombosis",
        "Rheumatoid arthritis is distinguished from osteoarthritis by the presence of",
        "The stages of wound healing include hemostasis, inflammation, proliferation",
        "Liver cirrhosis complications include portal hypertension, ascites, and hepatic",
        "Neuroplasticity allows the brain to reorganize neural pathways in response to",
    ],
    'legal': [
        "The doctrine of stare decisis requires courts to follow precedent",
        "Under the Fourth Amendment, warrantless searches are presumptively unreasonable",
        "The elements of negligence include duty, breach, causation, and damages",
        "Contract formation requires offer, acceptance, consideration, and mutual assent",
        "The Commerce Clause grants Congress power to regulate interstate commerce",
        "Due process under the Fourteenth Amendment includes both procedural and",
        "The parol evidence rule generally excludes prior oral agreements that contradict",
        "Strict liability applies to abnormally dangerous activities regardless of the",
        "The doctrine of sovereign immunity protects governments from being sued without",
        "Intellectual property rights include patents, trademarks, copyrights, and trade",
        "The burden of proof in criminal cases requires the prosecution to",
        "Fiduciary duty requires acting in the best interest of the beneficiary",
        "The Uniform Commercial Code governs transactions in goods and provides",
        "Res judicata prevents parties from relitigating issues that have already been",
        "The statute of limitations sets the maximum time after an event within",
        "Habeas corpus allows prisoners to challenge the legality of their detention",
        "The exclusionary rule prevents illegally obtained evidence from being admitted",
        "Corporate governance structures include the board of directors, officers, and",
        "Employment at-will doctrine allows either party to terminate the employment",
        "The Clean Air Act establishes national ambient air quality standards for",
        "Antitrust law under the Sherman Act prohibits monopolization and restraint",
        "The right to counsel under the Sixth Amendment applies to all",
        "Tort reform measures include caps on damages and modification of joint",
        "International law sources include treaties, customary law, and general principles",
        "The Equal Protection Clause prohibits states from denying any person within",
        "Bankruptcy proceedings under Chapter 11 allow reorganization of debts while",
        "The Miranda warning must be given before custodial interrogation to protect",
        "Environmental law establishes frameworks for regulating pollution and protecting natural",
        "The doctrine of promissory estoppel enforces promises that induce reasonable",
        "Securities regulation under the SEC requires disclosure of material information",
        "Product liability theories include strict liability, negligence, and breach of",
        "The First Amendment protects freedom of speech from government censorship but",
        "Class action lawsuits allow one party to represent a group with common",
        "Administrative law governs the activities of government agencies including rulemaking",
        "The Takings Clause requires just compensation when private property is taken",
        "Mediation differs from arbitration in that the mediator facilitates negotiation",
        "The legal standard for defamation requires proof of a false statement",
        "Property law distinguishes between real property and personal property based on",
        "The Americans with Disabilities Act prohibits discrimination against individuals with",
        "Maritime law governs activities on navigable waters and includes admiralty",
    ],
    'code': [
        "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr)",
        "class BinarySearchTree:\n    def __init__(self):\n        self.root = None\n    def insert(self",
        "import asyncio\nasync def fetch_data(url):\n    async with aiohttp.ClientSession() as session",
        "def fibonacci(n, memo={}):\n    if n in memo:\n        return memo[n]\n    if n <= 1",
        "SELECT customers.name, orders.total FROM customers INNER JOIN orders ON",
        "const express = require('express');\nconst app = express();\napp.get('/api",
        "def dijkstra(graph, start):\n    distances = {node: float('inf') for node in graph",
        "class LinkedList:\n    class Node:\n        def __init__(self, data):\n            self.data",
        "import torch.nn as nn\nclass TransformerBlock(nn.Module):\n    def __init__(self",
        "def quick_sort(arr, low, high):\n    if low < high:\n        pivot = partition(arr",
        "from flask import Flask, request, jsonify\napp = Flask(__name__)\n@app.route",
        "def depth_first_search(graph, start, visited=None):\n    if visited is None",
        "CREATE TABLE users (\n    id SERIAL PRIMARY KEY,\n    username VARCHAR(50)",
        "class MaxHeap:\n    def __init__(self):\n        self.heap = []\n    def push(self",
        "import numpy as np\ndef gradient_descent(f, grad_f, x0, learning_rate=0.01",
        "async function fetchAPI(endpoint) {\n    try {\n        const response = await fetch",
        "def lru_cache(maxsize=128):\n    def decorator(func):\n        cache = OrderedDict()",
        "class Graph:\n    def __init__(self, directed=False):\n        self.adj = defaultdict",
        "import unittest\nclass TestStringMethods(unittest.TestCase):\n    def test_upper(self",
        "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left",
        "from dataclasses import dataclass, field\n@dataclass\nclass Config:\n    learning_rate",
        "def knapsack(weights, values, capacity):\n    n = len(weights)\n    dp = [[0",
        "import re\ndef validate_email(email):\n    pattern = r'^[a-zA-Z0-9._%+-]+@",
        "class ThreadPool:\n    def __init__(self, num_workers):\n        self.queue = Queue()",
        "def topological_sort(graph):\n    in_degree = {u: 0 for u in graph}\n    for u",
        "app.use(express.json());\napp.post('/users', async (req, res) => {\n    const",
        "def levenshtein_distance(s1, s2):\n    m, n = len(s1), len(s2)\n    dp = [[0",
        "class RedBlackTree:\n    RED = True\n    BLACK = False\n    def __init__(self",
        "import pandas as pd\ndf = pd.read_csv('data.csv')\ndf_clean = df.dropna(subset",
        "def producer_consumer(buffer_size=10):\n    buffer = queue.Queue(maxsize=buffer_size)",
        "from typing import List, Optional\ndef two_sum(nums: List[int], target: int)",
        "class Singleton:\n    _instance = None\n    def __new__(cls):\n        if cls._instance",
        "import logging\nlogger = logging.getLogger(__name__)\nlogger.setLevel(logging.DEBUG)",
        "def matrix_multiply(A, B):\n    rows_A, cols_A = len(A), len(A[0])\n    rows_B",
        "kubectl apply -f deployment.yaml\napiVersion: apps/v1\nkind: Deployment\nmetadata",
        "class Observable:\n    def __init__(self):\n        self._observers = []\n    def subscribe",
        "def count_inversions(arr):\n    if len(arr) <= 1:\n        return arr, 0\n    mid",
        "from collections import Counter\ndef most_common_words(text, n=10):\n    words",
        "class LRUCache:\n    def __init__(self, capacity):\n        self.capacity = capacity",
        "def a_star(graph, start, goal, heuristic):\n    open_set = [(0, start)]",
    ],
    'science': [
        "The standard model of particle physics describes three of the four fundamental",
        "Photosynthesis converts carbon dioxide and water into glucose using light energy",
        "The Heisenberg uncertainty principle states that position and momentum cannot both",
        "Plate tectonics theory explains how Earth's lithosphere is divided into moving",
        "The Drake equation estimates the number of detectable civilizations in the Milky",
        "CRISPR-Cas9 gene editing allows precise modification of DNA sequences by guiding",
        "The second law of thermodynamics states that entropy in an isolated system",
        "Dark matter comprises approximately 27 percent of the universe's mass-energy",
        "The theory of general relativity describes gravity as the curvature of spacetime",
        "Quantum entanglement occurs when particles become correlated such that the state",
        "The Krebs cycle occurs in the mitochondrial matrix and produces electron",
        "Hubble's law describes the relationship between a galaxy's distance and its",
        "Natural selection acts on phenotypic variation within populations leading to",
        "The electromagnetic spectrum ranges from radio waves to gamma rays ordered by",
        "Nuclear fusion in the Sun's core converts hydrogen into helium releasing",
        "The central dogma of molecular biology describes the flow of genetic information",
        "Climate change is driven primarily by anthropogenic emissions of greenhouse gases",
        "Superconductivity occurs when certain materials exhibit zero electrical resistance below a",
        "The periodic table organizes elements by increasing atomic number and recurring",
        "Black holes form when massive stars collapse under their own gravity creating",
        "Enzyme catalysis lowers the activation energy of biochemical reactions through",
        "The double helix structure of DNA was discovered by Watson and Crick",
        "Gravitational waves are ripples in spacetime caused by accelerating massive objects",
        "The ozone layer in the stratosphere absorbs most ultraviolet radiation from",
        "Quantum computing uses qubits that can exist in superposition states unlike",
        "The Big Bang theory describes the origin of the universe from a",
        "Stem cells have the unique ability to differentiate into various specialized",
        "Brownian motion describes the random movement of particles suspended in a",
        "The human genome contains approximately three billion base pairs encoding around",
        "Semiconductor devices work by controlling the flow of electrons through materials",
        "Continental drift was proposed by Alfred Wegener based on the fit of",
        "The speed of light in vacuum is approximately 299,792,458 meters per second",
        "RNA interference is a biological process where RNA molecules inhibit gene expression",
        "The Coriolis effect causes moving objects to be deflected in rotating reference",
        "Photovoltaic cells convert sunlight directly into electricity using semiconductor materials",
        "The mitochondria are often called the powerhouse of the cell because they",
        "Speciation occurs when populations become reproductively isolated leading to divergent",
        "The Chandrasekhar limit is the maximum mass of a stable white dwarf",
        "Epigenetics studies heritable changes in gene expression that do not involve",
        "The Doppler effect describes the change in frequency of a wave relative",
    ],
    'general': [
        "The industrial revolution transformed manufacturing processes beginning in Britain during",
        "Democracy as a form of government traces its origins to ancient Athens",
        "The global supply chain connects manufacturers, distributors, and retailers across",
        "Cultural diversity enriches societies by bringing together different perspectives and",
        "Education systems around the world vary significantly in their approach to",
        "The internet has fundamentally changed how people communicate, work, and access",
        "Urban planning involves designing cities to accommodate growing populations while",
        "The history of jazz music reflects the cultural melting pot of",
        "Agricultural practices have evolved from subsistence farming to large-scale industrial",
        "International trade agreements facilitate economic cooperation between nations by reducing",
        "The concept of human rights has evolved through centuries of philosophical",
        "Public transportation systems reduce traffic congestion and carbon emissions in",
        "The Renaissance marked a period of cultural and intellectual rebirth in",
        "Social media platforms have transformed political discourse and public engagement",
        "Climate adaptation strategies help communities prepare for the impacts of",
        "The printing press invented by Gutenberg revolutionized the dissemination of",
        "Modern architecture emphasizes functionality, minimal ornamentation, and the use of",
        "Globalization has increased economic interdependence among nations while raising concerns",
        "The space race between the United States and Soviet Union drove",
        "Sustainable development balances economic growth with environmental protection and social",
        "The evolution of language reflects cultural changes and human migration patterns",
        "Artificial intelligence applications range from virtual assistants to autonomous vehicles",
        "The Olympic Games promote international cooperation and athletic excellence dating back",
        "Food security remains a global challenge affected by climate change and",
        "The civil rights movement in the United States fought for equality",
        "Renewable energy sources including solar, wind, and hydroelectric power are",
        "The United Nations was established in 1945 to maintain international peace",
        "Philosophy of science examines the foundations, methods, and implications of",
        "Migration patterns throughout history have shaped demographic and cultural landscapes",
        "The digital divide refers to the gap between those with and without",
        "Conservation efforts aim to protect biodiversity and preserve natural habitats",
        "The development of antibiotics in the twentieth century dramatically reduced mortality",
        "Economic inequality has been increasing in many developed nations raising questions",
        "The French Revolution fundamentally altered the political landscape of Europe",
        "Cybersecurity threats include malware, phishing, ransomware, and social engineering",
        "The history of mathematics spans from ancient counting systems to modern",
        "Water scarcity affects billions of people worldwide and is projected to",
        "The silk road facilitated trade and cultural exchange between East and",
        "Mental health awareness has increased significantly in recent decades leading to",
        "The theory of evolution by natural selection was independently conceived by",
    ],
}


class SimpleTokenDataset(Dataset):
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {'input_ids': self.input_ids[idx],
                'attention_mask': self.attention_mask[idx]}


def get_lr(step, max_steps, base_lr, warmup_steps=100):
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    return base_lr * max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))


# ============================================================
# Domain Data
# ============================================================

def load_domain_data(domain, tokenizer, max_length=256, max_samples=5000):
    """Load and tokenize domain-specific training data.

    Uses HuggingFace datasets for domain corpora.
    Returns tokenized input_ids and attention_mask tensors.
    """
    from datasets import load_dataset

    print(f'  Loading {domain} data...')

    if domain == 'general':
        ds = load_dataset('wikitext', 'wikitext-103-v1', split='train')
        texts = [t for t in ds['text'] if len(t.strip()) > 50][:max_samples]

    elif domain == 'medical':
        try:
            ds = load_dataset('pubmed_qa', 'pqa_labeled', split='train')
            texts = [f"{row['question']} {row['long_answer']}" for row in ds][:max_samples]
        except Exception:
            ds = load_dataset('wikitext', 'wikitext-103-v1', split='train')
            texts = [t for t in ds['text'] if any(w in t.lower() for w in
                     ['medical', 'disease', 'patient', 'treatment', 'clinical',
                      'health', 'diagnosis', 'therapy', 'drug', 'cell'])][:max_samples]

    elif domain == 'legal':
        try:
            ds = load_dataset('pile-of-law/pile-of-law', 'r_legaladvice',
                              split='train', streaming=True)
            texts = []
            for row in ds:
                if len(row['text'].strip()) > 50:
                    texts.append(row['text'][:1000])
                if len(texts) >= max_samples:
                    break
        except Exception:
            ds = load_dataset('wikitext', 'wikitext-103-v1', split='train')
            texts = [t for t in ds['text'] if any(w in t.lower() for w in
                     ['law', 'court', 'legal', 'judge', 'statute', 'constitution',
                      'rights', 'contract', 'defendant', 'plaintiff'])][:max_samples]

    elif domain == 'code':
        try:
            ds = load_dataset('bigcode/starcoderdata', 'python',
                              split='train', streaming=True)
            texts = []
            for row in ds:
                if len(row['content'].strip()) > 50:
                    texts.append(row['content'][:1000])
                if len(texts) >= max_samples:
                    break
        except Exception:
            ds = load_dataset('wikitext', 'wikitext-103-v1', split='train')
            texts = [t for t in ds['text'] if any(w in t.lower() for w in
                     ['function', 'algorithm', 'program', 'computer', 'software',
                      'variable', 'data structure', 'compile', 'debug'])][:max_samples]

    elif domain == 'science':
        try:
            ds = load_dataset('scientific_papers', 'arxiv', split='train')
            texts = [row['abstract'] for row in ds
                     if len(row['abstract'].strip()) > 50][:max_samples]
        except Exception:
            ds = load_dataset('wikitext', 'wikitext-103-v1', split='train')
            texts = [t for t in ds['text'] if any(w in t.lower() for w in
                     ['experiment', 'hypothesis', 'molecule', 'physics', 'quantum',
                      'evolution', 'chemical', 'energy', 'theory', 'equation'])][:max_samples]

    else:
        raise ValueError(f'Unknown domain: {domain}')

    print(f'    Got {len(texts)} texts')

    # Tokenize
    all_ids = []
    all_masks = []
    chunk_size = 500
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i+chunk_size]
        encoded = tokenizer(
            chunk, padding='max_length', truncation=True,
            max_length=max_length, return_tensors='pt',
        )
        all_ids.append(encoded['input_ids'])
        all_masks.append(encoded['attention_mask'])

    if not all_ids:
        raise RuntimeError(f'No data loaded for domain {domain}')

    return torch.cat(all_ids), torch.cat(all_masks)


# ============================================================
# Specialist Training
# ============================================================

def train_specialist(domain, generalist_ckpt_path, model_cfg, tokenizer,
                     output_dir, device='cpu', use_amp=False,
                     max_steps=SPECIALIST_TRAIN_STEPS):
    """Fine-tune a specialist from the generalist checkpoint on domain data."""
    print(f'\nTraining {domain} specialist...')

    # Load generalist weights
    ckpt = torch.load(generalist_ckpt_path, weights_only=False, map_location='cpu')
    gen_sd = {k: v.float() for k, v in ckpt['state_dict'].items()}

    model = _get_model_cls()(model_cfg)
    model.load_state_dict(gen_sd)
    model.to(device)
    del gen_sd

    # Load domain data
    input_ids, attention_mask = load_domain_data(domain, tokenizer)
    dataset = SimpleTokenDataset(input_ids, attention_mask)
    dataloader = DataLoader(dataset, batch_size=SPECIALIST_BATCH_SIZE,
                            shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=SPECIALIST_LR, weight_decay=0.01)

    # Disable AMP for non-CUDA devices
    if device not in ('cuda',) and not str(device).startswith('cuda:'):
        use_amp = False
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler(device, enabled=(use_amp and amp_dtype == torch.float16))

    model.train()
    step = 0
    data_iter = iter(dataloader)
    t0 = time.time()

    while step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        batch = {k: v.to(device) for k, v in batch.items()}
        lr = get_lr(step, max_steps, SPECIALIST_LR)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        device_type = device.split(':')[0] if isinstance(device, str) else device.type
        with torch.amp.autocast(device_type=device_type, enabled=use_amp, dtype=amp_dtype):
            out = model(input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['input_ids'])
            loss = out.loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        step += 1
        if step % 200 == 0:
            print(f'  [{domain}] Step {step}/{max_steps} | Loss: {loss.item():.4f} | '
                  f'LR: {lr:.2e} | {time.time()-t0:.0f}s')

    # Save specialist
    spec_dir = output_dir / domain
    spec_dir.mkdir(parents=True, exist_ok=True)
    spec_sd = {k: v.cpu().half() for k, v in model.state_dict().items()}
    torch.save({
        'state_dict': spec_sd,
        'domain': domain,
        'steps': max_steps,
        'final_loss': loss.item(),
    }, spec_dir / 'specialist.pt')

    elapsed = time.time() - t0
    print(f'  [{domain}] Done in {elapsed:.0f}s. Final loss: {loss.item():.4f}')

    del model, optimizer
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    return spec_sd


# ============================================================
# Probe Tokenization
# ============================================================

def tokenize_domain_probes(tokenizer, max_length=128):
    """Tokenize all domain probes, returning tagged probe set.

    Returns:
        dict with 'input_ids', 'attention_mask', 'domain_labels', 'texts'
    """
    all_texts = []
    all_labels = []
    for domain in DOMAINS:
        probes = DOMAIN_PROBES[domain]
        all_texts.extend(probes)
        all_labels.extend([domain] * len(probes))

    encoded = tokenizer(
        all_texts, padding='max_length', truncation=True,
        max_length=max_length, return_tensors='pt',
    )

    return {
        'input_ids': encoded['input_ids'],
        'attention_mask': encoded['attention_mask'],
        'domain_labels': all_labels,
        'texts': all_texts,
    }


# ============================================================
# Pass 1: Index Construction
# ============================================================

def run_pass1_index(specialist_dirs, model_cfg, probe_set, device='cpu'):
    """Build specialist index from activation fingerprints on probe set."""
    print('\n' + '=' * 60)
    print('PASS 1: Index Construction')
    print('=' * 60)

    index = {}  # specialist_id -> list of per-domain mean fingerprints

    for domain in DOMAINS:
        if domain == 'general':
            continue  # General is the generalist, not a specialist

        spec_path = specialist_dirs / domain / 'specialist.pt'
        if not spec_path.exists():
            print(f'  WARNING: No specialist for {domain}, skipping')
            continue

        print(f'  Building index for {domain} specialist...')
        spec_ckpt = torch.load(spec_path, weights_only=False, map_location='cpu')
        spec_sd = {k: v.float() for k, v in spec_ckpt['state_dict'].items()}

        model = _get_model_cls()(model_cfg)
        model.load_state_dict(spec_sd)
        model.to(device)

        # Capture fingerprint on full probe set
        fp = capture_at_checkpoint(model, probe_set, ['final_hidden'],
                                    device=str(device), batch_size=32)
        activations = fp['final_hidden']  # (200, hidden_dim)

        # Compute mean fingerprint per domain
        domain_fps = []
        for d_idx, d_name in enumerate(DOMAINS):
            start = d_idx * PROBES_PER_DOMAIN
            end = start + PROBES_PER_DOMAIN
            domain_fp = activations[start:end].mean(axis=0)
            domain_fps.append(domain_fp)

        index[domain] = domain_fps
        print(f'    {domain}: {len(domain_fps)} domain entries, '
              f'fp dim={activations.shape[1]}')

        del model, spec_sd
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()

    return index


# ============================================================
# Pass 2: Retrieval Evaluation
# ============================================================

def run_pass2_retrieval(generalist_sd, model_cfg, probe_set, index, device='cpu'):
    """Evaluate retrieval: does generalist fingerprint identify correct specialists?"""
    print('\n' + '=' * 60)
    print('PASS 2: Retrieval Evaluation')
    print('=' * 60)

    model = _get_model_cls()(model_cfg)
    model.load_state_dict({k: v.float() for k, v in generalist_sd.items()})
    model.to(device)

    # Get generalist fingerprint for each probe
    fp = capture_at_checkpoint(model, probe_set, ['final_hidden'],
                                device=str(device), batch_size=32)
    gen_activations = fp['final_hidden']  # (200, hidden_dim)

    domain_labels = probe_set['domain_labels']
    results = []

    for i in range(len(domain_labels)):
        true_domain = domain_labels[i]
        query_fp = gen_activations[i]

        # Compute similarity to each specialist's index entries
        scores = compute_correlation_scores(query_fp, index)

        # Sort by score (descending)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top1 = ranked[0][0] if ranked else None
        top3 = [r[0] for r in ranked[:3]]

        results.append({
            'probe_idx': i,
            'true_domain': true_domain,
            'top1_retrieved': top1,
            'top1_correct': top1 == true_domain,
            'top3_retrieved': top3,
            'top3_correct': true_domain in top3,
            'scores': {k: float(v) for k, v in scores.items()},
        })

    del model
    gc.collect()

    # Summary
    by_domain = {}
    for d in DOMAINS:
        if d == 'general':
            continue
        domain_results = [r for r in results if r['true_domain'] == d]
        if domain_results:
            top1_prec = np.mean([r['top1_correct'] for r in domain_results])
            top3_recall = np.mean([r['top3_correct'] for r in domain_results])
            by_domain[d] = {'top1_precision': float(top1_prec),
                            'top3_recall': float(top3_recall),
                            'n': len(domain_results)}

    overall_top1 = np.mean([r['top1_correct'] for r in results
                           if r['true_domain'] != 'general'])
    overall_top3 = np.mean([r['top3_correct'] for r in results
                           if r['true_domain'] != 'general'])

    summary = {
        'overall_top1_precision': float(overall_top1),
        'overall_top3_recall': float(overall_top3),
        'by_domain': by_domain,
    }

    print(f'\n  Overall Top-1 Precision: {overall_top1:.1%}')
    print(f'  Overall Top-3 Recall:    {overall_top3:.1%}')
    for d, s in by_domain.items():
        print(f'    {d:10s}: top1={s["top1_precision"]:.1%}, '
              f'top3={s["top3_recall"]:.1%} (n={s["n"]})')

    return {'evaluations': results, 'summary': summary}


# ============================================================
# Pass 3: Correlation Evaluation
# ============================================================

def run_pass3_correlation(generalist_sd, specialist_dirs, model_cfg, probe_set,
                          index, device='cpu'):
    """Evaluate correlation between activation signature and answer quality."""
    print('\n' + '=' * 60)
    print('PASS 3: Correlation Evaluation')
    print('=' * 60)

    domain_labels = probe_set['domain_labels']
    results = []

    # Get generalist fingerprints
    gen_model = _get_model_cls()(model_cfg)
    gen_model.load_state_dict({k: v.float() for k, v in generalist_sd.items()})
    gen_model.to(device)

    gen_fp = capture_at_checkpoint(gen_model, probe_set, ['final_hidden'],
                                    device=str(device), batch_size=32)
    gen_activations = gen_fp['final_hidden']
    del gen_model
    gc.collect()

    # For each specialist, compute correlation score and perplexity on each probe
    for domain in DOMAINS:
        if domain == 'general':
            continue

        spec_path = specialist_dirs / domain / 'specialist.pt'
        if not spec_path.exists():
            continue

        print(f'  Evaluating {domain} specialist...')
        spec_ckpt = torch.load(spec_path, weights_only=False, map_location='cpu')
        spec_sd = {k: v.float() for k, v in spec_ckpt['state_dict'].items()}

        model = _get_model_cls()(model_cfg)
        model.load_state_dict(spec_sd)
        model.to(device)

        for i in range(len(domain_labels)):
            true_domain = domain_labels[i]
            query_fp = gen_activations[i]

            # Correlation score
            scores = compute_correlation_scores(query_fp, index)
            c_score = scores.get(domain, 0.0)

            # Activation signature magnitude
            act_sig = get_activation_signature(
                model,
                probe_set['input_ids'][i:i+1],
                probe_set['attention_mask'][i:i+1],
                device=str(device),
            )
            act_magnitude = float(np.linalg.norm(act_sig))

            # Perplexity on this probe (answer quality proxy)
            val_batch = {
                'input_ids': probe_set['input_ids'][i:i+1].to(device),
                'attention_mask': probe_set['attention_mask'][i:i+1].to(device),
            }
            loss = compute_val_loss(model, val_batch)

            results.append({
                'probe_idx': i,
                'true_domain': true_domain,
                'specialist_domain': domain,
                'is_matching': true_domain == domain,
                'correlation_score': c_score,
                'activation_magnitude': act_magnitude,
                'combined_signal': c_score * act_magnitude,
                'loss': loss,
                'perplexity': math.exp(min(loss, 20)),
            })

        del model, spec_sd
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()

    # Compute Pearson correlation between combined_signal and quality
    signals = np.array([r['combined_signal'] for r in results])
    losses = np.array([r['loss'] for r in results])
    # Lower loss = better quality, so we expect negative correlation with loss
    pearson_r = float(np.corrcoef(signals, -losses)[0, 1])

    # Also compute for matching vs non-matching
    matching = [r for r in results if r['is_matching']]
    non_matching = [r for r in results if not r['is_matching']]

    summary = {
        'pearson_correlation_signal_vs_quality': pearson_r,
        'mean_loss_matching': float(np.mean([r['loss'] for r in matching])) if matching else 0,
        'mean_loss_non_matching': float(np.mean([r['loss'] for r in non_matching])) if non_matching else 0,
        'mean_signal_matching': float(np.mean([r['combined_signal'] for r in matching])) if matching else 0,
        'mean_signal_non_matching': float(np.mean([r['combined_signal'] for r in non_matching])) if non_matching else 0,
        'n_evaluations': len(results),
    }

    print(f'\n  Pearson correlation (signal vs quality): {pearson_r:.3f}')
    print(f'  Mean loss (matching specialist): {summary["mean_loss_matching"]:.4f}')
    print(f'  Mean loss (non-matching):        {summary["mean_loss_non_matching"]:.4f}')

    return {'evaluations': results, 'summary': summary}


# ============================================================
# Pass 4: Composition Evaluation
# ============================================================

def run_pass4_composition(generalist_sd, specialist_dirs, model_cfg, probe_set,
                          index, device='cpu', normalize='two_stage'):
    """Evaluate composed model against baselines."""
    print('\n' + '=' * 60)
    print(f'PASS 4: Composition Evaluation (normalize={normalize})')
    print('=' * 60)

    domain_labels = probe_set['domain_labels']

    # Load all specialist state dicts
    specialist_sds = {}
    for domain in DOMAINS:
        if domain == 'general':
            continue
        spec_path = specialist_dirs / domain / 'specialist.pt'
        if not spec_path.exists():
            continue
        ckpt = torch.load(spec_path, weights_only=False, map_location='cpu')
        specialist_sds[domain] = {k: v.float() for k, v in ckpt['state_dict'].items()}

    if not specialist_sds:
        print('  No specialists found. Skipping.')
        return {'evaluations': [], 'summary': {}}

    # Get generalist fingerprints — create ONE reusable model instance
    reuse_model = _get_model_cls()(model_cfg)
    gen_sd_float = {k: v.float() for k, v in generalist_sd.items()}
    reuse_model.load_state_dict(gen_sd_float)
    reuse_model.to(device)

    gen_fp = capture_at_checkpoint(reuse_model, probe_set, ['final_hidden'],
                                    device=str(device), batch_size=32)
    gen_activations = gen_fp['final_hidden']

    # Baselines: generalist loss per probe
    generalist_losses = []
    for i in range(NUM_PROBES):
        val_batch = {
            'input_ids': probe_set['input_ids'][i:i+1].to(device),
            'attention_mask': probe_set['attention_mask'][i:i+1].to(device),
        }
        generalist_losses.append(compute_val_loss(reuse_model, val_batch))

    # Uniform average baseline — reuse same model instance
    uniform_sd = {k: torch.zeros_like(v, dtype=torch.float32)
                  for k, v in gen_sd_float.items()}
    for k in uniform_sd:
        total = gen_sd_float[k].float()
        count = 1
        for sid, ssd in specialist_sds.items():
            if k in ssd:
                total = total + ssd[k].float()
                count += 1
        uniform_sd[k] = total / count

    reuse_model.load_state_dict(uniform_sd)
    uniform_losses = []
    for i in range(NUM_PROBES):
        val_batch = {
            'input_ids': probe_set['input_ids'][i:i+1].to(device),
            'attention_mask': probe_set['attention_mask'][i:i+1].to(device),
        }
        uniform_losses.append(compute_val_loss(reuse_model, val_batch))

    results = []
    t0 = time.time()

    for i in range(NUM_PROBES):
        true_domain = domain_labels[i]
        query_fp = gen_activations[i]

        # Step 1: Get correlation scores
        corr_scores = compute_correlation_scores(query_fp, index)

        # Select top-k specialists
        ranked = sorted(corr_scores.items(), key=lambda x: x[1], reverse=True)
        top_k = ranked[:TOP_K_SPECIALISTS]
        selected = {sid: score for sid, score in top_k}

        # Step 2: Get activation signatures for selected specialists
        act_sigs = {}
        for sid in selected:
            reuse_model.load_state_dict(specialist_sds[sid])
            act_sigs[sid] = get_activation_signature(
                reuse_model,
                probe_set['input_ids'][i:i+1],
                probe_set['attention_mask'][i:i+1],
                device=str(device),
            )

        # Step 3: Compute node-level weights
        node_w = compute_node_weights(selected, act_sigs, normalize=normalize)
        entropy = weight_entropy(node_w)

        # Step 4: Compose parameters
        selected_sds = {sid: specialist_sds[sid] for sid in selected}
        composed_sd = compose_parameters(gen_sd_float, selected_sds, node_w)

        # Step 5: Evaluate composed model — reuse same model instance
        reuse_model.load_state_dict(composed_sd)
        val_batch = {
            'input_ids': probe_set['input_ids'][i:i+1].to(device),
            'attention_mask': probe_set['attention_mask'][i:i+1].to(device),
        }
        composed_loss = compute_val_loss(reuse_model, val_batch)

        # Oracle specialist loss
        oracle_loss = float('inf')
        if true_domain in specialist_sds and true_domain != 'general':
            reuse_model.load_state_dict(specialist_sds[true_domain])
            oracle_loss = compute_val_loss(reuse_model, val_batch)

        results.append({
            'probe_idx': i,
            'true_domain': true_domain,
            'selected_specialists': list(selected.keys()),
            'correlation_scores': {k: float(v) for k, v in selected.items()},
            'weight_entropy': entropy,
            'generalist_loss': generalist_losses[i],
            'composed_loss': composed_loss,
            'uniform_loss': uniform_losses[i],
            'oracle_loss': oracle_loss if oracle_loss < float('inf') else None,
            'composed_beats_generalist': composed_loss < generalist_losses[i],
            'composed_beats_uniform': composed_loss < uniform_losses[i],
            'composed_beats_oracle': composed_loss < oracle_loss,
        })

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (NUM_PROBES - i - 1) / rate if rate > 0 else 0
            print(f'  Evaluated {i+1}/{NUM_PROBES} probes ({elapsed:.0f}s, ETA {eta:.0f}s)')

    del reuse_model
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    # Summary
    non_general = [r for r in results if r['true_domain'] != 'general']
    summary = {
        'normalize': normalize,
        'win_rate_vs_generalist': float(np.mean([r['composed_beats_generalist'] for r in non_general])),
        'win_rate_vs_uniform': float(np.mean([r['composed_beats_uniform'] for r in non_general])),
        'win_rate_vs_oracle': float(np.mean([r['composed_beats_oracle'] for r in non_general])),
        'mean_composed_loss': float(np.mean([r['composed_loss'] for r in non_general])),
        'mean_generalist_loss': float(np.mean([r['generalist_loss'] for r in non_general])),
        'mean_uniform_loss': float(np.mean([r['uniform_loss'] for r in non_general])),
        'mean_weight_entropy': float(np.mean([r['weight_entropy'] for r in non_general])),
    }

    print(f'\n  Win rate vs generalist: {summary["win_rate_vs_generalist"]:.1%}')
    print(f'  Win rate vs uniform:    {summary["win_rate_vs_uniform"]:.1%}')
    print(f'  Win rate vs oracle:     {summary["win_rate_vs_oracle"]:.1%}')
    print(f'  Mean composed loss:     {summary["mean_composed_loss"]:.4f}')
    print(f'  Mean generalist loss:   {summary["mean_generalist_loss"]:.4f}')
    print(f'  Mean weight entropy:    {summary["mean_weight_entropy"]:.2f} bits')

    return {'evaluations': results, 'summary': summary}


# ============================================================
# Cross-Domain Probes
# ============================================================

# Each probe is designed to genuinely require expertise from exactly 2 domains.
# The expected_domains tuple lists the two relevant specialist domains.
# These are NOT ambiguous queries — they have clear dual-domain structure.
CROSS_DOMAIN_PROBES = [
    # medical + legal
    ("Medical malpractice claims require proving that the physician's treatment deviated from the accepted standard of care, which in cases involving off-label drug prescription means demonstrating that no reasonable physician would have prescribed",
     ('medical', 'legal')),
    ("The FDA regulatory pathway for emergency use authorization of vaccines differs from full approval in that clinical trial data requirements for safety and efficacy are",
     ('medical', 'legal')),
    ("Informed consent doctrine requires physicians to disclose material risks of a procedure, where materiality is determined by whether a reasonable patient would consider the information significant when deciding whether to undergo",
     ('medical', 'legal')),
    ("HIPAA privacy regulations govern the disclosure of protected health information by covered entities, with exceptions for public health surveillance requiring notification when a patient presents with",
     ('medical', 'legal')),
    ("Expert witness testimony in toxic tort litigation must establish both general causation showing the substance can cause the disease and specific causation showing the plaintiff's exposure to the",
     ('medical', 'legal')),

    # medical + science
    ("The CRISPR-Cas9 system has shown therapeutic potential for treating sickle cell disease by editing the BCL11A enhancer region to reactivate fetal hemoglobin production in hematopoietic stem cells",
     ('medical', 'science')),
    ("Phase III clinical trials of mRNA vaccines demonstrated that the spike protein encoded by the modified nucleoside RNA elicited neutralizing antibody titers that correlated with",
     ('medical', 'science')),
    ("Pharmacogenomics uses genome-wide association studies to identify single nucleotide polymorphisms that predict adverse drug reactions, allowing personalized dosing of medications such as",
     ('medical', 'science')),
    ("Magnetic resonance spectroscopy detects metabolite concentrations in brain tissue by measuring the precession frequencies of hydrogen nuclei in different chemical environments, which in patients with glioblastoma typically shows elevated choline and decreased",
     ('medical', 'science')),
    ("Monoclonal antibody therapeutics are engineered using recombinant DNA technology to target specific epitopes on tumor cells, with chimeric antigen receptor T-cell therapy representing a more personalized approach where the patient's own lymphocytes are modified to express",
     ('medical', 'science')),

    # legal + code
    ("Software patent claims under the Alice framework must demonstrate that the abstract idea is implemented with an inventive concept, which for machine learning algorithms means showing that the specific implementation of gradient descent optimization provides a technical improvement over",
     ('legal', 'code')),
    ("The Digital Millennium Copyright Act safe harbor provisions protect platform operators from liability for user-uploaded content when they implement automated content identification systems using hash-matching algorithms that compare uploaded files against",
     ('legal', 'code')),
    ("Data protection regulations under GDPR require implementing privacy by design, which for database systems means incorporating access control lists, encryption at rest using AES-256, and automated deletion procedures triggered by",
     ('legal', 'code')),
    ("Open source license compliance in corporate software development requires tracking transitive dependencies in package managers, where a single GPL-licensed library in the dependency tree can impose copyleft obligations on",
     ('legal', 'code')),
    ("Algorithmic fairness regulations require that automated decision systems used in lending must demonstrate statistical parity across protected classes, which can be implemented by adding fairness constraints to the loss function during model training such as",
     ('legal', 'code')),

    # science + code
    ("Implementing the Runge-Kutta method for solving ordinary differential equations in numerical simulations of planetary orbits requires careful handling of floating-point precision to maintain energy conservation over",
     ('science', 'code')),
    ("Monte Carlo methods for estimating integrals in high-dimensional spaces use pseudorandom number generators with the Mersenne Twister algorithm providing a period of 2^19937-1 which is sufficient for simulating",
     ('science', 'code')),
    ("Molecular dynamics simulations using the Verlet integration algorithm compute pairwise interatomic forces according to the Lennard-Jones potential, with neighbor list optimizations reducing computational complexity from O(N^2) to",
     ('science', 'code')),
    ("Finite element analysis of stress distributions in materials requires assembling sparse stiffness matrices that can be efficiently solved using conjugate gradient methods with incomplete Cholesky preconditioning for",
     ('science', 'code')),
    ("Training neural network potentials for quantum chemistry calculations involves fitting to density functional theory energy surfaces, where the architecture uses message-passing layers to maintain rotational equivariance of the predicted",
     ('science', 'code')),

    # medical + code
    ("Electronic health record systems implement HL7 FHIR APIs for interoperability, where patient medication lists are represented as MedicationStatement resources that reference standardized drug codes from RxNorm and can be queried using",
     ('medical', 'code')),
    ("Medical image segmentation using U-Net architectures requires training on annotated CT scans where the loss function combines cross-entropy for pixel-wise classification with Dice coefficient to handle class imbalance between tumor and healthy tissue",
     ('medical', 'code')),
    ("Clinical decision support systems use Bayesian networks to compute posterior probabilities of diagnoses given observed symptoms, where the directed acyclic graph structure encodes conditional independence assumptions and inference is performed using",
     ('medical', 'code')),
    ("Implementing a drug interaction checker requires parsing SMILES molecular representations and comparing computed pharmacophore fingerprints against a database of known interactions, flagging combinations where the cytochrome P450 metabolic pathway overlap exceeds",
     ('medical', 'code')),
    ("Wearable biosensor data pipelines process continuous glucose monitor readings at 5-minute intervals, applying Kalman filters to estimate true blood glucose from noisy sensor measurements and triggering insulin pump adjustments when predicted levels exceed",
     ('medical', 'code')),

    # legal + science
    ("Patent prosecution for biotechnology inventions requires demonstrating that gene sequences are markedly different from naturally occurring DNA, following the Supreme Court's ruling in Association for Molecular Pathology v. Myriad Genetics which held that",
     ('legal', 'science')),
    ("Environmental impact assessments under NEPA must quantify the atmospheric dispersion of pollutants using Gaussian plume models that account for wind speed, atmospheric stability class, and stack height to determine whether emissions of particulate matter exceed",
     ('legal', 'science')),
    ("The Daubert standard for admissibility of scientific expert testimony requires courts to evaluate whether the methodology is testable, peer-reviewed, has known error rates, and is generally accepted, which for statistical evidence means assessing whether the confidence interval and p-value of",
     ('legal', 'science')),
    ("Carbon credit trading under cap-and-trade regulations requires verified measurement of greenhouse gas emissions reductions, where direct air capture facilities must demonstrate net removal using lifecycle analysis that accounts for the energy consumed by the amine sorbent regeneration process at",
     ('legal', 'science')),
    ("Regulation of dual-use research of concern in gain-of-function virology requires institutional biosafety committees to evaluate whether the proposed modifications to pathogen transmissibility provide sufficient scientific benefit to justify the enhanced pandemic potential when the research involves",
     ('legal', 'science')),
]


# ============================================================
# Pass 5: Cross-Domain Evaluation
# ============================================================

def run_pass5_cross_domain(generalist_sd, specialist_dirs, model_cfg, index,
                            tokenizer, device='cpu', normalize='two_stage'):
    """Evaluate composition on queries requiring multiple domain expertise."""
    print('\n' + '=' * 60)
    print(f'PASS 5: Cross-Domain Evaluation (normalize={normalize})')
    print('=' * 60)

    # Load specialist state dicts
    specialist_sds = {}
    for domain in DOMAINS:
        if domain == 'general':
            continue
        spec_path = specialist_dirs / domain / 'specialist.pt'
        if not spec_path.exists():
            continue
        ckpt = torch.load(spec_path, weights_only=False, map_location='cpu')
        specialist_sds[domain] = {k: v.float() for k, v in ckpt['state_dict'].items()}

    if not specialist_sds:
        print('  No specialists found. Skipping.')
        return {'evaluations': [], 'summary': {}}

    # Tokenize cross-domain probes (use extended set if available)
    probe_list = EXTENDED_CROSS_DOMAIN_PROBES if _USE_EXTENDED_PROBES else CROSS_DOMAIN_PROBES
    print(f'  Using {"extended" if _USE_EXTENDED_PROBES else "built-in"} probes: {len(probe_list)} total')
    texts = [p[0] for p in probe_list]
    expected_domains = [p[1] for p in probe_list]

    encoded = tokenizer(
        texts, padding='max_length', truncation=True,
        max_length=128, return_tensors='pt',
    )
    cross_probe_set = {
        'input_ids': encoded['input_ids'],
        'attention_mask': encoded['attention_mask'],
    }

    gen_sd_float = {k: v.float() for k, v in generalist_sd.items()}

    # Create ONE reusable model instance for all evaluations
    reuse_model = _get_model_cls()(model_cfg)
    reuse_model.load_state_dict(gen_sd_float)
    reuse_model.to(device)

    gen_fp = capture_at_checkpoint(reuse_model, cross_probe_set, ['final_hidden'],
                                    device=str(device), batch_size=32)
    gen_activations = gen_fp['final_hidden']

    generalist_losses = []
    for i in range(len(texts)):
        val_batch = {
            'input_ids': cross_probe_set['input_ids'][i:i+1].to(device),
            'attention_mask': cross_probe_set['attention_mask'][i:i+1].to(device),
        }
        generalist_losses.append(compute_val_loss(reuse_model, val_batch))

    results = []
    t0 = time.time()
    for i, (text, (dom_a, dom_b)) in enumerate(probe_list):
        query_fp = gen_activations[i]

        # Get correlation scores — we expect both expected domains to rank high
        corr_scores = compute_correlation_scores(query_fp, index)
        ranked = sorted(corr_scores.items(), key=lambda x: x[1], reverse=True)

        # Check if both expected domains are in top-3
        top3 = [r[0] for r in ranked[:3]]
        both_in_top3 = dom_a in top3 and dom_b in top3

        # Rank of each expected domain
        rank_a = next((j for j, (sid, _) in enumerate(ranked) if sid == dom_a), -1)
        rank_b = next((j for j, (sid, _) in enumerate(ranked) if sid == dom_b), -1)

        # Compose with top-3 specialists
        selected = {sid: score for sid, score in ranked[:TOP_K_SPECIALISTS]}

        act_sigs = {}
        for sid in selected:
            if sid not in specialist_sds:
                continue
            reuse_model.load_state_dict(specialist_sds[sid])
            act_sigs[sid] = get_activation_signature(
                reuse_model,
                cross_probe_set['input_ids'][i:i+1],
                cross_probe_set['attention_mask'][i:i+1],
                device=str(device),
            )

        # Filter selected to only those with activation signatures
        selected = {sid: score for sid, score in selected.items() if sid in act_sigs}

        if not selected:
            continue

        node_w = compute_node_weights(selected, act_sigs, normalize=normalize)
        entropy = weight_entropy(node_w)

        selected_sds = {sid: specialist_sds[sid] for sid in selected}
        composed_sd = compose_parameters(gen_sd_float, selected_sds, node_w)

        reuse_model.load_state_dict(composed_sd)
        val_batch = {
            'input_ids': cross_probe_set['input_ids'][i:i+1].to(device),
            'attention_mask': cross_probe_set['attention_mask'][i:i+1].to(device),
        }
        composed_loss = compute_val_loss(reuse_model, val_batch)

        # Task arithmetic baseline: uniform alpha across selected specialists
        ta_losses = {}
        for ta_alpha in [0.3, 0.5, 1.0]:
            ta_sd = task_arithmetic_compose(gen_sd_float, selected_sds, alpha=ta_alpha)
            reuse_model.load_state_dict(ta_sd)
            ta_losses[ta_alpha] = compute_val_loss(reuse_model, val_batch)
        best_ta_alpha = min(ta_losses, key=ta_losses.get)
        best_ta_loss = ta_losses[best_ta_alpha]

        # Individual specialist losses for comparison
        specialist_losses = {}
        for sid in [dom_a, dom_b]:
            if sid in specialist_sds:
                reuse_model.load_state_dict(specialist_sds[sid])
                specialist_losses[sid] = compute_val_loss(reuse_model, val_batch)

        # Best single specialist for this query
        best_single = min(specialist_losses.values()) if specialist_losses else float('inf')

        results.append({
            'probe_idx': i,
            'text_preview': text[:80],
            'expected_domains': [dom_a, dom_b],
            'both_in_top3': both_in_top3,
            'rank_domain_a': rank_a,
            'rank_domain_b': rank_b,
            'selected_specialists': list(selected.keys()),
            'correlation_scores': {k: float(v) for k, v in corr_scores.items()},
            'weight_entropy': entropy,
            'generalist_loss': generalist_losses[i],
            'composed_loss': composed_loss,
            'task_arithmetic_losses': {str(k): float(v) for k, v in ta_losses.items()},
            'best_task_arithmetic_loss': best_ta_loss,
            'best_task_arithmetic_alpha': best_ta_alpha,
            'specialist_losses': {k: float(v) for k, v in specialist_losses.items()},
            'best_single_specialist_loss': best_single if best_single < float('inf') else None,
            'composed_beats_generalist': composed_loss < generalist_losses[i],
            'composed_beats_best_single': composed_loss < best_single,
            'composed_beats_task_arithmetic': composed_loss < best_ta_loss,
            'task_arithmetic_beats_generalist': best_ta_loss < generalist_losses[i],
        })

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(probe_list) - i - 1) / rate if rate > 0 else 0
            print(f'  Evaluated {i+1}/{len(probe_list)} cross-domain probes ({elapsed:.0f}s, ETA {eta:.0f}s)')

    del reuse_model
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    # Summary
    summary = {
        'n_probes': len(results),
        'both_domains_in_top3_rate': float(np.mean([r['both_in_top3'] for r in results])) if results else 0,
        'win_rate_vs_generalist': float(np.mean([r['composed_beats_generalist'] for r in results])) if results else 0,
        'win_rate_vs_best_single': float(np.mean([r['composed_beats_best_single'] for r in results])) if results else 0,
        'mean_composed_loss': float(np.mean([r['composed_loss'] for r in results])) if results else 0,
        'mean_generalist_loss': float(np.mean([r['generalist_loss'] for r in results])) if results else 0,
        'mean_weight_entropy': float(np.mean([r['weight_entropy'] for r in results])) if results else 0,
        'mean_best_task_arithmetic_loss': float(np.mean([r['best_task_arithmetic_loss'] for r in results])) if results else 0,
        'constellation_win_rate_vs_task_arithmetic': float(np.mean([r['composed_beats_task_arithmetic'] for r in results])) if results else 0,
        'task_arithmetic_win_rate_vs_generalist': float(np.mean([r['task_arithmetic_beats_generalist'] for r in results])) if results else 0,
    }

    # Per-domain-pair breakdown
    pair_results = {}
    for r in results:
        pair = tuple(sorted(r['expected_domains']))
        pair_key = f'{pair[0]}+{pair[1]}'
        if pair_key not in pair_results:
            pair_results[pair_key] = []
        pair_results[pair_key].append(r)

    summary['by_pair'] = {}
    for pair_key, pr in pair_results.items():
        summary['by_pair'][pair_key] = {
            'n': len(pr),
            'both_in_top3': float(np.mean([r['both_in_top3'] for r in pr])),
            'win_vs_generalist': float(np.mean([r['composed_beats_generalist'] for r in pr])),
            'win_vs_best_single': float(np.mean([r['composed_beats_best_single'] for r in pr])),
            'constellation_vs_ta': float(np.mean([r['composed_beats_task_arithmetic'] for r in pr])),
        }

    print(f'\n  Both expected domains in top-3: {summary["both_domains_in_top3_rate"]:.1%}')
    print(f'  Win rate vs generalist:        {summary["win_rate_vs_generalist"]:.1%}')
    print(f'  Win rate vs best single:       {summary["win_rate_vs_best_single"]:.1%}')
    print(f'  Win rate vs task arithmetic:   {summary["constellation_win_rate_vs_task_arithmetic"]:.1%}')
    print(f'  Task arith vs generalist:      {summary["task_arithmetic_win_rate_vs_generalist"]:.1%}')
    print(f'  Mean weight entropy:           {summary["mean_weight_entropy"]:.2f} bits')
    for pk, ps in summary['by_pair'].items():
        print(f'    {pk:20s}: top3={ps["both_in_top3"]:.0%}, '
              f'vs_gen={ps["win_vs_generalist"]:.0%}, vs_single={ps["win_vs_best_single"]:.0%}, '
              f'vs_ta={ps["constellation_vs_ta"]:.0%}')

    return {'evaluations': results, 'summary': summary}


# ============================================================
# Model Class Helper
# ============================================================

_MODEL_CLS = None
_MODEL_NAME = None  # For from_pretrained-style models

def _get_model_cls():
    """Returns a callable(config) -> model.

    For GPT2: GPT2LMHeadModel(config) works directly.
    For Auto models: wraps from_config(config).
    """
    global _MODEL_CLS
    if _MODEL_CLS is None:
        _MODEL_CLS = GPT2LMHeadModel
    return _MODEL_CLS

def _set_model_cls(cls):
    """Set model class. For AutoModelForCausalLM, wraps with from_config."""
    global _MODEL_CLS
    if cls is AutoModelForCausalLM:
        _MODEL_CLS = lambda cfg: AutoModelForCausalLM.from_config(cfg)
    else:
        _MODEL_CLS = cls


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Paper 3: Constellation-Indexed Model Composition')
    parser.add_argument('--train-specialists', action='store_true')
    parser.add_argument('--run-all', action='store_true')
    parser.add_argument('--pass', type=int, dest='run_pass', default=0)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--use-amp', action='store_true')
    parser.add_argument('--generalist-ckpt', type=str,
                        default='results/phase2c/seed42/checkpoints/step_02005.pt')
    parser.add_argument('--output-dir', type=str, default='results/paper3')
    parser.add_argument('--normalize', type=str, default='two_stage',
                        choices=['joint', 'two_stage'])
    parser.add_argument('--model', type=str, default='gpt2',
                        help='Model name: gpt2, Qwen/Qwen2.5-1.5B, etc.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    import random; random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    specialist_dirs = output_dir / 'specialists'

    device = args.device

    print('Paper 3: Constellation-Indexed Model Composition')
    print(f'Model: {args.model}')
    print(f'Device: {device}, AMP: {args.use_amp}')
    print(f'Generalist: {args.generalist_ckpt}')
    print(f'Output: {output_dir}')

    # Configure model class based on --model
    if args.model == 'gpt2':
        _set_model_cls(GPT2LMHeadModel)
        model_cfg = AutoConfig.from_pretrained('gpt2')
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
    else:
        # For any HuggingFace model, use AutoModelForCausalLM
        _set_model_cls(AutoModelForCausalLM)
        model_cfg = AutoConfig.from_pretrained(args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load generalist state dict
    gen_ckpt = torch.load(args.generalist_ckpt, weights_only=False, map_location='cpu')
    generalist_sd = gen_ckpt['state_dict']

    # Tokenize domain probes
    print('\nTokenizing domain probes...')
    probe_set = tokenize_domain_probes(tokenizer)
    print(f'  {len(probe_set["texts"])} probes across {len(DOMAINS)} domains')

    # Train specialists
    if args.train_specialists:
        specialist_dirs.mkdir(parents=True, exist_ok=True)
        for domain in DOMAINS:
            if domain == 'general':
                continue
            train_specialist(domain, args.generalist_ckpt, model_cfg, tokenizer,
                             specialist_dirs, device=device, use_amp=args.use_amp)

    # Run passes
    run_passes = []
    if args.run_all:
        run_passes = [1, 2, 3, 4, 5]
    elif args.run_pass > 0:
        run_passes = [args.run_pass]

    all_results = {}

    if 1 in run_passes:
        index = run_pass1_index(specialist_dirs, model_cfg, probe_set, device=device)
        # Save index
        index_save = {k: [fp.tolist() for fp in fps] for k, fps in index.items()}
        with open(output_dir / 'pass1_index.json', 'w') as f:
            json.dump(index_save, f, indent=2)
        all_results['pass1'] = {'index_size': {k: len(v) for k, v in index.items()}}
    else:
        # Load existing index
        index_path = output_dir / 'pass1_index.json'
        if index_path.exists():
            with open(index_path) as f:
                index_raw = json.load(f)
            index = {k: [np.array(fp) for fp in fps] for k, fps in index_raw.items()}
        else:
            index = None

    if 2 in run_passes and index:
        pass2 = run_pass2_retrieval(generalist_sd, model_cfg, probe_set, index, device=device)
        with open(output_dir / 'pass2_retrieval.json', 'w') as f:
            json.dump(pass2, f, indent=2, default=str)
        all_results['pass2'] = pass2['summary']

    if 3 in run_passes and index:
        pass3 = run_pass3_correlation(generalist_sd, specialist_dirs, model_cfg,
                                       probe_set, index, device=device)
        with open(output_dir / 'pass3_correlation.json', 'w') as f:
            json.dump(pass3, f, indent=2, default=str)
        all_results['pass3'] = pass3['summary']

    if 4 in run_passes and index:
        pass4 = run_pass4_composition(generalist_sd, specialist_dirs, model_cfg,
                                       probe_set, index, device=device,
                                       normalize=args.normalize)
        with open(output_dir / 'pass4_composition.json', 'w') as f:
            json.dump(pass4, f, indent=2, default=str)
        all_results['pass4'] = pass4['summary']

    if 5 in run_passes and index:
        pass5 = run_pass5_cross_domain(generalist_sd, specialist_dirs, model_cfg,
                                        index, tokenizer, device=device,
                                        normalize=args.normalize)
        with open(output_dir / 'pass5_cross_domain.json', 'w') as f:
            json.dump(pass5, f, indent=2, default=str)
        all_results['pass5'] = pass5['summary']

    # Save combined results
    with open(output_dir / 'paper3_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print('\n' + '=' * 60)
    print('DONE')
    print(f'Results saved to {output_dir}')
    print('=' * 60)


if __name__ == '__main__':
    main()
