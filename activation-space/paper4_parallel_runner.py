"""Paper 4: Parallel INLP experiment runner.

Captures activations for all scales, then runs all noise variants in parallel
using multiprocessing. Designed for high-core-count machines.

Usage: python3 -u paper4_parallel_runner.py
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import sys
import json
import time
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

SHAPES = ['hierarchical', 'causal', 'constraint', 'evidence']
DOMAINS = ['medical', 'legal', 'code', 'science']

HIERARCHICAL_PROBES = {
    'medical': [
        "Classify the subtypes of non-Hodgkin lymphoma by cell of origin and clinical behavior.",
        "Break down the components of the renin-angiotensin-aldosterone system and their regulatory relationships.",
        "Categorize the types of bone fractures by mechanism, location, and displacement pattern.",
        "Decompose the differential diagnosis of chest pain by organ system and acuity.",
        "Classify antibiotics by mechanism of action, spectrum, and bacterial target.",
        "Organize the branches of the brachial plexus and the muscles each branch innervates.",
        "Categorize types of anemia by MCV, reticulocyte count, and underlying mechanism.",
        "Break down the WHO classification of brain tumors by grade, histology, and molecular markers.",
        "Classify hypersensitivity reactions by Gell-Coombs type and provide the mechanism for each.",
        "Decompose the stages of wound healing and the cell types active in each stage.",
    ],
    'legal': [
        "Break down the elements of a negligence claim and the defenses available at each element.",
        "Classify the types of intellectual property protection by scope, duration, and registration requirements.",
        "Decompose the hierarchy of legal authority from constitutional provisions to agency guidance.",
        "Categorize the exceptions to the hearsay rule by their reliability rationale.",
        "Break down the types of business entities by liability, taxation, and governance structure.",
        "Classify the categories of speech protection under First Amendment doctrine.",
        "Decompose the elements of a breach of contract claim and available remedies at each stage.",
        "Categorize the types of criminal intent from strict liability through specific intent.",
        "Break down the taxonomy of regulatory takings from per se to Penn Central balancing.",
        "Classify the forms of alternative dispute resolution by formality, binding nature, and typical use.",
    ],
    'code': [
        "Classify design patterns by creational, structural, and behavioral categories with examples.",
        "Break down the layers of the OSI networking model and the protocols at each layer.",
        "Decompose a compiler into its phases and the data structures each phase produces.",
        "Categorize database index types by structure, use case, and performance characteristics.",
        "Classify sorting algorithms by time complexity, space complexity, and stability.",
        "Break down the components of a Kubernetes cluster by control plane and worker node roles.",
        "Decompose the types of software testing from unit to end-to-end by scope and purpose.",
        "Categorize concurrency primitives by mechanism: locks, semaphores, channels, and STM.",
        "Classify memory management strategies from manual allocation through garbage collection approaches.",
        "Break down the taxonomy of SQL joins by type and the result set each produces.",
    ],
    'science': [
        "Classify the fundamental forces of nature by strength, range, and carrier particle.",
        "Break down the taxonomic hierarchy from domain to species using Homo sapiens as example.",
        "Decompose the electromagnetic spectrum by wavelength, energy, and common applications.",
        "Categorize chemical bonds by type: ionic, covalent, metallic, and intermolecular forces.",
        "Classify types of volcanic eruptions by explosivity, magma composition, and landform produced.",
        "Break down the stages of stellar evolution by mass and the end states for each path.",
        "Decompose the layers of Earth's atmosphere by temperature profile and key phenomena.",
        "Categorize organic reaction mechanisms by type: substitution, elimination, addition, and rearrangement.",
        "Classify types of spectrometry by the physical principle and the information each provides.",
        "Break down the components of a eukaryotic cell by organelle function and membrane structure.",
    ],
}

CAUSAL_PROBES = {
    'medical': ["Trace the pathophysiological cascade from deep vein thrombosis to pulmonary embolism to right heart failure.","Explain the chain of events from H. pylori infection through chronic gastritis to gastric adenocarcinoma.","Follow the progression from insulin resistance through compensatory hyperinsulinemia to beta cell exhaustion in type 2 diabetes.","Trace how untreated streptococcal pharyngitis leads to rheumatic fever through molecular mimicry.","Describe the cascade from atherosclerotic plaque rupture through thrombus formation to myocardial infarction.","Follow the sequence from portal hypertension through varices formation to variceal hemorrhage.","Trace the chain from chronic hypertension through nephrosclerosis to end-stage renal disease.","Explain how a mutation in the CFTR gene leads through chloride channel dysfunction to the clinical manifestations of cystic fibrosis.","Follow the cascade from traumatic brain injury through secondary injury mechanisms to long-term neurological deficit.","Trace the progression from chronic hepatitis C through fibrosis stages to cirrhosis and hepatocellular carcinoma."],
    'legal': ["Trace how a single SEC filing omission cascades through materiality analysis to securities fraud liability.","Follow the chain from a police officer's warrantless search through the exclusionary rule to case dismissal.","Explain how a manufacturer's design defect leads through failure to warn to strict product liability.","Trace the sequence from a constitutional amendment's proposal through ratification to judicial enforcement.","Follow the chain from a breach of fiduciary duty through damages to disgorgement of profits.","Describe how precedent from a circuit court decision propagates through subsequent cases to a circuit split to Supreme Court review.","Trace how a zoning variance denial cascades through administrative appeal to judicial review under the arbitrary and capricious standard.","Follow the sequence from discriminatory intent through disparate impact evidence to a Title VII violation finding.","Explain how a contract clause's ambiguity leads through parol evidence consideration to reformation.","Trace the chain from an environmental violation through EPA enforcement action to consent decree to ongoing compliance monitoring."],
    'code': ["Trace how a buffer overflow in user input leads through stack corruption to arbitrary code execution.","Follow the chain from a DNS misconfiguration through failed certificate validation to a man-in-the-middle vulnerability.","Explain how a memory leak in a long-running service leads through heap exhaustion to cascading service failures.","Trace the sequence from a race condition in a database write through inconsistent state to data corruption.","Follow the cascade from a single microservice failure through timeout propagation to full system brownout.","Describe how an N+1 query pattern leads through connection pool exhaustion to database lock contention to user-visible latency.","Trace how a floating-point rounding error propagates through iterative computation to produce a catastrophically wrong result.","Follow the chain from a misconfigured load balancer through uneven traffic distribution to hot-spot failure.","Explain how a dependency version conflict leads through incompatible API calls to a runtime crash in production.","Trace the sequence from a Git merge conflict through an incorrect resolution to a regression that passes CI but fails in production."],
    'science': ["Trace the chain from increased atmospheric CO2 through radiative forcing to ocean acidification and coral bleaching.","Follow the sequence from a supernova explosion through nucleosynthesis to the formation of heavy elements found on Earth.","Explain how a single base-pair mutation leads through protein misfolding to prion propagation and neurodegeneration.","Trace the cascade from ozone depletion through increased UV radiation to increased mutation rates in surface organisms.","Follow the chain from tectonic plate subduction through magma generation to volcanic eruption and atmospheric effects.","Describe how a perturbation in initial conditions leads through chaotic amplification to divergent weather outcomes (butterfly effect).","Trace the sequence from photon absorption in a semiconductor through electron-hole pair generation to current flow in a solar cell.","Follow the cascade from antibiotic overuse through selection pressure to resistance gene horizontal transfer to multi-drug resistant organisms.","Explain how Milankovitch cycle variations lead through insolation changes to ice sheet advance and retreat.","Trace the chain from quantum tunneling in hydrogen fusion through radiation pressure to stellar equilibrium."],
}

CONSTRAINT_PROBES = {
    'medical': ["A patient has renal failure, a penicillin allergy, and a methicillin-resistant staph infection. Choose an antibiotic regimen that addresses all three constraints.","Design a post-surgical pain management plan for a patient with a history of opioid addiction, hepatic impairment, and NSAID-induced gastric ulcers.","A pregnant patient in her first trimester presents with new-onset epilepsy. Select an anticonvulsant that minimizes teratogenicity while controlling seizures.","Choose an antihypertensive for a diabetic patient with bilateral renal artery stenosis and hyperkalemia.","A patient needs anticoagulation for atrial fibrillation but has a recent GI bleed and thrombocytopenia. Design a management strategy.","Select an anesthetic protocol for a patient with malignant hyperthermia susceptibility, difficult airway, and severe cardiac disease.","Design a chemotherapy regimen for a cancer patient with pre-existing cardiomyopathy, neuropathy, and neutropenia.","A patient with Parkinson's disease needs an antiemetic but cannot receive dopamine antagonists. Identify appropriate options.","Choose a contraceptive method for a patient with a history of DVT, migraine with aura, and liver disease.","Design a nutritional plan for a critically ill patient with diabetes, celiac disease, and severe nut allergy."],
    'legal': ["Draft a contract clause that satisfies consumer protection disclosure requirements, company IP protection needs, and GDPR data processing constraints simultaneously.","Structure a corporate transaction that achieves tax-free reorganization status while maintaining regulatory approval and satisfying minority shareholder rights.","Design a sentencing recommendation that accounts for mandatory minimums, mitigating factors, and proportionality requirements under the Eighth Amendment.","Construct a trust that satisfies the rule against perpetuities, achieves the desired estate tax treatment, and protects assets from the beneficiary's creditors.","Draft an employment agreement that complies with non-compete enforceability requirements, trade secret protection, and employee mobility rights in California.","Structure a plea agreement that satisfies the victim's restitution rights, the defendant's constitutional protections, and the prosecution's sentencing guidelines obligations.","Design a regulatory compliance program that simultaneously meets SEC, FINRA, and state blue sky law requirements for a multi-state securities offering.","Draft a licensing agreement that satisfies open-source copyleft obligations, proprietary code protection, and export control restrictions.","Structure a bankruptcy reorganization plan that satisfies absolute priority, feasibility, and best interests of creditors tests simultaneously.","Design a class action settlement that meets adequacy of representation, commonality, and fairness hearing requirements."],
    'code': ["Design a database schema that satisfies third normal form, supports the required query patterns without joins, and stays within the storage budget.","Architect a system that achieves sub-100ms latency, 99.99% availability, and strong consistency across three geographic regions.","Write a scheduling algorithm that respects task dependencies, meets all deadlines, and minimizes total resource usage.","Design an API rate limiting system that prevents abuse, allows burst traffic from legitimate users, and maintains fairness across tenants.","Architect a data pipeline that preserves exactly-once processing semantics, handles schema evolution, and scales horizontally.","Design a cache eviction policy that maximizes hit rate, respects memory limits, and guarantees freshness for time-sensitive data.","Build an authentication system that supports SSO, maintains backward compatibility with legacy tokens, and meets SOC 2 compliance requirements.","Design a feature flag system that supports gradual rollout, instant rollback, and audit logging without adding latency to the hot path.","Architect a message queue that guarantees ordered delivery, supports dead letter handling, and achieves at-least-once delivery within bounded memory.","Design a search index that supports fuzzy matching, real-time updates, and relevance ranking while staying within a 10GB memory footprint."],
    'science': ["Design a chemical synthesis route that achieves the target molecule with high enantiomeric excess, avoids toxic reagents, and works at room temperature.","Select materials for a spacecraft heat shield that withstands 3000K reentry temperature, minimizes mass, and maintains structural integrity under vibration.","Design an ecological preserve that maintains viable populations of three competing predator species, supports their prey base, and fits within 500 square kilometers.","Choose experimental parameters for a particle detector that maximizes detection efficiency, minimizes background noise, and operates within the available beam energy.","Design a protein with specific binding affinity, thermal stability above 80C, and solubility in aqueous buffer at physiological pH.","Select a catalyst that achieves high selectivity for the desired product, operates at moderate pressure, and resists poisoning by sulfur compounds.","Design a telescope observation schedule that maximizes sky coverage, accounts for atmospheric seeing conditions, and avoids satellite interference windows.","Choose a battery chemistry that achieves 500 Wh/kg energy density, sustains 1000 charge cycles, and operates safely across -20C to 60C.","Design a gene therapy vector that achieves tissue-specific expression, avoids immune detection, and integrates at a safe genomic locus.","Select parameters for a climate model that resolves mesoscale convection, runs within available compute budget, and maintains energy balance closure."],
}

EVIDENCE_PROBES = {
    'medical': ["A patient has mildly elevated troponin, nonspecific ST changes, pleuritic chest pain, and a recent upper respiratory infection. Weigh the evidence for myocarditis vs acute MI.","Given a positive ANA, joint pain, oral ulcers, and borderline low complement, assess the strength of evidence for systemic lupus vs other autoimmune conditions.","A screening mammogram shows a 1cm lesion with irregular margins but no calcifications; family history is negative; patient is 35. Synthesize the evidence for biopsy vs watchful waiting.","CT shows a 2cm adrenal incidentaloma; cortisol is borderline high; catecholamines are normal; patient is asymptomatic. Weigh the evidence for surgical removal vs monitoring.","Child presents with fever, rash, and strawberry tongue; rapid strep is negative but ASO titer is elevated. Evaluate the evidence for Kawasaki disease vs scarlet fever.","MRI shows a brain lesion with ring enhancement; patient is immunocompromised; CSF shows elevated protein. Weigh evidence for toxoplasmosis vs CNS lymphoma vs abscess.","Pulmonary function tests show mixed obstructive-restrictive pattern; DLCO is reduced; CT shows ground glass and honeycombing. Assess evidence for IPF vs combined COPD-fibrosis.","A patient with chronic fatigue has borderline TSH, low-normal B12, mildly positive celiac antibodies, and normal CBC. Synthesize the diagnostic evidence.","An infant has mild jaundice, elevated direct bilirubin, pale stools, and normal liver enzymes. Weigh evidence for biliary atresia vs neonatal hepatitis.","Post-surgical patient develops tachycardia, mild hypoxia, and elevated D-dimer but a negative CT angiogram. Evaluate the evidence for PE vs post-operative atelectasis."],
    'legal': ["Weigh the evidence for trade secret misappropriation given: departed employee, similar product launch 6 months later, no documented reverse engineering, but different implementation details.","Assess whether a contract was formed given: oral agreement, partial performance, emails referencing terms, but no signed document and a statute of frauds defense.","Evaluate the evidence for discriminatory termination given: statistical disparity in layoffs, protected class membership, positive performance reviews, but documented business restructuring.","Weigh whether a use constitutes fair use given: commercial purpose, transformative nature, small portion used, but significant market impact on the original.","Assess the strength of a medical malpractice claim given: bad outcome, deviation from one guideline, compliance with another guideline, and a pre-existing condition.","Evaluate evidence for actual malice in a defamation case: public figure plaintiff, factual errors in reporting, unnamed sources, but deadline pressure and partial correction.","Weigh the evidence for personal jurisdiction given: website accessible in the forum state, one contract with a forum resident, but no physical presence or targeted advertising.","Assess whether conduct rises to antitrust violation given: parallel pricing behavior, evidence of industry meetings, but also independent cost pressures explaining the price changes.","Evaluate the evidence for a hostile work environment given: three offensive comments over two years, one formal complaint, supervisor awareness, but no direct adverse action.","Weigh the evidence for patent infringement given: similar functionality, different underlying algorithm, access to the patent, and a design-around memo."],
    'code': ["Production latency spikes to 500ms intermittently; CPU is normal; memory shows minor growth; one dependency was recently updated; database connections are near the pool limit. Diagnose the root cause.","Test suite passes locally but fails in CI; the failing tests involve timestamps; CI runs in UTC; some tests use mocked time but others don't. Weigh the evidence for each possible cause.","Application memory grows 2% per hour; heap dumps show string accumulation; two recent PRs added caching; GC logs show full collections increasing. Identify the leak source.","Users report intermittent 403 errors; load balancer logs show normal routing; auth service shows no failures; CDN cache was recently purged; some users are on VPNs. Synthesize the evidence.","Deployment causes 5% error rate increase; rollback fixes it; diff shows only CSS changes and one config key rename; staging showed no issues. Evaluate the evidence for root cause.","ML model accuracy drops 3% after retraining; training data volume increased 20%; no code changes; feature distributions look similar; new data source was added. Diagnose the degradation.","Database replication lag spikes during peak hours; write volume is steady; one replica shows higher lag than others; network metrics are normal; a new index was added yesterday. Assess the evidence.","A distributed system shows split-brain behavior; network partition tests pass; clock synchronization is within bounds; one node's disk is 95% full; consensus logs show election churn. Diagnose.","API response times degrade linearly over a week; request volume is flat; no deployments; one downstream service added request tracing; connection pool metrics are stable. Weigh the evidence.","Builds that were taking 3 minutes now take 12; dependency cache hit rate dropped from 95% to 60%; no changes to build config; CI provider changed their runner images last week. Assess."],
    'science': ["A galaxy's rotation curve is flat at large radii; lensing mass exceeds visible mass; but MOND predicts the rotation curve without dark matter. Weigh the evidence for dark matter vs modified gravity.","An exoplanet's atmosphere shows water absorption, methane, and no CO2; the star is quiet; but the detection is at 2.5 sigma and the instrument has known systematics. Assess biosignature strength.","A clinical trial shows p=0.03 for drug efficacy; effect size is small; prior probability from mechanism is moderate; the trial was stopped early; subgroup analysis shows benefit only in one group. Evaluate.","Fossil evidence shows a transitional form; dating places it in the expected period; but the geographic location is unexpected; and one morphological feature doesn't fit the predicted sequence. Weigh the evidence.","A new measurement of the Hubble constant disagrees with CMB-derived values by 4.4 sigma; systematic checks find no error; but three other methods give intermediate values. Assess the tension.","Seismic data suggests a subsurface ocean on an icy moon; thermal models support it; but gravity measurements are ambiguous; and the ice shell thickness estimate has large uncertainties. Evaluate.","A materials experiment shows superconductivity at 15C under 200GPa; two groups replicate partially; one group finds no signal; sample characterization shows possible impurity phases. Assess the claim.","Paleoclimate proxies from ice cores, tree rings, and ocean sediments give conflicting temperature reconstructions for the Medieval Warm Period. Weigh the evidence from each proxy's strengths and limitations.","A protein crystallography structure at 3.5A resolution shows an unexpected binding mode; MD simulations support it; but mutagenesis data partially contradicts the predicted contacts. Synthesize.","An astronomical transient shows properties of both a kilonova and a long gamma-ray burst; the host galaxy is unusual for either; gravitational wave data is ambiguous. Classify the event."],
}

SHAPE_PROBES = {'hierarchical': HIERARCHICAL_PROBES, 'causal': CAUSAL_PROBES, 'constraint': CONSTRAINT_PROBES, 'evidence': EVIDENCE_PROBES}


def get_all_probes():
    triples = []
    for shape in SHAPES:
        for domain in DOMAINS:
            for text in SHAPE_PROBES[shape][domain]:
                triples.append((shape, domain, text))
    return triples


def capture_activations(model, tokenizer, probes, max_length=128, batch_size=16):
    texts = [p[2] for p in probes]
    encoded = tokenizer(texts, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
    model.eval()
    device = next(model.parameters()).device
    activations = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_ids = encoded['input_ids'][i:i+batch_size].to(device)
            batch_mask = encoded['attention_mask'][i:i+batch_size].to(device)
            outputs = model(input_ids=batch_ids, attention_mask=batch_mask, output_hidden_states=True)
            hidden = outputs.hidden_states[-1].float()
            mask_expanded = batch_mask.unsqueeze(-1).float()
            pooled = (hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
            activations.append(pooled.cpu().numpy())
            print(f'    batch {i//batch_size + 1}/{(len(texts)+batch_size-1)//batch_size}', flush=True)
    return np.concatenate(activations, axis=0)


def classify(X, labels, cv=5):
    clf = LogisticRegression(max_iter=5000, solver='lbfgs')
    scores = cross_val_score(clf, X, labels, cv=cv, scoring='accuracy')
    return scores.mean(), scores.std()


def run_inlp(X, domain_labels, shape_labels, max_iter=50, threshold=0.30, label='Standard',
             noise_fn=None):
    """Generic INLP with optional noise injection."""
    X = X.copy()
    removed = []
    history = []
    for iteration in range(max_iter):
        d_acc, _ = classify(X, domain_labels)
        s_acc, _ = classify(X, shape_labels)
        history.append({'iter': iteration, 'domain': float(d_acc), 'shape': float(s_acc), 'method': label})
        print(f'  [{label}] INLP {iteration:2d}: domain={d_acc:.3f}, shape={s_acc:.3f}', flush=True)
        if d_acc < threshold:
            break

        if noise_fn is not None:
            X_train, y_train = noise_fn(X, domain_labels, removed)
        else:
            X_train, y_train = X, domain_labels

        clf = LogisticRegression(max_iter=5000, solver='lbfgs')
        clf.fit(X_train, y_train)
        W = clf.coef_
        _, _, Vt = np.linalg.svd(W, full_matrices=False)
        d = Vt[0]; d = d / np.linalg.norm(d)
        X = X - np.outer(X @ d, d)
        removed.append(d)
    return X, np.array(removed), history


def make_shaped_noise_fn(shape_labels_all, noise_scale, n_samples=5):
    """Create shaped noise function (closure over shape basis)."""
    def fn(X, domain_labels, removed):
        # Compute shape basis from current X
        shape_clf = LogisticRegression(max_iter=5000, solver='lbfgs')
        shape_clf.fit(X, shape_labels_all)
        W_shape = shape_clf.coef_
        _, _, Vt = np.linalg.svd(W_shape, full_matrices=False)
        S_basis = Vt[:len(SHAPES)-1]
        P_S = S_basis.T @ S_basis
        P_S_perp = np.eye(X.shape[1]) - P_S

        act_std = np.std(X)
        X_all = [X.copy()]
        for _ in range(n_samples):
            noise = np.random.randn(*X.shape) * act_std * noise_scale
            X_all.append(X + noise @ P_S_perp)
        return np.vstack(X_all), np.tile(domain_labels, n_samples + 1)
    return fn


def make_isotropic_noise_fn(noise_scale, n_samples=5):
    """Create isotropic noise function."""
    def fn(X, domain_labels, removed):
        act_std = np.std(X)
        X_all = [X.copy()]
        for _ in range(n_samples):
            noise = np.random.randn(*X.shape) * act_std * noise_scale
            X_all.append(X + noise)
        return np.vstack(X_all), np.tile(domain_labels, n_samples + 1)
    return fn


def run_single_experiment(args_tuple):
    """Worker function for parallel execution."""
    label, activations_file, out_dir, shape_labels, domain_labels, noise_type, noise_scale = args_tuple

    act = np.load(activations_file)
    np.random.seed(42 + hash(label) % 10000)

    if noise_type == 'standard':
        noise_fn = None
    elif noise_type == 'shaped':
        noise_fn = make_shaped_noise_fn(shape_labels, noise_scale)
    elif noise_type == 'isotropic':
        noise_fn = make_isotropic_noise_fn(noise_scale)

    t0 = time.time()
    _, dirs, history = run_inlp(act, domain_labels, shape_labels, max_iter=50, threshold=0.30,
                                 label=label, noise_fn=noise_fn)
    elapsed = time.time() - t0

    # Save results immediately
    out = Path(out_dir)
    with open(out / f'{label}_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    np.save(out / f'{label}_dirs.npy', dirs)

    result = {
        'label': label,
        'n_iters': len(history),
        'final_domain': history[-1]['domain'],
        'final_shape': history[-1]['shape'],
        'elapsed_s': elapsed,
        'n_dirs': len(dirs),
    }
    print(f'\n  DONE [{label}]: {len(dirs)} dirs, domain={history[-1]["domain"]:.3f}, shape={history[-1]["shape"]:.3f}, {elapsed:.0f}s', flush=True)
    return result


def cross_prediction(all_dirs_dict):
    """Measure collinearity among INLP directions across scales."""
    print('\n' + '=' * 60)
    print('CROSS-PREDICTION: INLP DIRECTION COLLINEARITY')
    print('=' * 60, flush=True)

    results = {}
    for name, dirs in all_dirs_dict.items():
        if len(dirs) < 2:
            continue
        n = len(dirs)
        cos_sims = []
        for i in range(n):
            for j in range(i+1, n):
                cos = abs(np.dot(dirs[i], dirs[j]))
                cos_sims.append(cos)

        r = {
            'n_dirs': n,
            'mean_cosine': float(np.mean(cos_sims)),
            'max_cosine': float(np.max(cos_sims)),
            'std_cosine': float(np.std(cos_sims)),
        }
        results[name] = r
        print(f'  {name}: {n} dirs, mean|cos|={r["mean_cosine"]:.4f}, max={r["max_cosine"]:.4f}', flush=True)

    return results


def main():
    t_start = time.time()
    probes = get_all_probes()
    shape_labels = np.array([p[0] for p in probes])
    domain_labels = np.array([p[1] for p in probes])

    out_dir = Path('results/paper4_shaped')
    out_dir.mkdir(parents=True, exist_ok=True)

    models = ['Qwen/Qwen2.5-0.5B', 'Qwen/Qwen2.5-1.5B', 'Qwen/Qwen2.5-3B', 'Qwen/Qwen2.5-7B']

    # ---- Phase 1: Capture all activations (sequential, GPU-bound) ----
    print('=' * 60)
    print('PHASE 1: CAPTURE ACTIVATIONS')
    print('=' * 60, flush=True)

    for model_name in models:
        ms = model_name.split('/')[-1]
        cache = out_dir / f'{ms}_activations.npy'
        if cache.exists():
            print(f'  {ms}: cached ({np.load(cache).shape})', flush=True)
            continue
        print(f'  {ms}: loading...', flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float16, device_map='auto', low_cpu_mem_usage=True)
        t0 = time.time()
        act = capture_activations(model, tokenizer, probes, batch_size=16)
        np.save(cache, act)
        print(f'  {ms}: {act.shape} in {time.time()-t0:.0f}s', flush=True)
        del model, tokenizer
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ---- Phase 2: Run ALL INLP experiments in parallel ----
    print('\n' + '=' * 60)
    print('PHASE 2: PARALLEL INLP EXPERIMENTS')
    print('=' * 60, flush=True)

    # Build job list
    jobs = []
    for model_name in models:
        ms = model_name.split('/')[-1]
        act_file = str(out_dir / f'{ms}_activations.npy')
        label = f'{ms}_standard'
        hist_file = out_dir / f'{label}_history.json'
        if not hist_file.exists():
            jobs.append((label, act_file, str(out_dir), shape_labels, domain_labels, 'standard', 0))

    # Noise experiments on 7B
    act_7b_file = str(out_dir / 'Qwen2.5-7B_activations.npy')
    for ns in [0.05, 0.1, 0.2]:
        label = f'7B_shaped_{ns}'
        if not (out_dir / f'{label}_history.json').exists():
            jobs.append((label, act_7b_file, str(out_dir), shape_labels, domain_labels, 'shaped', ns))
        label = f'7B_isotropic_{ns}'
        if not (out_dir / f'{label}_history.json').exists():
            jobs.append((label, act_7b_file, str(out_dir), shape_labels, domain_labels, 'isotropic', ns))

    print(f'  {len(jobs)} experiments to run', flush=True)
    for j in jobs:
        print(f'    - {j[0]}', flush=True)

    if len(jobs) == 0:
        print('  All experiments cached!')
    else:
        # Run in parallel — each gets ~18 cores on a 128-core machine
        n_workers = min(len(jobs), 10)
        print(f'  Running with {n_workers} parallel workers...', flush=True)

        all_results = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(run_single_experiment, job): job[0] for job in jobs}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                    all_results.append(result)
                except Exception as e:
                    print(f'  ERROR [{name}]: {e}', flush=True)

    # ---- Phase 3: Cross-prediction ----
    all_dirs = {}
    for model_name in models:
        ms = model_name.split('/')[-1]
        dirs_file = out_dir / f'{ms}_standard_dirs.npy'
        if dirs_file.exists():
            all_dirs[ms] = np.load(dirs_file)
        # Also check old naming convention
        dirs_file2 = out_dir / f'{ms}_inlp_dirs.npy'
        if dirs_file2.exists() and ms not in all_dirs:
            all_dirs[ms] = np.load(dirs_file2)

    xpred = cross_prediction(all_dirs)

    # ---- Phase 4: Summary ----
    t_total = time.time() - t_start
    print(f'\n{"="*60}')
    print(f'TOTAL TIME: {t_total:.0f}s ({t_total/60:.1f}m)')
    print(f'{"="*60}')

    # Collect all histories
    all_histories = {}
    for f in sorted(out_dir.glob('*_history.json')):
        with open(f) as fh:
            all_histories[f.stem.replace('_history', '')] = json.load(fh)

    # 7B noise comparison table
    print('\n--- 7B NOISE EXPERIMENT COMPARISON ---')
    print(f'{"Method":<25} {"Iters":>5} {"Final Domain":>13} {"Final Shape":>12} {"Survival":>10}')
    for key in sorted(all_histories.keys()):
        if '7B' not in key and 'Qwen2.5-7B' not in key:
            continue
        h = all_histories[key]
        last = h[-1]
        first_s = h[0]['shape']
        surv = last['shape'] / first_s if first_s > 0 else 0
        print(f'{key:<25} {last["iter"]:>5d} {last["domain"]:>13.3f} {last["shape"]:>12.3f} {surv:>9.1%}')

    # Scale comparison
    print('\n--- STANDARD INLP BY SCALE ---')
    print(f'{"Scale":<20} {"Iters":>5} {"Final Domain":>13} {"Final Shape":>12}')
    for model_name in models:
        ms = model_name.split('/')[-1]
        key = f'{ms}_standard'
        if key in all_histories:
            h = all_histories[key]
            last = h[-1]
            print(f'{ms:<20} {last["iter"]:>5d} {last["domain"]:>13.3f} {last["shape"]:>12.3f}')

    # Collinearity
    if xpred:
        print('\n--- INLP DIRECTION COLLINEARITY ---')
        print(f'{"Scale":<20} {"N dirs":>6} {"Mean |cos|":>11} {"Max |cos|":>10}')
        for name, r in xpred.items():
            print(f'{name:<20} {r["n_dirs"]:>6d} {r["mean_cosine"]:>11.4f} {r["max_cosine"]:>10.4f}')

    summary = {'total_time_s': t_total, 'cross_prediction': xpred}
    with open(out_dir / 'full_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nResults saved to {out_dir}/', flush=True)


if __name__ == '__main__':
    main()
