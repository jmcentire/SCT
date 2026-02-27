"""Paper 4 Scale Validation: Full pipeline for any model scale.

Self-contained script that runs the complete Paper 4 analysis:
  1. Capture shape-labeled activations
  2. INLP domain erasure
  3. All transfer tests
  4. Save results

Usage: python3 paper4_scale_validation.py --model Qwen/Qwen2.5-0.5B
       python3 paper4_scale_validation.py --model Qwen/Qwen2.5-1.5B
       python3 paper4_scale_validation.py --model Qwen/Qwen2.5-3B
       python3 paper4_scale_validation.py --model Qwen/Qwen2.5-7B
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import cosine as cosine_dist
from scipy import stats

# ============================================================
# Probe definitions (embedded for self-containment)
# ============================================================

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
    'medical': [
        "Trace the pathophysiological cascade from deep vein thrombosis to pulmonary embolism to right heart failure.",
        "Explain the chain of events from H. pylori infection through chronic gastritis to gastric adenocarcinoma.",
        "Follow the progression from insulin resistance through compensatory hyperinsulinemia to beta cell exhaustion in type 2 diabetes.",
        "Trace how untreated streptococcal pharyngitis leads to rheumatic fever through molecular mimicry.",
        "Describe the cascade from atherosclerotic plaque rupture through thrombus formation to myocardial infarction.",
        "Follow the sequence from portal hypertension through varices formation to variceal hemorrhage.",
        "Trace the chain from chronic hypertension through nephrosclerosis to end-stage renal disease.",
        "Explain how a mutation in the CFTR gene leads through chloride channel dysfunction to the clinical manifestations of cystic fibrosis.",
        "Follow the cascade from traumatic brain injury through secondary injury mechanisms to long-term neurological deficit.",
        "Trace the progression from chronic hepatitis C through fibrosis stages to cirrhosis and hepatocellular carcinoma.",
    ],
    'legal': [
        "Trace how a single SEC filing omission cascades through materiality analysis to securities fraud liability.",
        "Follow the chain from a police officer's warrantless search through the exclusionary rule to case dismissal.",
        "Explain how a manufacturer's design defect leads through failure to warn to strict product liability.",
        "Trace the sequence from a constitutional amendment's proposal through ratification to judicial enforcement.",
        "Follow the chain from a breach of fiduciary duty through damages to disgorgement of profits.",
        "Describe how precedent from a circuit court decision propagates through subsequent cases to a circuit split to Supreme Court review.",
        "Trace how a zoning variance denial cascades through administrative appeal to judicial review under the arbitrary and capricious standard.",
        "Follow the sequence from discriminatory intent through disparate impact evidence to a Title VII violation finding.",
        "Explain how a contract clause's ambiguity leads through parol evidence consideration to reformation.",
        "Trace the chain from an environmental violation through EPA enforcement action to consent decree to ongoing compliance monitoring.",
    ],
    'code': [
        "Trace how a buffer overflow in user input leads through stack corruption to arbitrary code execution.",
        "Follow the chain from a DNS misconfiguration through failed certificate validation to a man-in-the-middle vulnerability.",
        "Explain how a memory leak in a long-running service leads through heap exhaustion to cascading service failures.",
        "Trace the sequence from a race condition in a database write through inconsistent state to data corruption.",
        "Follow the cascade from a single microservice failure through timeout propagation to full system brownout.",
        "Describe how an N+1 query pattern leads through connection pool exhaustion to database lock contention to user-visible latency.",
        "Trace how a floating-point rounding error propagates through iterative computation to produce a catastrophically wrong result.",
        "Follow the chain from a misconfigured load balancer through uneven traffic distribution to hot-spot failure.",
        "Explain how a dependency version conflict leads through incompatible API calls to a runtime crash in production.",
        "Trace the sequence from a Git merge conflict through an incorrect resolution to a regression that passes CI but fails in production.",
    ],
    'science': [
        "Trace the chain from increased atmospheric CO2 through radiative forcing to ocean acidification and coral bleaching.",
        "Follow the sequence from a supernova explosion through nucleosynthesis to the formation of heavy elements found on Earth.",
        "Explain how a single base-pair mutation leads through protein misfolding to prion propagation and neurodegeneration.",
        "Trace the cascade from ozone depletion through increased UV radiation to increased mutation rates in surface organisms.",
        "Follow the chain from tectonic plate subduction through magma generation to volcanic eruption and atmospheric effects.",
        "Describe how a perturbation in initial conditions leads through chaotic amplification to divergent weather outcomes (butterfly effect).",
        "Trace the sequence from photon absorption in a semiconductor through electron-hole pair generation to current flow in a solar cell.",
        "Follow the cascade from antibiotic overuse through selection pressure to resistance gene horizontal transfer to multi-drug resistant organisms.",
        "Explain how Milankovitch cycle variations lead through insolation changes to ice sheet advance and retreat.",
        "Trace the chain from quantum tunneling in hydrogen fusion through radiation pressure to stellar equilibrium.",
    ],
}

CONSTRAINT_PROBES = {
    'medical': [
        "A patient has renal failure, a penicillin allergy, and a methicillin-resistant staph infection. Choose an antibiotic regimen that addresses all three constraints.",
        "Design a post-surgical pain management plan for a patient with a history of opioid addiction, hepatic impairment, and NSAID-induced gastric ulcers.",
        "A pregnant patient in her first trimester presents with new-onset epilepsy. Select an anticonvulsant that minimizes teratogenicity while controlling seizures.",
        "Choose an antihypertensive for a diabetic patient with bilateral renal artery stenosis and hyperkalemia.",
        "A patient needs anticoagulation for atrial fibrillation but has a recent GI bleed and thrombocytopenia. Design a management strategy.",
        "Select an anesthetic protocol for a patient with malignant hyperthermia susceptibility, difficult airway, and severe cardiac disease.",
        "Design a chemotherapy regimen for a cancer patient with pre-existing cardiomyopathy, neuropathy, and neutropenia.",
        "A patient with Parkinson's disease needs an antiemetic but cannot receive dopamine antagonists. Identify appropriate options.",
        "Choose a contraceptive method for a patient with a history of DVT, migraine with aura, and liver disease.",
        "Design a nutritional plan for a critically ill patient with diabetes, celiac disease, and severe nut allergy.",
    ],
    'legal': [
        "Draft a contract clause that satisfies consumer protection disclosure requirements, company IP protection needs, and GDPR data processing constraints simultaneously.",
        "Structure a corporate transaction that achieves tax-free reorganization status while maintaining regulatory approval and satisfying minority shareholder rights.",
        "Design a sentencing recommendation that accounts for mandatory minimums, mitigating factors, and proportionality requirements under the Eighth Amendment.",
        "Construct a trust that satisfies the rule against perpetuities, achieves the desired estate tax treatment, and protects assets from the beneficiary's creditors.",
        "Draft an employment agreement that complies with non-compete enforceability requirements, trade secret protection, and employee mobility rights in California.",
        "Structure a plea agreement that satisfies the victim's restitution rights, the defendant's constitutional protections, and the prosecution's sentencing guidelines obligations.",
        "Design a regulatory compliance program that simultaneously meets SEC, FINRA, and state blue sky law requirements for a multi-state securities offering.",
        "Draft a licensing agreement that satisfies open-source copyleft obligations, proprietary code protection, and export control restrictions.",
        "Structure a bankruptcy reorganization plan that satisfies absolute priority, feasibility, and best interests of creditors tests simultaneously.",
        "Design a class action settlement that meets adequacy of representation, commonality, and fairness hearing requirements.",
    ],
    'code': [
        "Design a database schema that satisfies third normal form, supports the required query patterns without joins, and stays within the storage budget.",
        "Architect a system that achieves sub-100ms latency, 99.99% availability, and strong consistency across three geographic regions.",
        "Write a scheduling algorithm that respects task dependencies, meets all deadlines, and minimizes total resource usage.",
        "Design an API rate limiting system that prevents abuse, allows burst traffic from legitimate users, and maintains fairness across tenants.",
        "Architect a data pipeline that preserves exactly-once processing semantics, handles schema evolution, and scales horizontally.",
        "Design a cache eviction policy that maximizes hit rate, respects memory limits, and guarantees freshness for time-sensitive data.",
        "Build an authentication system that supports SSO, maintains backward compatibility with legacy tokens, and meets SOC 2 compliance requirements.",
        "Design a feature flag system that supports gradual rollout, instant rollback, and audit logging without adding latency to the hot path.",
        "Architect a message queue that guarantees ordered delivery, supports dead letter handling, and achieves at-least-once delivery within bounded memory.",
        "Design a search index that supports fuzzy matching, real-time updates, and relevance ranking while staying within a 10GB memory footprint.",
    ],
    'science': [
        "Design a chemical synthesis route that achieves the target molecule with high enantiomeric excess, avoids toxic reagents, and works at room temperature.",
        "Select materials for a spacecraft heat shield that withstands 3000K reentry temperature, minimizes mass, and maintains structural integrity under vibration.",
        "Design an ecological preserve that maintains viable populations of three competing predator species, supports their prey base, and fits within 500 square kilometers.",
        "Choose experimental parameters for a particle detector that maximizes detection efficiency, minimizes background noise, and operates within the available beam energy.",
        "Design a protein with specific binding affinity, thermal stability above 80C, and solubility in aqueous buffer at physiological pH.",
        "Select a catalyst that achieves high selectivity for the desired product, operates at moderate pressure, and resists poisoning by sulfur compounds.",
        "Design a telescope observation schedule that maximizes sky coverage, accounts for atmospheric seeing conditions, and avoids satellite interference windows.",
        "Choose a battery chemistry that achieves 500 Wh/kg energy density, sustains 1000 charge cycles, and operates safely across -20C to 60C.",
        "Design a gene therapy vector that achieves tissue-specific expression, avoids immune detection, and integrates at a safe genomic locus.",
        "Select parameters for a climate model that resolves mesoscale convection, runs within available compute budget, and maintains energy balance closure.",
    ],
}

EVIDENCE_PROBES = {
    'medical': [
        "A patient has mildly elevated troponin, nonspecific ST changes, pleuritic chest pain, and a recent upper respiratory infection. Weigh the evidence for myocarditis vs acute MI.",
        "Given a positive ANA, joint pain, oral ulcers, and borderline low complement, assess the strength of evidence for systemic lupus vs other autoimmune conditions.",
        "A screening mammogram shows a 1cm lesion with irregular margins but no calcifications; family history is negative; patient is 35. Synthesize the evidence for biopsy vs watchful waiting.",
        "CT shows a 2cm adrenal incidentaloma; cortisol is borderline high; catecholamines are normal; patient is asymptomatic. Weigh the evidence for surgical removal vs monitoring.",
        "Child presents with fever, rash, and strawberry tongue; rapid strep is negative but ASO titer is elevated. Evaluate the evidence for Kawasaki disease vs scarlet fever.",
        "MRI shows a brain lesion with ring enhancement; patient is immunocompromised; CSF shows elevated protein. Weigh evidence for toxoplasmosis vs CNS lymphoma vs abscess.",
        "Pulmonary function tests show mixed obstructive-restrictive pattern; DLCO is reduced; CT shows ground glass and honeycombing. Assess evidence for IPF vs combined COPD-fibrosis.",
        "A patient with chronic fatigue has borderline TSH, low-normal B12, mildly positive celiac antibodies, and normal CBC. Synthesize the diagnostic evidence.",
        "An infant has mild jaundice, elevated direct bilirubin, pale stools, and normal liver enzymes. Weigh evidence for biliary atresia vs neonatal hepatitis.",
        "Post-surgical patient develops tachycardia, mild hypoxia, and elevated D-dimer but a negative CT angiogram. Evaluate the evidence for PE vs post-operative atelectasis.",
    ],
    'legal': [
        "Weigh the evidence for trade secret misappropriation given: departed employee, similar product launch 6 months later, no documented reverse engineering, but different implementation details.",
        "Assess whether a contract was formed given: oral agreement, partial performance, emails referencing terms, but no signed document and a statute of frauds defense.",
        "Evaluate the evidence for discriminatory termination given: statistical disparity in layoffs, protected class membership, positive performance reviews, but documented business restructuring.",
        "Weigh whether a use constitutes fair use given: commercial purpose, transformative nature, small portion used, but significant market impact on the original.",
        "Assess the strength of a medical malpractice claim given: bad outcome, deviation from one guideline, compliance with another guideline, and a pre-existing condition.",
        "Evaluate evidence for actual malice in a defamation case: public figure plaintiff, factual errors in reporting, unnamed sources, but deadline pressure and partial correction.",
        "Weigh the evidence for personal jurisdiction given: website accessible in the forum state, one contract with a forum resident, but no physical presence or targeted advertising.",
        "Assess whether conduct rises to antitrust violation given: parallel pricing behavior, evidence of industry meetings, but also independent cost pressures explaining the price changes.",
        "Evaluate the evidence for a hostile work environment given: three offensive comments over two years, one formal complaint, supervisor awareness, but no direct adverse action.",
        "Weigh the evidence for patent infringement given: similar functionality, different underlying algorithm, access to the patent, and a design-around memo.",
    ],
    'code': [
        "Production latency spikes to 500ms intermittently; CPU is normal; memory shows minor growth; one dependency was recently updated; database connections are near the pool limit. Diagnose the root cause.",
        "Test suite passes locally but fails in CI; the failing tests involve timestamps; CI runs in UTC; some tests use mocked time but others don't. Weigh the evidence for each possible cause.",
        "Application memory grows 2% per hour; heap dumps show string accumulation; two recent PRs added caching; GC logs show full collections increasing. Identify the leak source.",
        "Users report intermittent 403 errors; load balancer logs show normal routing; auth service shows no failures; CDN cache was recently purged; some users are on VPNs. Synthesize the evidence.",
        "Deployment causes 5% error rate increase; rollback fixes it; diff shows only CSS changes and one config key rename; staging showed no issues. Evaluate the evidence for root cause.",
        "ML model accuracy drops 3% after retraining; training data volume increased 20%; no code changes; feature distributions look similar; new data source was added. Diagnose the degradation.",
        "Database replication lag spikes during peak hours; write volume is steady; one replica shows higher lag than others; network metrics are normal; a new index was added yesterday. Assess the evidence.",
        "A distributed system shows split-brain behavior; network partition tests pass; clock synchronization is within bounds; one node's disk is 95% full; consensus logs show election churn. Diagnose.",
        "API response times degrade linearly over a week; request volume is flat; no deployments; one downstream service added request tracing; connection pool metrics are stable. Weigh the evidence.",
        "Builds that were taking 3 minutes now take 12; dependency cache hit rate dropped from 95% to 60%; no changes to build config; CI provider changed their runner images last week. Assess.",
    ],
    'science': [
        "A galaxy's rotation curve is flat at large radii; lensing mass exceeds visible mass; but MOND predicts the rotation curve without dark matter. Weigh the evidence for dark matter vs modified gravity.",
        "An exoplanet's atmosphere shows water absorption, methane, and no CO2; the star is quiet; but the detection is at 2.5 sigma and the instrument has known systematics. Assess biosignature strength.",
        "A clinical trial shows p=0.03 for drug efficacy; effect size is small; prior probability from mechanism is moderate; the trial was stopped early; subgroup analysis shows benefit only in one group. Evaluate.",
        "Fossil evidence shows a transitional form; dating places it in the expected period; but the geographic location is unexpected; and one morphological feature doesn't fit the predicted sequence. Weigh the evidence.",
        "A new measurement of the Hubble constant disagrees with CMB-derived values by 4.4 sigma; systematic checks find no error; but three other methods give intermediate values. Assess the tension.",
        "Seismic data suggests a subsurface ocean on an icy moon; thermal models support it; but gravity measurements are ambiguous; and the ice shell thickness estimate has large uncertainties. Evaluate.",
        "A materials experiment shows superconductivity at 15C under 200GPa; two groups replicate partially; one group finds no signal; sample characterization shows possible impurity phases. Assess the claim.",
        "Paleoclimate proxies from ice cores, tree rings, and ocean sediments give conflicting temperature reconstructions for the Medieval Warm Period. Weigh the evidence from each proxy's strengths and limitations.",
        "A protein crystallography structure at 3.5A resolution shows an unexpected binding mode; MD simulations support it; but mutagenesis data partially contradicts the predicted contacts. Synthesize.",
        "An astronomical transient shows properties of both a kilonova and a long gamma-ray burst; the host galaxy is unusual for either; gravitational wave data is ambiguous. Classify the event.",
    ],
}

SHAPE_PROBES = {
    'hierarchical': HIERARCHICAL_PROBES,
    'causal': CAUSAL_PROBES,
    'constraint': CONSTRAINT_PROBES,
    'evidence': EVIDENCE_PROBES,
}

def get_all_probes():
    triples = []
    for shape in SHAPES:
        for domain in DOMAINS:
            for text in SHAPE_PROBES[shape][domain]:
                triples.append((shape, domain, text))
    return triples


# ============================================================
# Core functions
# ============================================================

def capture_activations(model, tokenizer, probes, max_length=128, batch_size=8):
    texts = [p[2] for p in probes]
    encoded = tokenizer(
        texts, padding='max_length', truncation=True,
        max_length=max_length, return_tensors='pt',
    )
    model.eval()
    device = next(model.parameters()).device
    activations = []
    n_batches = (len(texts) + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_ids = encoded['input_ids'][i:i+batch_size].to(device)
            batch_mask = encoded['attention_mask'][i:i+batch_size].to(device)
            outputs = model(input_ids=batch_ids, attention_mask=batch_mask,
                           output_hidden_states=True)
            hidden = outputs.hidden_states[-1].float()
            mask_expanded = batch_mask.unsqueeze(-1).float()
            pooled = (hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
            activations.append(pooled.cpu().numpy())
            batch_num = i // batch_size + 1
            if batch_num % 5 == 1:
                print(f'    batch {batch_num}/{n_batches}')

    return np.concatenate(activations, axis=0)


def classify(X, labels, cv=5):
    clf = LogisticRegression(max_iter=2000, solver='lbfgs')
    scores = cross_val_score(clf, X, labels, cv=cv, scoring='accuracy')
    return scores.mean(), scores.std()


def domain_erase_inlp(activations, domain_labels, max_iter=30, threshold=0.35):
    """INLP: iteratively remove domain directions until domain accuracy ~ chance."""
    X = activations.copy()
    removed = []
    n_classes = len(set(domain_labels))
    history = []

    for iteration in range(max_iter):
        d_acc, _ = classify(X, domain_labels)
        s_acc, _ = classify(X, np.array([p[0] for p in get_all_probes()]))
        history.append({'iter': iteration, 'domain_acc': float(d_acc), 'shape_acc': float(s_acc)})
        print(f'  INLP iter {iteration:2d}: domain={d_acc:.3f}, shape={s_acc:.3f}, dirs_removed={len(removed)}')

        if d_acc < threshold:
            print(f'  Domain near chance ({1/n_classes:.3f}). Done.')
            break

        clf = LogisticRegression(max_iter=2000, solver='lbfgs')
        clf.fit(X, domain_labels)
        W = clf.coef_
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        direction = Vt[0]
        direction = direction / np.linalg.norm(direction)
        X = X - np.outer(X @ direction, direction)
        removed.append(direction)

    return X, np.array(removed) if removed else np.zeros((0, activations.shape[1])), history


def domain_erase(X, domain_dirs):
    X = X.copy()
    for d in domain_dirs:
        d = d / (np.linalg.norm(d) + 1e-10)
        X = X - np.outer(X @ d, d)
    return X


def run_transfer_tests(activations, X_erased, domain_dirs, shape_labels, domain_labels):
    """Run all transfer tests, return results dict."""
    results = {}

    # Test 1: Cross-domain shape classification
    print('\n  TEST 1: Cross-domain shape classification')
    t1_accs = {}
    for target in DOMAINS:
        train_mask = domain_labels != target
        test_mask = domain_labels == target
        clf = LogisticRegression(max_iter=2000, solver='lbfgs')
        clf.fit(X_erased[train_mask], shape_labels[train_mask])
        acc = clf.score(X_erased[test_mask], shape_labels[test_mask])
        t1_accs[target] = float(acc)
        print(f'    {target:8s}: {acc:.3f}')
    t1_overall = np.mean(list(t1_accs.values()))
    print(f'    Overall: {t1_overall:.3f} (chance = 0.250)')
    results['test1_cross_domain'] = t1_overall
    results['test1_by_domain'] = t1_accs

    # Test 2: Nearest prototype
    print('\n  TEST 2: Nearest structural prototype')
    t2_accs = {}
    for target in DOMAINS:
        correct = 0
        total = 0
        source_mask = domain_labels != target
        for true_shape in SHAPES:
            target_mask = (domain_labels == target) & (shape_labels == true_shape)
            prototypes = {}
            for shape in SHAPES:
                smask = source_mask & (shape_labels == shape)
                prototypes[shape] = X_erased[smask].mean(axis=0)
            for vec in X_erased[target_mask]:
                dists = {s: cosine_dist(vec, p) for s, p in prototypes.items()}
                if min(dists, key=dists.get) == true_shape:
                    correct += 1
                total += 1
        t2_accs[target] = float(correct / total)
        print(f'    {target:8s}: {correct}/{total} = {correct/total:.3f}')
    t2_overall = np.mean(list(t2_accs.values()))
    print(f'    Overall: {t2_overall:.3f} (chance = 0.250)')
    results['test2_nearest_proto'] = t2_overall
    results['test2_by_domain'] = t2_accs

    # Test 3: Strip + rehydrate
    print('\n  TEST 3: Activation arithmetic (strip + rehydrate)')
    global_mean = activations.mean(axis=0)
    domain_means = {d: activations[domain_labels == d].mean(axis=0) for d in DOMAINS}

    correct_closer = 0
    total = 0
    by_shape = {s: {'correct': 0, 'total': 0} for s in SHAPES}
    by_direction = {}

    for src in DOMAINS:
        for tgt in DOMAINS:
            if src == tgt:
                continue
            key = f'{src}->{tgt}'
            dir_correct = 0
            dir_total = 0
            for shape in SHAPES:
                src_mask = (domain_labels == src) & (shape_labels == shape)
                tgt_mask = (domain_labels == tgt) & (shape_labels == shape)
                tgt_mean = activations[tgt_mask].mean(axis=0)
                wrong_means = []
                for ws in SHAPES:
                    if ws != shape:
                        ws_mask = (domain_labels == tgt) & (shape_labels == ws)
                        wrong_means.append(activations[ws_mask].mean(axis=0))

                for src_vec in activations[src_mask]:
                    erased = domain_erase(src_vec.reshape(1, -1), domain_dirs).squeeze()
                    rehydrated = erased + (domain_means[tgt] - global_mean)
                    dist_correct = cosine_dist(rehydrated, tgt_mean)
                    dist_wrong = np.mean([cosine_dist(rehydrated, wm) for wm in wrong_means])
                    if dist_correct < dist_wrong:
                        correct_closer += 1
                        dir_correct += 1
                        by_shape[shape]['correct'] += 1
                    total += 1
                    dir_total += 1
                    by_shape[shape]['total'] += 1
            by_direction[key] = float(dir_correct / dir_total)

    t3_overall = correct_closer / total
    print(f'    Overall: {correct_closer}/{total} = {t3_overall:.1%} (chance ~50%)')
    for shape in SHAPES:
        s = by_shape[shape]
        print(f'    {shape:15s}: {s["correct"]}/{s["total"]} = {s["correct"]/s["total"]:.1%}')

    t_stat, p_val = stats.ttest_rel(
        [1 if r else 0 for r in [True]*correct_closer + [False]*(total-correct_closer)],
        [0.5]*total
    )
    binom_p = stats.binomtest(correct_closer, total, 0.5, alternative='greater').pvalue
    print(f'    Binomial p = {binom_p:.2e}')

    results['test3_rehydrate'] = float(t3_overall)
    results['test3_binom_p'] = float(binom_p)
    results['test3_by_shape'] = {s: float(by_shape[s]['correct']/by_shape[s]['total']) for s in SHAPES}
    results['test3_by_direction'] = by_direction

    # RSA
    print('\n  RSA: Representational similarity analysis')
    norms = np.linalg.norm(activations, axis=1, keepdims=True)
    X_normed = activations / (norms + 1e-10)
    K_act = X_normed @ X_normed.T
    N = len(activations)
    within_both, within_domain, within_shape, within_neither = [], [], [], []
    for i in range(N):
        for j in range(i+1, N):
            sim = K_act[i, j]
            sd = domain_labels[i] == domain_labels[j]
            ss = shape_labels[i] == shape_labels[j]
            if sd and ss: within_both.append(sim)
            elif sd: within_domain.append(sim)
            elif ss: within_shape.append(sim)
            else: within_neither.append(sim)

    domain_effect = np.mean(within_domain) - np.mean(within_neither)
    shape_effect = np.mean(within_shape) - np.mean(within_neither)
    print(f'    Domain effect: {domain_effect:+.4f}')
    print(f'    Shape effect:  {shape_effect:+.4f}')
    print(f'    Ratio:         {domain_effect/(shape_effect+1e-10):.2f}')

    results['rsa_domain_effect'] = float(domain_effect)
    results['rsa_shape_effect'] = float(shape_effect)
    results['rsa_ratio'] = float(domain_effect / (shape_effect + 1e-10))

    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='HuggingFace model name')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--output-dir', type=str, default='results/paper4_scale')
    args = parser.parse_args()

    model_name = args.model
    model_short = model_name.split('/')[-1]

    print(f'Paper 4 Scale Validation: {model_name}')
    print('=' * 60)

    t_start = time.time()

    # Load model
    print(f'\nLoading {model_name}...')
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Choose device
    if torch.cuda.is_available():
        device_map = 'auto'
        dtype = torch.float16
    else:
        device_map = 'cpu'
        dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype,
        device_map=device_map, low_cpu_mem_usage=True,
    )
    hidden_dim = model.config.hidden_size
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f'  Hidden dim: {hidden_dim}')
    print(f'  Parameters: {n_params:.1f}B')
    print(f'  Device: {next(model.parameters()).device}')

    # Capture activations
    probes = get_all_probes()
    print(f'\nCapturing activations for {len(probes)} probes...')
    activations = capture_activations(model, tokenizer, probes, batch_size=args.batch_size)
    print(f'  Shape: {activations.shape}')

    # Free model
    del model, tokenizer
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    t_capture = time.time()
    print(f'  Capture time: {t_capture - t_start:.0f}s')

    shape_labels = np.array([p[0] for p in probes])
    domain_labels = np.array([p[1] for p in probes])

    # Baseline classifiers
    print('\n--- Baseline classifiers ---')
    d_acc, d_std = classify(activations, domain_labels)
    s_acc, s_std = classify(activations, shape_labels)
    print(f'  Domain: {d_acc:.3f} +/- {d_std:.3f}')
    print(f'  Shape:  {s_acc:.3f} +/- {s_std:.3f}')

    # INLP
    print('\n--- INLP domain erasure ---')
    X_erased, domain_dirs, inlp_history = domain_erase_inlp(activations, domain_labels)

    # Final INLP state
    final_d, _ = classify(X_erased, domain_labels)
    final_s, _ = classify(X_erased, shape_labels)
    shape_survival = final_s / s_acc if s_acc > 0 else 0
    print(f'\n  Final: domain={final_d:.3f}, shape={final_s:.3f}')
    print(f'  Domain directions removed: {len(domain_dirs)}')
    print(f'  Shape survival: {shape_survival:.1%}')

    # Transfer tests
    print('\n--- Transfer tests ---')
    transfer_results = run_transfer_tests(activations, X_erased, domain_dirs, shape_labels, domain_labels)

    t_end = time.time()

    # Summary
    print('\n' + '=' * 60)
    print(f'SCALE VALIDATION SUMMARY: {model_short}')
    print('=' * 60)
    print(f'  Model: {model_name}')
    print(f'  Hidden dim: {hidden_dim}')
    print(f'  Parameters: {n_params:.1f}B')
    print(f'  Domain directions removed: {len(domain_dirs)}')
    print(f'  Shape survival after domain erasure: {shape_survival:.1%}')
    print(f'  Cross-domain shape transfer: {transfer_results["test1_cross_domain"]:.3f}')
    print(f'  Nearest prototype transfer:  {transfer_results["test2_nearest_proto"]:.3f}')
    print(f'  Strip + rehydrate:           {transfer_results["test3_rehydrate"]:.1%}')
    print(f'  RSA domain/shape ratio:      {transfer_results["rsa_ratio"]:.2f}')
    print(f'  Total time: {t_end - t_start:.0f}s')

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = {
        'model': model_name,
        'model_short': model_short,
        'hidden_dim': hidden_dim,
        'n_params': float(n_params),
        'baseline_domain_acc': float(d_acc),
        'baseline_shape_acc': float(s_acc),
        'n_domain_dirs': len(domain_dirs),
        'final_domain_acc': float(final_d),
        'final_shape_acc': float(final_s),
        'shape_survival': float(shape_survival),
        'inlp_history': inlp_history,
        **transfer_results,
        'total_time_s': float(t_end - t_start),
    }

    with open(out_dir / f'{model_short}_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    np.save(out_dir / f'{model_short}_activations.npy', activations)
    np.save(out_dir / f'{model_short}_domain_dirs.npy', domain_dirs)
    print(f'\n  Saved to {out_dir}/{model_short}_*')


if __name__ == '__main__':
    main()
