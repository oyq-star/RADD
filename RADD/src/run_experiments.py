"""Main experiment runner for RADD.

Runs all experiment blocks:
- Block 1: Main results (baselines vs RADD)
- Block 2: Ablation study
- Block 3: Sparse user analysis
- Block 4: Social robustness (edge dropout)
- Block 5: Top-k sensitivity
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.optim as optim
from collections import defaultdict
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

from data_loader import preprocess_dataset
from anomaly_injection import AnomalyInjector
from model import RADD, GRUAutoencoder, LSTMAutoencoder, EarlyFusionAE


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pad_sequences(seqs, max_len=None):
    """Pad POI sequences to the same length."""
    if max_len is None:
        max_len = max(len(s) for s in seqs)
    max_len = min(max_len, 50)  # cap at 50

    padded = []
    lengths = []
    for s in seqs:
        s = s[:max_len]
        lengths.append(len(s))
        padded.append(s + [0] * (max_len - len(s)))
    return padded, lengths


def train_model(model, train_data, metadata, device, epochs=30, lr=5e-4, batch_size=128):
    """Train a model (GRU-AE, LSTM-AE, or RADD encoder) on normal trajectories."""
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Flatten training data
    all_samples = []
    for uid, trajs in train_data.items():
        for traj in trajs:
            if len(traj['poi_seq']) >= 2:
                all_samples.append((uid, traj))

    print(f"  Training on {len(all_samples)} trajectories, {epochs} epochs")

    model.train()
    for epoch in range(epochs):
        np.random.shuffle(all_samples)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, len(all_samples), batch_size):
            batch = all_samples[i:i + batch_size]
            seqs = [s[1]['poi_seq'] for s in batch]
            buckets = [s[1]['time_bucket'] for s in batch]

            padded, lengths = pad_sequences(seqs)
            poi_tensor = torch.LongTensor(padded).to(device)
            bucket_tensor = torch.LongTensor(buckets).to(device)
            length_tensor = torch.LongTensor(lengths)

            optimizer.zero_grad()
            loss, h = model.reconstruction_loss(poi_tensor, bucket_tensor, length_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            # Update personal prototypes for RADD
            if isinstance(model, RADD) and h is not None:
                with torch.no_grad():
                    for j, (uid, traj) in enumerate(batch):
                        model.update_prototypes(uid, h[j:j+1], traj['time_bucket'])

        avg_loss = total_loss / max(n_batches, 1)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    return model


def evaluate_model(model, test_augmented, metadata, device, model_type='radd',
                   graph=None, friend_sim=None, train_data=None):
    """Evaluate anomaly detection performance."""
    model.eval()
    all_scores = []
    all_labels = []
    all_types = []

    with torch.no_grad():
        for uid, samples in test_augmented.items():
            for traj, label, anom_type in samples:
                seq = traj['poi_seq']
                if len(seq) < 2:
                    continue
                bucket = traj['time_bucket']

                poi_tensor = torch.LongTensor([seq[:50]]).to(device)
                bucket_tensor = torch.LongTensor([bucket]).to(device)
                length_tensor = torch.LongTensor([min(len(seq), 50)])

                if model_type in ('radd', 'radd_adaptive', 'radd_personal_only',
                                  'radd_social_only', 'radd_no_disagree', 'radd_fixed_alpha'):
                    # All RADD variants use the same compute path
                    history_len = len(train_data.get(uid, []))
                    friend_count = len(list(graph.neighbors(uid))) if uid in graph else 0
                    sims = friend_sim.get(uid, {})
                    avg_sim = np.mean(list(sims.values())) if sims else 0.0

                    # Select fusion mode
                    if model_type == 'radd_adaptive':
                        fusion_mode = 'adaptive'
                    elif model_type == 'radd':
                        fusion_mode = 'learned'
                    else:
                        fusion_mode = 'fixed'

                    score, s_p, s_s = model.compute_anomaly_score(
                        uid, poi_tensor, bucket_tensor, length_tensor,
                        history_len, friend_count, avg_sim,
                        fusion_mode=fusion_mode
                    )

                    if model_type in ('radd', 'radd_adaptive'):
                        all_scores.append(score.item())
                    elif model_type == 'radd_personal_only':
                        all_scores.append(s_p.item())
                    elif model_type == 'radd_social_only':
                        all_scores.append(s_s.item())
                    elif model_type == 'radd_no_disagree':
                        import torch as th
                        sp_n = th.sigmoid(s_p - 1.0).item()
                        ss_n = s_s.item() / 2.0
                        all_scores.append(0.5 * sp_n + 0.5 * ss_n)
                    elif model_type == 'radd_fixed_alpha':
                        import torch as th
                        sp_n = th.sigmoid(s_p - 1.0).item()
                        ss_n = s_s.item() / 2.0
                        all_scores.append(0.5 * sp_n + 0.5 * ss_n + model.beta * abs(sp_n - ss_n))
                elif model_type == 'early_fusion':
                    # EarlyFusionAE baseline
                    score = model.compute_anomaly_score(uid, poi_tensor, bucket_tensor, length_tensor)
                    all_scores.append(score.item())
                else:
                    # Baseline models (GRU-AE, LSTM-AE)
                    score = model.compute_anomaly_score(poi_tensor, bucket_tensor, length_tensor)
                    all_scores.append(score.item())

                all_labels.append(label)
                all_types.append(anom_type)

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    all_types = np.array(all_types)

    # Handle NaN/Inf
    all_scores = np.nan_to_num(all_scores, nan=0.0, posinf=1.0, neginf=0.0)

    results = {}

    # Overall metrics
    if len(np.unique(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_scores)
        # Find optimal threshold
        thresholds = np.percentile(all_scores, np.arange(1, 100))
        best_f1 = 0
        for t in thresholds:
            preds = (all_scores > t).astype(int)
            f1 = f1_score(all_labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        preds = (all_scores > best_t).astype(int)
        results['overall'] = {
            'auc': auc,
            'f1': best_f1,
            'precision': precision_score(all_labels, preds, zero_division=0),
            'recall': recall_score(all_labels, preds, zero_division=0),
        }

        # Per-anomaly-type metrics
        for atype in ['poi_replace', 'time_shift', 'splice', 'social_inconsist']:
            mask = (all_types == atype) | (all_types == 'normal')
            if mask.sum() > 0 and len(np.unique(all_labels[mask])) > 1:
                auc_t = roc_auc_score(all_labels[mask], all_scores[mask])
                preds_t = (all_scores[mask] > best_t).astype(int)
                f1_t = f1_score(all_labels[mask], preds_t, zero_division=0)
                results[atype] = {'auc': auc_t, 'f1': f1_t}

    return results


def run_block1_main_results(train_data, test_data, metadata, device, seed=42, train_data_full=None):
    """Block 1: Main results - compare RADD vs baselines."""
    if train_data_full is None:
        train_data_full = train_data
    set_seed(seed)
    graph = metadata['graph']
    friend_sim = metadata['friend_sim']
    num_pois = metadata['num_pois']

    # Inject anomalies
    injector = AnomalyInjector(train_data, metadata, seed=seed)
    test_augmented = injector.inject_anomalies(test_data)

    results = {}

    # 1. GRU-AE baseline
    print("\n  [1/4] Training GRU-AE baseline...")
    gru_ae = GRUAutoencoder(num_pois, embed_dim=64, hidden_dim=64)
    gru_ae = train_model(gru_ae, train_data, metadata, device)
    r = evaluate_model(gru_ae, test_augmented, metadata, device, model_type='baseline')
    results['GRU-AE'] = r
    print(f"    GRU-AE: AUC={r['overall']['auc']:.4f}, F1={r['overall']['f1']:.4f}")
    del gru_ae
    torch.cuda.empty_cache()

    # 2. LSTM-AE baseline
    print("\n  [2/4] Training LSTM-AE baseline...")
    lstm_ae = LSTMAutoencoder(num_pois, embed_dim=64, hidden_dim=64)
    lstm_ae = train_model(lstm_ae, train_data, metadata, device)
    r = evaluate_model(lstm_ae, test_augmented, metadata, device, model_type='baseline')
    results['LSTM-AE'] = r
    print(f"    LSTM-AE: AUC={r['overall']['auc']:.4f}, F1={r['overall']['f1']:.4f}")
    del lstm_ae
    torch.cuda.empty_cache()

    # 3. RADD (ours)
    print("\n  [3/4] Training RADD...")
    radd = RADD(num_pois, embed_dim=64, hidden_dim=64)
    radd = train_model(radd, train_data, metadata, device)
    print("    Building social prototypes...")
    radd.build_social_prototypes(graph, friend_sim, train_data_full)
    r = evaluate_model(radd, test_augmented, metadata, device,
                       model_type='radd', graph=graph, friend_sim=friend_sim,
                       train_data=train_data)
    results['RADD'] = r
    print(f"    RADD: AUC={r['overall']['auc']:.4f}, F1={r['overall']['f1']:.4f}")

    # 4. RADD-Adaptive (per-user α based on social coverage)
    print("\n  [4/6] Evaluating RADD-Adaptive (per-user α)...")
    r = evaluate_model(radd, test_augmented, metadata, device,
                       model_type='radd_adaptive', graph=graph,
                       friend_sim=friend_sim, train_data=train_data)
    results['RADD-Adaptive'] = r
    print(f"    RADD-Adaptive: AUC={r['overall']['auc']:.4f}, F1={r['overall']['f1']:.4f}")

    # 5. Ablation variants (reuse RADD model)
    print("\n  [5/6] Running ablation variants...")
    for variant_name, variant_type in [
        ('Personal-only', 'radd_personal_only'),
        ('Social-only', 'radd_social_only'),
        ('Fusion (no disagree)', 'radd_no_disagree'),
        ('Fusion (fixed α)', 'radd_fixed_alpha'),
    ]:
        r = evaluate_model(radd, test_augmented, metadata, device,
                           model_type=variant_type, graph=graph,
                           friend_sim=friend_sim, train_data=train_data)
        results[variant_name] = r
        if 'overall' in r:
            print(f"    {variant_name}: AUC={r['overall']['auc']:.4f}, F1={r['overall']['f1']:.4f}")

    del radd
    torch.cuda.empty_cache()

    # 6. Early Fusion baseline
    print("\n  [6/6] Training Early Fusion baseline...")
    early_fusion = EarlyFusionAE(num_pois, embed_dim=64, hidden_dim=64)
    early_fusion = train_model(early_fusion, train_data, metadata, device)
    print("    Building social prototypes for early fusion...")
    early_fusion.build_social_prototypes(graph, friend_sim, train_data_full)
    r = evaluate_model(early_fusion, test_augmented, metadata, device,
                       model_type='early_fusion', graph=graph,
                       friend_sim=friend_sim, train_data=train_data)
    results['Early Fusion'] = r
    print(f"    Early Fusion: AUC={r['overall']['auc']:.4f}, F1={r['overall']['f1']:.4f}")
    del early_fusion
    torch.cuda.empty_cache()

    return results


def run_block4_robustness(train_data, test_data, metadata, device, seed=42, train_data_full=None):
    """Block 4: Social robustness - edge dropout."""
    if train_data_full is None:
        train_data_full = train_data
    import copy
    set_seed(seed)
    graph = metadata['graph']
    friend_sim = metadata['friend_sim']
    num_pois = metadata['num_pois']

    injector = AnomalyInjector(train_data, metadata, seed=seed)
    test_augmented = injector.inject_anomalies(test_data)

    # Train RADD once
    radd = RADD(num_pois, embed_dim=64, hidden_dim=64)
    radd = train_model(radd, train_data, metadata, device, epochs=30)

    results = {}
    for dropout_rate in [0.0, 0.2, 0.4, 0.6, 0.8]:
        # Create dropout graph
        edges = list(graph.edges())
        n_drop = int(len(edges) * dropout_rate)
        rng = np.random.RandomState(seed)
        drop_indices = rng.choice(len(edges), n_drop, replace=False)
        keep_edges = [e for i, e in enumerate(edges) if i not in set(drop_indices)]

        dropped_graph = graph.__class__()
        dropped_graph.add_nodes_from(graph.nodes())
        dropped_graph.add_edges_from(keep_edges)

        # Rebuild social prototypes with dropped graph
        radd.build_social_prototypes(dropped_graph, friend_sim, train_data_full)

        r = evaluate_model(radd, test_augmented, metadata, device,
                           model_type='radd', graph=dropped_graph,
                           friend_sim=friend_sim, train_data=train_data)
        results[f'dropout_{dropout_rate}'] = r
        if 'overall' in r:
            print(f"    Dropout {dropout_rate}: AUC={r['overall']['auc']:.4f}, F1={r['overall']['f1']:.4f}")

    del radd
    torch.cuda.empty_cache()
    return results


def run_block5_topk(train_data, test_data, metadata, device, seed=42, train_data_full=None):
    """Block 5: Top-k sensitivity."""
    if train_data_full is None:
        train_data_full = train_data
    set_seed(seed)
    graph = metadata['graph']
    friend_sim = metadata['friend_sim']
    num_pois = metadata['num_pois']

    injector = AnomalyInjector(train_data, metadata, seed=seed)
    test_augmented = injector.inject_anomalies(test_data)

    results = {}
    for k in [3, 5, 10, 20]:
        radd = RADD(num_pois, embed_dim=64, hidden_dim=64, top_k=k)
        radd = train_model(radd, train_data, metadata, device, epochs=30)
        radd.build_social_prototypes(graph, friend_sim, train_data_full)

        r = evaluate_model(radd, test_augmented, metadata, device,
                           model_type='radd', graph=graph,
                           friend_sim=friend_sim, train_data=train_data)
        results[f'top_{k}'] = r
        if 'overall' in r:
            print(f"    Top-{k}: AUC={r['overall']['auc']:.4f}, F1={r['overall']['f1']:.4f}")

        del radd
        torch.cuda.empty_cache()

    return results


def run_block3_sparse(train_data, test_data, metadata, device, seed=42, train_data_full=None):
    """Block 3: Sparse user analysis."""
    if train_data_full is None:
        train_data_full = train_data
    set_seed(seed)
    graph = metadata['graph']
    friend_sim = metadata['friend_sim']
    num_pois = metadata['num_pois']

    injector = AnomalyInjector(train_data, metadata, seed=seed)
    test_augmented = injector.inject_anomalies(test_data)

    # Train GRU-AE and RADD
    gru_ae = GRUAutoencoder(num_pois, embed_dim=64, hidden_dim=64)
    gru_ae = train_model(gru_ae, train_data, metadata, device, epochs=30)

    radd = RADD(num_pois, embed_dim=64, hidden_dim=64)
    radd = train_model(radd, train_data, metadata, device, epochs=30)
    radd.build_social_prototypes(graph, friend_sim, train_data_full)

    # Group users by history length
    groups = {
        '20-50': [], '50-100': [], '100-200': [], '200+': []
    }
    for uid in test_augmented:
        n = len(train_data.get(uid, []))
        if n < 50:
            groups['20-50'].append(uid)
        elif n < 100:
            groups['50-100'].append(uid)
        elif n < 200:
            groups['100-200'].append(uid)
        else:
            groups['200+'].append(uid)

    results = {}
    for group_name, uids in groups.items():
        if not uids:
            continue
        subset = {uid: test_augmented[uid] for uid in uids if uid in test_augmented}
        if not subset:
            continue

        r_gru = evaluate_model(gru_ae, subset, metadata, device, model_type='baseline')
        r_radd = evaluate_model(radd, subset, metadata, device, model_type='radd',
                                graph=graph, friend_sim=friend_sim, train_data=train_data)

        results[group_name] = {
            'n_users': len(uids),
            'GRU-AE': r_gru.get('overall', {}),
            'RADD': r_radd.get('overall', {}),
        }
        gru_auc = r_gru.get('overall', {}).get('auc', 0)
        radd_auc = r_radd.get('overall', {}).get('auc', 0)
        print(f"    {group_name} ({len(uids)} users): GRU-AE AUC={gru_auc:.4f}, RADD AUC={radd_auc:.4f}, Δ={radd_auc-gru_auc:+.4f}")

    del gru_ae, radd
    torch.cuda.empty_cache()
    return results


def run_block6_friend_count(train_data, test_data, metadata, device, seed=42, train_data_full=None):
    """Block 6: Within-dataset friend-count analysis.
    Groups users by number of friends to show correlation between social density and RADD benefit.
    """
    if train_data_full is None:
        train_data_full = train_data
    set_seed(seed)
    graph = metadata['graph']
    friend_sim = metadata['friend_sim']
    num_pois = metadata['num_pois']

    injector = AnomalyInjector(train_data, metadata, seed=seed)
    test_augmented = injector.inject_anomalies(test_data)

    # Train models
    gru_ae = GRUAutoencoder(num_pois, embed_dim=64, hidden_dim=64)
    gru_ae = train_model(gru_ae, train_data, metadata, device, epochs=30)

    radd = RADD(num_pois, embed_dim=64, hidden_dim=64)
    radd = train_model(radd, train_data, metadata, device, epochs=30)
    radd.build_social_prototypes(graph, friend_sim, train_data_full)

    # Group users by friend count
    groups = {
        '0-3': [], '3-5': [], '5-10': [], '10-20': [], '20+': []
    }
    for uid in test_augmented:
        n_friends = len(list(graph.neighbors(uid))) if uid in graph else 0
        if n_friends <= 3:
            groups['0-3'].append(uid)
        elif n_friends <= 5:
            groups['3-5'].append(uid)
        elif n_friends <= 10:
            groups['5-10'].append(uid)
        elif n_friends <= 20:
            groups['10-20'].append(uid)
        else:
            groups['20+'].append(uid)

    results = {}
    for group_name, uids in groups.items():
        if not uids:
            continue
        subset = {uid: test_augmented[uid] for uid in uids if uid in test_augmented}
        if not subset:
            continue

        r_gru = evaluate_model(gru_ae, subset, metadata, device, model_type='baseline')
        r_radd = evaluate_model(radd, subset, metadata, device, model_type='radd',
                                graph=graph, friend_sim=friend_sim, train_data=train_data)
        r_adaptive = evaluate_model(radd, subset, metadata, device, model_type='radd_adaptive',
                                    graph=graph, friend_sim=friend_sim, train_data=train_data)

        results[group_name] = {
            'n_users': len(uids),
            'GRU-AE': r_gru.get('overall', {}),
            'RADD': r_radd.get('overall', {}),
            'RADD-Adaptive': r_adaptive.get('overall', {}),
        }
        gru_auc = r_gru.get('overall', {}).get('auc', 0)
        radd_auc = r_radd.get('overall', {}).get('auc', 0)
        adapt_auc = r_adaptive.get('overall', {}).get('auc', 0)
        print(f"    {group_name} ({len(uids)} users): GRU-AE={gru_auc:.4f}, RADD={radd_auc:.4f}, Adaptive={adapt_auc:.4f}")

    del gru_ae, radd
    torch.cuda.empty_cache()
    return results


def compute_bootstrap_ci(scores, labels, n_bootstrap=1000, ci=0.95, seed=42):
    """Compute bootstrap confidence interval for AUC."""
    rng = np.random.RandomState(seed)
    aucs = []
    n = len(scores)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(labels[idx])) < 2:
            continue
        aucs.append(roc_auc_score(labels[idx], scores[idx]))
    aucs = sorted(aucs)
    lo = aucs[int((1 - ci) / 2 * len(aucs))]
    hi = aucs[int((1 + ci) / 2 * len(aucs))]
    return np.mean(aucs), lo, hi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--dataset', default='gowalla', choices=['gowalla', 'brightkite'])
    parser.add_argument('--results_dir', default='./results')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456, 789, 1024])
    parser.add_argument('--blocks', type=str, nargs='+',
                        default=['block1', 'block3', 'block4', 'block5', 'block6'],
                        help='Which experiment blocks to run')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--sample_ratio', type=float, default=1.0,
                        help='Sample ratio of users (0-1). Use 0.2 to sample 20%% users.')
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load and preprocess data
    print(f"\n{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"{'='*60}")
    train_data, test_data, metadata = preprocess_dataset(args.data_dir, args.dataset)

    # Sample users if requested
    train_data_full = train_data  # keep full for social prototype building
    if args.sample_ratio < 1.0:
        all_uids = sorted(train_data.keys())
        np.random.seed(42)
        n_sample = max(500, int(len(all_uids) * args.sample_ratio))
        sampled_uids = set(np.random.choice(all_uids, n_sample, replace=False))
        # Train encoder only on sampled users
        train_data = {uid: trajs for uid, trajs in train_data.items() if uid in sampled_uids}
        test_data = {uid: trajs for uid, trajs in test_data.items() if uid in sampled_uids}
        # Rebuild friend_sim for sampled users (using full data for friend info)
        from data_loader import compute_friend_mobility_sim
        graph = metadata['graph']
        metadata['friend_sim'] = compute_friend_mobility_sim(train_data_full, graph)
        print(f"Sampled {len(sampled_uids)} users (encoder training)")
        print(f"Train trajectories: {sum(len(v) for v in train_data.values())}")
        print(f"Test trajectories: {sum(len(v) for v in test_data.values())}")
    else:
        train_data_full = train_data

    all_results = {}

    for block in args.blocks:
        print(f"\n{'='*60}")
        print(f"Running {block}")
        print(f"{'='*60}")

        if block == 'block1':
            # Run with multiple seeds for main results
            seed_results = []
            for seed in args.seeds:
                print(f"\n--- Seed {seed} ---")
                r = run_block1_main_results(train_data, test_data, metadata, device, seed=seed, train_data_full=train_data_full)
                seed_results.append(r)

            # Aggregate results across seeds
            aggregated = {}
            for method in seed_results[0]:
                auc_list = [sr[method]['overall']['auc'] for sr in seed_results if 'overall' in sr.get(method, {})]
                f1_list = [sr[method]['overall']['f1'] for sr in seed_results if 'overall' in sr.get(method, {})]
                if auc_list:
                    aggregated[method] = {
                        'auc_mean': np.mean(auc_list),
                        'auc_std': np.std(auc_list),
                        'f1_mean': np.mean(f1_list),
                        'f1_std': np.std(f1_list),
                    }

                    # Per-type results from last seed
                    for atype in ['poi_replace', 'time_shift', 'splice', 'social_inconsist']:
                        type_aucs = [sr[method].get(atype, {}).get('auc', 0) for sr in seed_results if atype in sr.get(method, {})]
                        if type_aucs:
                            aggregated[method][f'{atype}_auc'] = np.mean(type_aucs)

            all_results['block1'] = aggregated

            # Print summary table
            print(f"\n{'='*60}")
            print(f"Block 1 Summary: {args.dataset}")
            print(f"{'='*60}")
            print(f"{'Method':<25} {'AUC':>12} {'F1':>12}")
            print("-" * 50)
            for method, r in sorted(aggregated.items(), key=lambda x: x[1].get('auc_mean', 0), reverse=True):
                print(f"{method:<25} {r['auc_mean']:.4f}±{r['auc_std']:.4f} {r['f1_mean']:.4f}±{r['f1_std']:.4f}")

        elif block == 'block3':
            r = run_block3_sparse(train_data, test_data, metadata, device, train_data_full=train_data_full)
            all_results['block3'] = r

        elif block == 'block4':
            r = run_block4_robustness(train_data, test_data, metadata, device, train_data_full=train_data_full)
            all_results['block4'] = r

        elif block == 'block5':
            r = run_block5_topk(train_data, test_data, metadata, device, train_data_full=train_data_full)
            all_results['block5'] = r

        elif block == 'block6':
            r = run_block6_friend_count(train_data, test_data, metadata, device, train_data_full=train_data_full)
            all_results['block6'] = r

    # Save results
    results_file = os.path.join(args.results_dir, f'{args.dataset}_results.json')

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=convert)

    print(f"\nResults saved to {results_file}")


if __name__ == '__main__':
    main()
