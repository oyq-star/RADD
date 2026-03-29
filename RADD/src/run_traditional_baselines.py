"""Run traditional (non-deep-learning) anomaly detection baselines.

Baselines: Isolation Forest, LOF, One-Class SVM
These operate on the trajectory embeddings from the trained GRU encoder.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from data_loader import preprocess_dataset
from anomaly_injection import AnomalyInjector
from model import GRUAutoencoder
from run_experiments import pad_sequences, train_model, set_seed


def extract_embeddings(model, data_dict, device, max_len=50):
    """Extract trajectory embeddings using trained GRU encoder."""
    model.eval()
    embeddings = {}
    with torch.no_grad():
        for uid, items in data_dict.items():
            uid_embs = []
            for item in items:
                if isinstance(item, tuple):
                    traj, label, atype = item
                else:
                    traj = item
                    label, atype = 0, 'normal'

                seq = traj['poi_seq']
                if len(seq) < 2:
                    continue
                bucket = traj['time_bucket']
                poi_tensor = torch.LongTensor([seq[:max_len]]).to(device)
                bucket_tensor = torch.LongTensor([bucket]).to(device)
                length_tensor = torch.LongTensor([min(len(seq), max_len)])

                _, h = model.reconstruction_loss(poi_tensor, bucket_tensor, length_tensor)
                uid_embs.append({
                    'embedding': h[0].cpu().numpy(),
                    'label': label,
                    'atype': atype,
                })
            if uid_embs:
                embeddings[uid] = uid_embs
    return embeddings


def evaluate_traditional(method_name, train_embs_flat, test_embs_flat, test_labels, test_types):
    """Train and evaluate a traditional anomaly detector."""
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_embs_flat)
    test_X = scaler.transform(test_embs_flat)

    if method_name == 'IF':
        clf = IsolationForest(n_estimators=200, contamination=0.1, random_state=42, n_jobs=-1)
        clf.fit(train_X)
        scores = -clf.decision_function(test_X)  # higher = more anomalous
    elif method_name == 'LOF':
        clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True, n_jobs=-1)
        clf.fit(train_X)
        scores = -clf.decision_function(test_X)
    elif method_name == 'OCSVM':
        # Subsample for speed if too large
        if len(train_X) > 50000:
            idx = np.random.choice(len(train_X), 50000, replace=False)
            train_sub = train_X[idx]
        else:
            train_sub = train_X
        clf = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
        clf.fit(train_sub)
        scores = -clf.decision_function(test_X)
    else:
        raise ValueError(f"Unknown method: {method_name}")

    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    labels = np.array(test_labels)
    types = np.array(test_types)

    results = {}
    if len(np.unique(labels)) > 1:
        auc = roc_auc_score(labels, scores)
        thresholds = np.percentile(scores, np.arange(1, 100))
        best_f1 = 0
        best_t = thresholds[0]
        for t in thresholds:
            preds = (scores > t).astype(int)
            f1 = f1_score(labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        preds = (scores > best_t).astype(int)
        results['overall'] = {
            'auc': float(auc),
            'f1': float(best_f1),
            'precision': float(precision_score(labels, preds, zero_division=0)),
            'recall': float(recall_score(labels, preds, zero_division=0)),
        }

        for atype in ['poi_replace', 'time_shift', 'splice', 'social_inconsist']:
            mask = (types == atype) | (types == 'normal')
            if mask.sum() > 0 and len(np.unique(labels[mask])) > 1:
                auc_t = roc_auc_score(labels[mask], scores[mask])
                preds_t = (scores[mask] > best_t).astype(int)
                f1_t = f1_score(labels[mask], preds_t, zero_division=0)
                results[atype] = {'auc': float(auc_t), 'f1': float(f1_t)}

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--dataset', default='gowalla', choices=['gowalla', 'brightkite'])
    parser.add_argument('--results_dir', default='./results')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print(f"\nDataset: {args.dataset}")
    train_data, test_data, metadata = preprocess_dataset(args.data_dir, args.dataset)
    num_pois = metadata['num_pois']

    all_results = {}

    for seed in args.seeds:
        print(f"\n--- Seed {seed} ---")
        set_seed(seed)

        # Train GRU encoder (shared, same as in main experiments)
        print("  Training GRU encoder for embedding extraction...")
        encoder = GRUAutoencoder(num_pois, embed_dim=64, hidden_dim=64)
        encoder = train_model(encoder, train_data, metadata, device, epochs=30)

        # Extract train embeddings
        print("  Extracting train embeddings...")
        train_emb_dict = extract_embeddings(encoder, train_data, device)
        train_embs_flat = []
        for uid, items in train_emb_dict.items():
            for item in items:
                train_embs_flat.append(item['embedding'])
        train_embs_flat = np.array(train_embs_flat)
        print(f"    Train embeddings: {train_embs_flat.shape}")

        # Inject anomalies and extract test embeddings
        injector = AnomalyInjector(train_data, metadata, seed=seed)
        test_augmented = injector.inject_anomalies(test_data)

        print("  Extracting test embeddings...")
        test_embs_flat = []
        test_labels = []
        test_types = []
        with torch.no_grad():
            for uid, samples in test_augmented.items():
                for traj, label, atype in samples:
                    seq = traj['poi_seq']
                    if len(seq) < 2:
                        continue
                    bucket = traj['time_bucket']
                    poi_tensor = torch.LongTensor([seq[:50]]).to(device)
                    bucket_tensor = torch.LongTensor([bucket]).to(device)
                    length_tensor = torch.LongTensor([min(len(seq), 50)])
                    _, h = encoder.reconstruction_loss(poi_tensor, bucket_tensor, length_tensor)
                    test_embs_flat.append(h[0].cpu().numpy())
                    test_labels.append(label)
                    test_types.append(atype)

        test_embs_flat = np.array(test_embs_flat)
        print(f"    Test embeddings: {test_embs_flat.shape}")

        del encoder
        torch.cuda.empty_cache()

        # Run traditional methods
        for method in ['IF', 'LOF', 'OCSVM']:
            print(f"  Running {method}...")
            r = evaluate_traditional(method, train_embs_flat, test_embs_flat, test_labels, test_types)
            if 'overall' in r:
                print(f"    {method}: AUC={r['overall']['auc']:.4f}, F1={r['overall']['f1']:.4f}")

            if method not in all_results:
                all_results[method] = []
            all_results[method].append(r)

    # Aggregate across seeds
    aggregated = {}
    for method, seed_results in all_results.items():
        auc_list = [sr['overall']['auc'] for sr in seed_results if 'overall' in sr]
        f1_list = [sr['overall']['f1'] for sr in seed_results if 'overall' in sr]
        if auc_list:
            aggregated[method] = {
                'auc_mean': float(np.mean(auc_list)),
                'auc_std': float(np.std(auc_list)),
                'f1_mean': float(np.mean(f1_list)),
                'f1_std': float(np.std(f1_list)),
            }
            for atype in ['poi_replace', 'time_shift', 'splice', 'social_inconsist']:
                type_aucs = [sr.get(atype, {}).get('auc', 0) for sr in seed_results if atype in sr]
                if type_aucs:
                    aggregated[method][f'{atype}_auc'] = float(np.mean(type_aucs))

    results_file = os.path.join(args.results_dir, f'{args.dataset}_traditional.json')
    with open(results_file, 'w') as f:
        json.dump(aggregated, f, indent=2)

    print(f"\nResults saved to {results_file}")
    print("\nSummary:")
    for method, r in aggregated.items():
        print(f"  {method}: AUC={r['auc_mean']:.4f}±{r['auc_std']:.4f}, F1={r['f1_mean']:.4f}±{r['f1_std']:.4f}")


if __name__ == '__main__':
    main()
