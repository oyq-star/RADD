"""Data loading and preprocessing for RADD experiments.

Handles Gowalla and Brightkite datasets from SNAP Stanford.
- Parses check-in data and friendship edges
- Filters users (>=20 check-ins, >=3 friends)
- Builds daily trajectories
- Temporal train/test split (80/20)
- Computes friend mobility similarity for top-k selection
"""

import gzip
import os
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
from datetime import datetime


def load_checkins(filepath):
    """Load check-in data from SNAP format.
    Format: [user] [check-in time] [latitude] [longitude] [location id]
    """
    records = []
    opener = gzip.open if filepath.endswith('.gz') else open
    with opener(filepath, 'rt', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 5:
                continue
            try:
                user_id = int(parts[0])
                timestamp = parts[1]
                lat = float(parts[2])
                lon = float(parts[3])
                loc_id = parts[4]
                records.append((user_id, timestamp, lat, lon, loc_id))
            except (ValueError, IndexError):
                continue
    df = pd.DataFrame(records, columns=['user_id', 'timestamp', 'lat', 'lon', 'loc_id'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
    return df


def load_edges(filepath):
    """Load friendship edges from SNAP format.
    Format: [user1] [user2]
    """
    edges = []
    opener = gzip.open if filepath.endswith('.gz') else open
    with opener(filepath, 'rt', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
            try:
                u, v = int(parts[0]), int(parts[1])
                edges.append((u, v))
            except ValueError:
                continue
    G = nx.Graph()
    G.add_edges_from(edges)
    return G


def build_poi_vocab(df, max_pois=10000):
    """Build POI vocabulary (location_id -> index).
    Keep only the top max_pois most frequent POIs; rest map to UNK (index 1).
    Index 0 is reserved for padding.
    """
    poi_counts = df['loc_id'].value_counts()
    top_pois = poi_counts.head(max_pois).index.tolist()
    loc2idx = {loc: idx + 2 for idx, loc in enumerate(top_pois)}  # 0=PAD, 1=UNK
    return loc2idx


def get_time_bucket(dt):
    """Map datetime to time bucket index (0-5).
    {weekday, weekend} x {morning[6-12), afternoon[12-18), evening[18-6)}
    """
    hour = dt.hour
    is_weekend = 1 if dt.weekday() >= 5 else 0
    if 6 <= hour < 12:
        period = 0  # morning
    elif 12 <= hour < 18:
        period = 1  # afternoon
    else:
        period = 2  # evening/night
    return is_weekend * 3 + period


def build_daily_trajectories(df, loc2idx):
    """Group check-ins into daily trajectories.
    Returns dict: user_id -> list of (date, time_bucket, [poi_indices], [timestamps])
    """
    df = df.copy()
    df['date'] = df['timestamp'].dt.date
    df['poi_idx'] = df['loc_id'].map(loc2idx).fillna(1).astype(int)  # UNK=1

    user_trajs = defaultdict(list)
    for (uid, date), group in df.groupby(['user_id', 'date']):
        group = group.sort_values('timestamp')
        poi_seq = group['poi_idx'].tolist()
        ts_list = group['timestamp'].tolist()
        # Use the median time for the bucket
        median_ts = ts_list[len(ts_list) // 2]
        bucket = get_time_bucket(median_ts)
        user_trajs[uid].append({
            'date': date,
            'time_bucket': bucket,
            'poi_seq': poi_seq,
            'timestamps': ts_list,
        })

    # Sort each user's trajectories by date
    for uid in user_trajs:
        user_trajs[uid].sort(key=lambda x: x['date'])
    return dict(user_trajs)


def compute_friend_mobility_sim(user_trajs, G):
    """Compute Jaccard similarity of visited POIs between friends.
    Returns dict: user_id -> {friend_id: similarity}
    """
    # Build POI sets per user
    user_pois = {}
    for uid, trajs in user_trajs.items():
        pois = set()
        for t in trajs:
            pois.update(t['poi_seq'])
        user_pois[uid] = pois

    friend_sim = defaultdict(dict)
    for uid in user_trajs:
        if uid not in G:
            continue
        for fid in G.neighbors(uid):
            if fid not in user_pois:
                continue
            intersection = len(user_pois[uid] & user_pois[fid])
            union = len(user_pois[uid] | user_pois[fid])
            sim = intersection / union if union > 0 else 0.0
            friend_sim[uid][fid] = sim
    return dict(friend_sim)


def preprocess_dataset(data_dir, dataset_name, min_checkins=20, min_friends=3,
                       train_ratio=0.8):
    """Full preprocessing pipeline.

    Returns:
        train_data: dict with user trajectories for training
        test_data: dict with user trajectories for testing
        metadata: dict with vocab, graph, friend similarities, etc.
    """
    cache_path = os.path.join(data_dir, f'{dataset_name}_processed.pkl')
    if os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    print(f"Processing {dataset_name}...")

    # Load raw data
    checkin_file = os.path.join(data_dir, f'{dataset_name}_checkins.txt.gz')
    edges_file = os.path.join(data_dir, f'{dataset_name}_edges.txt.gz')

    # Try without .gz if not found
    if not os.path.exists(checkin_file):
        checkin_file = checkin_file.replace('.gz', '')
    if not os.path.exists(edges_file):
        edges_file = edges_file.replace('.gz', '')

    print("  Loading check-ins...")
    df = load_checkins(checkin_file)
    print(f"  Raw check-ins: {len(df)}")

    print("  Loading friendship graph...")
    G = load_edges(edges_file)
    print(f"  Raw edges: {G.number_of_edges()}, nodes: {G.number_of_nodes()}")

    # Filter users with enough check-ins
    user_counts = df['user_id'].value_counts()
    valid_users_checkins = set(user_counts[user_counts >= min_checkins].index)
    print(f"  Users with >= {min_checkins} check-ins: {len(valid_users_checkins)}")

    # Filter users with enough friends (who also have enough check-ins)
    valid_users = set()
    for u in valid_users_checkins:
        if u in G:
            friends_with_data = [f for f in G.neighbors(u) if f in valid_users_checkins]
            if len(friends_with_data) >= min_friends:
                valid_users.add(u)
    print(f"  Users with >= {min_friends} valid friends: {len(valid_users)}")

    # Filter dataframe
    df = df[df['user_id'].isin(valid_users)].reset_index(drop=True)
    print(f"  Filtered check-ins: {len(df)}")

    # Build subgraph
    G_sub = G.subgraph(valid_users).copy()

    # Build POI vocab (top 10K most frequent)
    loc2idx = build_poi_vocab(df, max_pois=10000)
    print(f"  Vocab POIs: {len(loc2idx)} (+ PAD + UNK)")

    # Build daily trajectories
    user_trajs = build_daily_trajectories(df, loc2idx)

    # Temporal split
    train_data = {}
    test_data = {}
    for uid, trajs in user_trajs.items():
        n = len(trajs)
        split_idx = int(n * train_ratio)
        if split_idx < 5 or n - split_idx < 2:
            continue  # skip users with too few train/test trajectories
        train_data[uid] = trajs[:split_idx]
        test_data[uid] = trajs[split_idx:]

    valid_users_final = set(train_data.keys())
    G_final = G_sub.subgraph(valid_users_final).copy()

    print(f"  Final users: {len(valid_users_final)}")
    print(f"  Final edges: {G_final.number_of_edges()}")
    print(f"  Train trajectories: {sum(len(v) for v in train_data.values())}")
    print(f"  Test trajectories: {sum(len(v) for v in test_data.values())}")

    # Compute friend mobility similarity (on train data only)
    friend_sim = compute_friend_mobility_sim(train_data, G_final)

    # Remap user IDs to contiguous indices
    uid_list = sorted(valid_users_final)
    uid2idx = {uid: idx for idx, uid in enumerate(uid_list)}

    metadata = {
        'loc2idx': loc2idx,
        'uid2idx': uid2idx,
        'idx2uid': {v: k for k, v in uid2idx.items()},
        'graph': G_final,
        'friend_sim': friend_sim,
        'num_pois': len(loc2idx) + 2,  # +2 for PAD and UNK
        'num_users': len(uid2idx),
        'dataset_name': dataset_name,
    }

    result = (train_data, test_data, metadata)

    # Cache
    print(f"  Saving cache to {cache_path}")
    with open(cache_path, 'wb') as f:
        pickle.dump(result, f)

    return result


if __name__ == '__main__':
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else './data'
    for name in ['gowalla', 'brightkite']:
        try:
            train, test, meta = preprocess_dataset(data_dir, name)
            print(f"\n{name}: {meta['num_users']} users, {meta['num_pois']} POIs")
        except FileNotFoundError as e:
            print(f"Skipping {name}: {e}")
