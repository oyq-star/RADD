"""Anomaly injection for trajectory anomaly detection evaluation.

Four anomaly types:
1. POI Replacement: Replace POIs with random globally popular POIs
2. Time Shift: Move trajectory to unusual time bucket
3. Subtrajectory Splice: Replace a segment with another user's segment
4. Social Inconsistency: Replace POIs with locations far from user AND friends
"""

import random
import numpy as np
from collections import Counter, defaultdict


class AnomalyInjector:
    def __init__(self, train_data, metadata, seed=42):
        self.train_data = train_data
        self.metadata = metadata
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.num_pois = metadata['num_pois']

        # Precompute POI popularity
        poi_counts = Counter()
        for uid, trajs in train_data.items():
            for t in trajs:
                poi_counts.update(t['poi_seq'])

        self.popular_pois = [p for p, _ in poi_counts.most_common(500)]

        # Precompute per-user POI sets and per-user+friends POI sets
        self.user_pois = {}
        self.user_friend_pois = {}
        graph = metadata['graph']

        for uid, trajs in train_data.items():
            pois = set()
            for t in trajs:
                pois.update(t['poi_seq'])
            self.user_pois[uid] = pois

            # Friends' POIs
            friend_pois = set(pois)
            if uid in graph:
                for fid in graph.neighbors(uid):
                    if fid in train_data:
                        for t in train_data[fid]:
                            friend_pois.update(t['poi_seq'])
            self.user_friend_pois[uid] = friend_pois

        # All POIs for social inconsistency
        self.all_pois = list(range(self.num_pois))

        # Collect all trajectories for splice
        self.all_trajs = []
        for uid, trajs in train_data.items():
            for t in trajs:
                if len(t['poi_seq']) >= 2:
                    self.all_trajs.append((uid, t))

        # Per-user time bucket distribution
        self.user_buckets = defaultdict(Counter)
        for uid, trajs in train_data.items():
            for t in trajs:
                self.user_buckets[uid][t['time_bucket']] += 1

    def inject_poi_replacement(self, traj, uid):
        """Replace 1-3 POIs with random popular POIs not in user's history."""
        seq = list(traj['poi_seq'])
        if len(seq) == 0:
            return traj, False

        n_replace = min(self.rng.randint(1, 3), len(seq))
        positions = self.rng.sample(range(len(seq)), n_replace)
        user_pois = self.user_pois.get(uid, set())

        for pos in positions:
            candidates = [p for p in self.popular_pois if p not in user_pois]
            if candidates:
                seq[pos] = self.rng.choice(candidates)

        new_traj = dict(traj)
        new_traj['poi_seq'] = seq
        return new_traj, True

    def inject_time_shift(self, traj, uid):
        """Shift trajectory to the least common time bucket for this user."""
        bucket_counts = self.user_buckets.get(uid, Counter())
        if not bucket_counts:
            return traj, False

        # Find least common bucket
        all_buckets = set(range(6))
        used_buckets = set(bucket_counts.keys())
        unused = all_buckets - used_buckets

        if unused:
            new_bucket = self.rng.choice(list(unused))
        else:
            new_bucket = min(bucket_counts, key=bucket_counts.get)
            if new_bucket == traj['time_bucket']:
                # Pick second least common
                sorted_buckets = sorted(bucket_counts, key=bucket_counts.get)
                new_bucket = sorted_buckets[1] if len(sorted_buckets) > 1 else sorted_buckets[0]

        if new_bucket == traj['time_bucket']:
            return traj, False

        new_traj = dict(traj)
        new_traj['time_bucket'] = new_bucket
        return new_traj, True

    def inject_splice(self, traj, uid):
        """Replace a segment with a random other user's segment."""
        seq = list(traj['poi_seq'])
        if len(seq) < 3:
            return traj, False

        # Find a donor trajectory from a different user
        attempts = 0
        while attempts < 10:
            donor_uid, donor_traj = self.rng.choice(self.all_trajs)
            if donor_uid != uid and len(donor_traj['poi_seq']) >= 2:
                break
            attempts += 1
        else:
            return traj, False

        # Replace a segment
        splice_len = max(1, len(seq) // 3)
        start = self.rng.randint(0, len(seq) - splice_len)
        donor_seq = donor_traj['poi_seq']
        donor_start = self.rng.randint(0, max(0, len(donor_seq) - splice_len))
        donor_segment = donor_seq[donor_start:donor_start + splice_len]

        seq[start:start + splice_len] = donor_segment
        new_traj = dict(traj)
        new_traj['poi_seq'] = seq
        return new_traj, True

    def inject_social_inconsistency(self, traj, uid):
        """Replace POIs with locations far from both user AND friends' patterns."""
        seq = list(traj['poi_seq'])
        if len(seq) == 0:
            return traj, False

        friend_pois = self.user_friend_pois.get(uid, set())
        # Find POIs that neither user nor friends visit
        candidates = [p for p in self.all_pois if p not in friend_pois]
        if len(candidates) < 2:
            return traj, False

        n_replace = min(self.rng.randint(1, 3), len(seq))
        positions = self.rng.sample(range(len(seq)), n_replace)
        for pos in positions:
            seq[pos] = self.rng.choice(candidates)

        new_traj = dict(traj)
        new_traj['poi_seq'] = seq
        return new_traj, True

    def inject_anomalies(self, test_data, anomaly_ratio=0.1):
        """Inject anomalies into test data.

        Returns:
            augmented_data: dict uid -> list of (traj, label, anomaly_type)
                label: 0=normal, 1=anomaly
                anomaly_type: 'normal'/'poi_replace'/'time_shift'/'splice'/'social_inconsist'
        """
        injectors = [
            ('poi_replace', self.inject_poi_replacement),
            ('time_shift', self.inject_time_shift),
            ('splice', self.inject_splice),
            ('social_inconsist', self.inject_social_inconsistency),
        ]

        augmented = {}
        stats = Counter()

        for uid, trajs in test_data.items():
            user_data = []
            n_per_type = max(1, int(len(trajs) * anomaly_ratio))

            # Add all normal trajectories
            for t in trajs:
                user_data.append((t, 0, 'normal'))
                stats['normal'] += 1

            # Inject each anomaly type
            for anom_name, injector in injectors:
                selected = self.rng.sample(range(len(trajs)), min(n_per_type, len(trajs)))
                for idx in selected:
                    new_traj, success = injector(trajs[idx], uid)
                    if success:
                        user_data.append((new_traj, 1, anom_name))
                        stats[anom_name] += 1

            augmented[uid] = user_data

        print(f"Anomaly injection stats: {dict(stats)}")
        total = sum(stats.values())
        anom = total - stats['normal']
        print(f"Total: {total}, Normal: {stats['normal']}, Anomaly: {anom} ({anom/total*100:.1f}%)")
        return augmented
