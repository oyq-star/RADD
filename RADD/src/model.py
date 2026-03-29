"""RADD: Reliability-Aware Dual-view Disagreement for Trajectory Anomaly Detection.

Components:
1. Shared GRU Trajectory Encoder
2. Personal Branch (time-bucketed prototypes)
3. Social Branch (top-k friend prototypes)
4. Reliability-Aware Disagreement Fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict


class TrajectoryEncoder(nn.Module):
    """Shared GRU encoder for check-in trajectories."""

    def __init__(self, num_pois, embed_dim=128, hidden_dim=128):
        super().__init__()
        self.poi_embed = nn.Embedding(num_pois, embed_dim, padding_idx=0)
        self.time_embed = nn.Embedding(6, 16)  # 6 time buckets
        self.gru = nn.GRU(embed_dim + 16, hidden_dim, batch_first=True)
        self.hidden_dim = hidden_dim

    def forward(self, poi_seq, time_bucket, lengths=None):
        """
        poi_seq: (batch, max_len) - POI indices
        time_bucket: (batch,) - time bucket index
        lengths: (batch,) - actual sequence lengths
        Returns: (batch, hidden_dim) - trajectory embedding
        """
        poi_emb = self.poi_embed(poi_seq)  # (B, L, embed_dim)
        time_emb = self.time_embed(time_bucket).unsqueeze(1)  # (B, 1, 16)
        time_emb = time_emb.expand(-1, poi_seq.size(1), -1)  # (B, L, 16)
        x = torch.cat([poi_emb, time_emb], dim=-1)  # (B, L, embed_dim+16)

        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)

        _, h = self.gru(x)  # h: (1, B, hidden_dim)
        return h.squeeze(0)  # (B, hidden_dim)


class TrajectoryDecoder(nn.Module):
    """GRU decoder for trajectory reconstruction."""

    def __init__(self, num_pois, hidden_dim=128, embed_dim=128):
        super().__init__()
        self.poi_embed = nn.Embedding(num_pois, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_pois)

    def forward(self, poi_seq, h0):
        """
        poi_seq: (B, L) input POI sequence (shifted right for teacher forcing)
        h0: (B, hidden_dim) initial hidden state from encoder
        Returns: (B, L, num_pois) logits
        """
        emb = self.poi_embed(poi_seq)
        output, _ = self.gru(emb, h0.unsqueeze(0))
        logits = self.fc(output)
        return logits


class RADD(nn.Module):
    """Reliability-Aware Dual-view Disagreement model."""

    def __init__(self, num_pois, embed_dim=128, hidden_dim=128,
                 beta=0.3, top_k=10, ema_decay=0.95):
        super().__init__()
        self.encoder = TrajectoryEncoder(num_pois, embed_dim, hidden_dim)
        self.decoder = TrajectoryDecoder(num_pois, hidden_dim, embed_dim)
        self.hidden_dim = hidden_dim
        self.beta = beta
        self.top_k = top_k
        self.ema_decay = ema_decay

        # Reliability gate: alpha = sigmoid(w @ features)
        # Features: [log(history_len), log(friend_count), avg_friend_sim]
        self.reliability_gate = nn.Linear(3, 1)
        nn.init.constant_(self.reliability_gate.weight, 0.0)
        nn.init.constant_(self.reliability_gate.bias, 0.0)  # start at alpha=0.5

        # Prototypes will be stored as buffers (not parameters)
        # Initialized during training
        self.personal_protos = {}  # uid -> {bucket: tensor}
        self.social_protos = {}   # uid -> {bucket: tensor}

    def encode(self, poi_seq, time_bucket, lengths=None):
        return self.encoder(poi_seq, time_bucket, lengths)

    def decode(self, poi_seq, h):
        return self.decoder(poi_seq, h)

    def reconstruction_loss(self, poi_seq, time_bucket, lengths=None):
        """Compute reconstruction loss for training."""
        h = self.encode(poi_seq, time_bucket, lengths)

        # Teacher forcing: input is [SOS, poi_0, ..., poi_{n-2}]
        # Target is [poi_0, poi_1, ..., poi_{n-1}]
        # For simplicity, use the same sequence shifted
        decoder_input = poi_seq  # (B, L)
        logits = self.decode(decoder_input, h)  # (B, L, V)

        # Shift: predict next token
        # logits[:, :-1] predicts poi_seq[:, 1:]
        logits = logits[:, :-1].contiguous()
        targets = poi_seq[:, 1:].contiguous()

        # Mask padding
        mask = (targets != 0).float()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                               targets.view(-1), reduction='none')
        loss = loss.view(targets.size())
        loss = (loss * mask).sum() / mask.sum().clamp(min=1)
        return loss, h

    def update_prototypes(self, uid, h, time_bucket, decay=None):
        """Update personal prototype with EMA."""
        if decay is None:
            decay = self.ema_decay

        h_detach = h.detach().squeeze(0)  # ensure 1D: (hidden_dim,)

        if uid not in self.personal_protos:
            self.personal_protos[uid] = {}

        bucket = time_bucket.item() if isinstance(time_bucket, torch.Tensor) else time_bucket

        if bucket in self.personal_protos[uid]:
            old = self.personal_protos[uid][bucket]
            self.personal_protos[uid][bucket] = decay * old + (1 - decay) * h_detach
        else:
            self.personal_protos[uid][bucket] = h_detach.clone()

    def build_social_prototypes(self, graph, friend_sim, train_data):
        """Build social prototypes from top-k friends' embeddings.
        Should be called after training, using trained encoder.
        """
        device = next(self.parameters()).device
        self.social_protos = {}

        # First, encode all users' training trajectories and store per-bucket embeddings
        user_bucket_embeds = defaultdict(lambda: defaultdict(list))

        self.eval()
        with torch.no_grad():
            for uid, trajs in train_data.items():
                for traj in trajs:
                    seq = traj['poi_seq']
                    if len(seq) < 2:
                        continue
                    bucket = traj['time_bucket']
                    poi_tensor = torch.LongTensor([seq]).to(device)
                    bucket_tensor = torch.LongTensor([bucket]).to(device)
                    h = self.encode(poi_tensor, bucket_tensor)
                    user_bucket_embeds[uid][bucket].append(h.squeeze(0))

        # Average per-bucket embeddings for each user
        user_bucket_avg = {}
        for uid in user_bucket_embeds:
            user_bucket_avg[uid] = {}
            for bucket, embeds in user_bucket_embeds[uid].items():
                user_bucket_avg[uid][bucket] = torch.stack(embeds).mean(0)

        # Build social prototype for each user
        for uid in train_data:
            if uid not in graph:
                continue

            # Get friends and their similarity scores
            sims = friend_sim.get(uid, {})
            friends = [(fid, sims.get(fid, 0.0))
                       for fid in graph.neighbors(uid)
                       if fid in user_bucket_avg]

            if not friends:
                continue

            # Top-k by similarity
            friends.sort(key=lambda x: x[1], reverse=True)
            top_friends = friends[:self.top_k]

            # Weighted average per bucket
            self.social_protos[uid] = {}
            for bucket in range(6):
                friend_embeds = []
                friend_weights = []
                for fid, sim in top_friends:
                    if bucket in user_bucket_avg.get(fid, {}):
                        friend_embeds.append(user_bucket_avg[fid][bucket])
                        friend_weights.append(max(sim, 0.01))  # min weight

                if friend_embeds:
                    weights = torch.FloatTensor(friend_weights).to(device)
                    weights = weights / weights.sum()
                    stacked = torch.stack(friend_embeds)
                    self.social_protos[uid][bucket] = (stacked * weights.unsqueeze(1)).sum(0)

    def compute_anomaly_score(self, uid, poi_seq, time_bucket, lengths=None,
                              history_len=None, friend_count=None, avg_friend_sim=None,
                              fusion_mode='adaptive'):
        """Compute anomaly score with dual-view fusion.

        Personal score = reconstruction error (aligned with training objective)
        Social score = cosine distance to friend prototype
        fusion_mode: 'adaptive' (per-user α), 'fixed' (α=0.5), 'learned' (gate)
        Returns: anomaly_score, personal_score, social_score
        """
        device = next(self.parameters()).device
        h = self.encode(poi_seq, time_bucket, lengths)

        bucket = time_bucket.item() if isinstance(time_bucket, torch.Tensor) and time_bucket.dim() == 0 else time_bucket[0].item()

        # Personal score = reconstruction error (same as GRU-AE baseline)
        logits = self.decode(poi_seq, h)
        logits_shift = logits[:, :-1].contiguous()
        targets = poi_seq[:, 1:].contiguous()
        mask = (targets != 0).float()
        ce = F.cross_entropy(logits_shift.view(-1, logits_shift.size(-1)),
                             targets.view(-1), reduction='none')
        ce = ce.view(targets.size())
        s_p = (ce * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        # Social score = cosine distance to friend prototype
        social_proto = None
        has_social = False
        if uid in self.social_protos and bucket in self.social_protos[uid]:
            social_proto = self.social_protos[uid][bucket].to(device)
            has_social = True

        if has_social:
            s_s = 1.0 - F.cosine_similarity(h, social_proto.unsqueeze(0), dim=-1)
        else:
            s_s = torch.zeros(h.size(0), device=device)  # no social signal

        # Normalize both scores to [0, 1] range approximately
        # s_p is CE loss (typically 0-10), s_s is cosine distance (0-2)
        s_p_norm = torch.sigmoid(s_p - 1.0)  # center around CE=1
        s_s_norm = s_s / 2.0  # cosine distance max is 2

        # Compute alpha based on fusion mode
        if fusion_mode == 'adaptive':
            # Per-user adaptive α: users with weak social signal lean on personal
            # Social confidence = f(friend_count, avg_friend_sim, has_social_proto)
            if not has_social or friend_count is None or friend_count == 0:
                alpha = torch.tensor(1.0, device=device)  # pure personal
            else:
                # Social weight increases with friend count (saturates at k)
                # and similarity (normalized by typical sim ~0.1)
                count_factor = min(friend_count / self.top_k, 1.0)
                sim_factor = min((avg_friend_sim or 0.0) / 0.1, 1.0)
                social_confidence = count_factor * sim_factor
                # alpha ranges from 1.0 (pure personal) to 0.5 (equal weight)
                alpha = torch.tensor(1.0 - 0.5 * social_confidence, device=device)
        elif fusion_mode == 'learned':
            if history_len is not None and friend_count is not None and avg_friend_sim is not None:
                features = torch.FloatTensor([[
                    np.log1p(history_len),
                    np.log1p(friend_count),
                    avg_friend_sim
                ]]).to(device)
                alpha = torch.sigmoid(self.reliability_gate(features)).squeeze()
            else:
                alpha = torch.tensor(0.5, device=device)
        else:  # fixed
            alpha = torch.tensor(0.5, device=device)

        score = alpha * s_p_norm + (1 - alpha) * s_s_norm + self.beta * torch.abs(s_p_norm - s_s_norm)

        return score, s_p, s_s


class EarlyFusionAE(nn.Module):
    """Baseline: Early fusion — concatenate trajectory embedding with social prototype,
    then score via a learned projection. Uses the same GRU encoder as RADD."""

    def __init__(self, num_pois, embed_dim=128, hidden_dim=128, top_k=10):
        super().__init__()
        self.encoder = TrajectoryEncoder(num_pois, embed_dim, hidden_dim)
        self.decoder = TrajectoryDecoder(num_pois, hidden_dim, embed_dim)
        self.top_k = top_k
        # Projection from concat(h, social_proto) to anomaly score
        self.score_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.social_protos = {}

    def reconstruction_loss(self, poi_seq, time_bucket, lengths=None):
        h = self.encoder(poi_seq, time_bucket, lengths)
        logits = self.decoder(poi_seq, h)
        logits = logits[:, :-1].contiguous()
        targets = poi_seq[:, 1:].contiguous()
        mask = (targets != 0).float()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                               targets.view(-1), reduction='none')
        loss = loss.view(targets.size())
        loss = (loss * mask).sum() / mask.sum().clamp(min=1)
        return loss, h

    def build_social_prototypes(self, graph, friend_sim, train_data):
        """Same as RADD — build per-user social prototypes."""
        device = next(self.parameters()).device
        self.social_protos = {}
        user_bucket_embeds = defaultdict(lambda: defaultdict(list))

        self.eval()
        with torch.no_grad():
            for uid, trajs in train_data.items():
                for traj in trajs:
                    seq = traj['poi_seq']
                    if len(seq) < 2:
                        continue
                    bucket = traj['time_bucket']
                    poi_tensor = torch.LongTensor([seq]).to(device)
                    bucket_tensor = torch.LongTensor([bucket]).to(device)
                    h = self.encoder(poi_tensor, bucket_tensor)
                    user_bucket_embeds[uid][bucket].append(h.squeeze(0))

        user_bucket_avg = {}
        for uid in user_bucket_embeds:
            user_bucket_avg[uid] = {}
            for bucket, embeds in user_bucket_embeds[uid].items():
                user_bucket_avg[uid][bucket] = torch.stack(embeds).mean(0)

        for uid in train_data:
            if uid not in graph:
                continue
            sims = friend_sim.get(uid, {})
            friends = [(fid, sims.get(fid, 0.0))
                       for fid in graph.neighbors(uid)
                       if fid in user_bucket_avg]
            if not friends:
                continue
            friends.sort(key=lambda x: x[1], reverse=True)
            top_friends = friends[:self.top_k]
            self.social_protos[uid] = {}
            for bucket in range(6):
                friend_embeds = []
                friend_weights = []
                for fid, sim in top_friends:
                    if bucket in user_bucket_avg.get(fid, {}):
                        friend_embeds.append(user_bucket_avg[fid][bucket])
                        friend_weights.append(max(sim, 0.01))
                if friend_embeds:
                    weights = torch.FloatTensor(friend_weights).to(device)
                    weights = weights / weights.sum()
                    stacked = torch.stack(friend_embeds)
                    self.social_protos[uid][bucket] = (stacked * weights.unsqueeze(1)).sum(0)

    def compute_anomaly_score(self, uid, poi_seq, time_bucket, lengths=None):
        """Score = projection(concat(h, social_proto)). Falls back to reconstruction error."""
        device = next(self.parameters()).device
        h = self.encoder(poi_seq, time_bucket, lengths)
        bucket = time_bucket.item() if isinstance(time_bucket, torch.Tensor) and time_bucket.dim() == 0 else time_bucket[0].item()

        if uid in self.social_protos and bucket in self.social_protos[uid]:
            social_proto = self.social_protos[uid][bucket].to(device).unsqueeze(0)
            concat = torch.cat([h, social_proto], dim=-1)
            score = self.score_proj(concat).squeeze(-1)
        else:
            # Fallback: reconstruction error
            logits = self.decoder(poi_seq, h)
            logits = logits[:, :-1].contiguous()
            targets = poi_seq[:, 1:].contiguous()
            mask = (targets != 0).float()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), reduction='none')
            loss = loss.view(targets.size())
            score = (loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        return score


class GRUAutoencoder(nn.Module):
    """Baseline: GRU Autoencoder (trajectory-only)."""

    def __init__(self, num_pois, embed_dim=128, hidden_dim=128):
        super().__init__()
        self.encoder = TrajectoryEncoder(num_pois, embed_dim, hidden_dim)
        self.decoder = TrajectoryDecoder(num_pois, hidden_dim, embed_dim)

    def reconstruction_loss(self, poi_seq, time_bucket, lengths=None):
        h = self.encoder(poi_seq, time_bucket, lengths)
        logits = self.decoder(poi_seq, h)
        logits = logits[:, :-1].contiguous()
        targets = poi_seq[:, 1:].contiguous()
        mask = (targets != 0).float()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                               targets.view(-1), reduction='none')
        loss = loss.view(targets.size())
        loss = (loss * mask).sum() / mask.sum().clamp(min=1)
        return loss, h

    def compute_anomaly_score(self, poi_seq, time_bucket, lengths=None):
        """Anomaly score = reconstruction error."""
        h = self.encoder(poi_seq, time_bucket, lengths)
        logits = self.decoder(poi_seq, h)
        logits = logits[:, :-1].contiguous()
        targets = poi_seq[:, 1:].contiguous()
        mask = (targets != 0).float()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                               targets.view(-1), reduction='none')
        loss = loss.view(targets.size())
        score = (loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return score


class LSTMAutoencoder(nn.Module):
    """Baseline: LSTM Autoencoder."""

    def __init__(self, num_pois, embed_dim=128, hidden_dim=128):
        super().__init__()
        self.poi_embed = nn.Embedding(num_pois, embed_dim, padding_idx=0)
        self.time_embed = nn.Embedding(6, 16)
        self.encoder_lstm = nn.LSTM(embed_dim + 16, hidden_dim, batch_first=True)
        self.decoder_lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_pois)
        self.hidden_dim = hidden_dim

    def forward(self, poi_seq, time_bucket, lengths=None):
        poi_emb = self.poi_embed(poi_seq)
        time_emb = self.time_embed(time_bucket).unsqueeze(1).expand(-1, poi_seq.size(1), -1)
        x = torch.cat([poi_emb, time_emb], dim=-1)

        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)

        _, (h, c) = self.encoder_lstm(x)
        dec_input = self.poi_embed(poi_seq)
        output, _ = self.decoder_lstm(dec_input, (h, c))
        logits = self.fc(output)
        return logits

    def reconstruction_loss(self, poi_seq, time_bucket, lengths=None):
        logits = self.forward(poi_seq, time_bucket, lengths)
        logits = logits[:, :-1].contiguous()
        targets = poi_seq[:, 1:].contiguous()
        mask = (targets != 0).float()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                               targets.view(-1), reduction='none')
        loss = loss.view(targets.size())
        loss = (loss * mask).sum() / mask.sum().clamp(min=1)
        return loss, None

    def compute_anomaly_score(self, poi_seq, time_bucket, lengths=None):
        logits = self.forward(poi_seq, time_bucket, lengths)
        logits = logits[:, :-1].contiguous()
        targets = poi_seq[:, 1:].contiguous()
        mask = (targets != 0).float()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                               targets.view(-1), reduction='none')
        loss = loss.view(targets.size())
        score = (loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return score
