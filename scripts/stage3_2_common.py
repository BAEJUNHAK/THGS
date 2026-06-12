"""
Stage 3.2 — shared infrastructure: pool-rank computation on the real SP pool.

Pool = all SP features at levels [2, 3] from sai_nag.pt (identical universe to
b7_a4_oracle_analysis.py's cache over unique labels; feature rows are indexed
by sp_id). Candidate ranking REPLACES the target SP's own entry, scored with
ClipSimMeasure canon-contrast (same as inference / A4). Raw-cos ranks are
computed alongside.

Fidelity gate helper: reconstruct the visibility-weighted baseline from the
replay pkl and check its pool rank against b7_a4's oracle_rank (±2).
"""

import pickle
import numpy as np
import torch
import torch.nn.functional as F

POOL_LEVELS = [2, 3]


def load_replay(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def visible_views(rec):
    """Views that contributed to the pipeline accumulation (visible & nonzero feat)."""
    return [v for v in rec['views'] if v['visible'] and v['mixed_feat'] is not None]


def baseline_aggregate(views):
    """Pipeline aggregation: sum(normalize(f_i) * portion_i) -> normalize."""
    acc = np.zeros(512, dtype=np.float64)
    for v in views:
        acc += v['mixed_feat'].astype(np.float64) * v['portion']
    n = np.linalg.norm(acc)
    return (acc / n).astype(np.float32) if n > 0 else acc.astype(np.float32)


class ScenePool:
    """Per-(scene) pool; per-prompt scores cached."""

    def __init__(self, model_path, vlm):
        from nag_data import SemanticNAG
        nag = torch.load(f"{model_path}/sai_nag.pt")
        self.snag = SemanticNAG(nag['nag'], nag['nag_feat'])
        self.vlm = vlm
        # pool index: list of (lvl, sp_id) aligned with concatenated features
        self.entries = []
        feats = []
        for lvl in POOL_LEVELS:
            f = self.snag.feat[lvl - 1]
            feats.append(f.cuda().float())
            self.entries += [(lvl, i) for i in range(f.shape[0])]
        self.pool_feat = torch.cat(feats, dim=0)          # (P, 512)
        self.index = {e: i for i, e in enumerate(self.entries)}
        self._cache = {}

    def prompt_scores(self, prompt):
        """(canon_scores, raw_scores, text_feat) for the whole pool."""
        if prompt not in self._cache:
            self.vlm.encode_text(prompt)
            canon = self.vlm.compute_similarity(self.pool_feat).cpu().numpy()
            text = self.vlm.text_feature[0].cpu().numpy()           # plain prompt emb
            raw = (self.pool_feat.cpu().numpy() @ text)
            self._cache[prompt] = (canon, raw, text)
        return self._cache[prompt]

    def cand_scores(self, prompt, cand_feats):
        """canon + raw scores for candidate feature batch (N, 512) np.float32."""
        canon_pool, raw_pool, text = self.prompt_scores(prompt)
        t = torch.from_numpy(np.ascontiguousarray(cand_feats)).cuda().float()
        # encode_text may have been called for another prompt since; re-encode
        self.vlm.encode_text(prompt)
        canon = self.vlm.compute_similarity(t).cpu().numpy()
        raw = cand_feats @ text
        return canon, raw

    def rank(self, prompt, lvl, sp_id, cand_canon, cand_raw):
        """Rank of candidate when it REPLACES (lvl, sp_id)'s pool entry."""
        canon_pool, raw_pool, _ = self.prompt_scores(prompt)
        i = self.index[(lvl, sp_id)]
        mask = np.ones(len(canon_pool), dtype=bool)
        mask[i] = False
        r_canon = int((canon_pool[mask] > cand_canon).sum()) + 1
        r_raw = int((raw_pool[mask] > cand_raw).sum()) + 1
        return r_canon, r_raw

    def ranks_batch(self, prompt, lvl, sp_id, cand_feats):
        canon, raw = self.cand_scores(prompt, cand_feats)
        out = [self.rank(prompt, lvl, sp_id, c, r) for c, r in zip(canon, raw)]
        return np.array([o[0] for o in out]), np.array([o[1] for o in out]), canon, raw


def make_vlm():
    from utils.vlm_utils import ClipSimMeasure
    vlm = ClipSimMeasure()
    vlm.load_model()
    return vlm
