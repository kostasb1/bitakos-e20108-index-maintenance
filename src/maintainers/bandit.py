"""The bandit policy, a contextual bandit that picks one of five actions for each partition.

`maintain` is one pass. Every partition is a trial, with a six number context, the set of actions
its guards allow, a LinUCB choice, the action, and a reward from the change in the total Quake cost.

The constructor carries options for experiments. The reported configuration is set in
`scripts/final_sweep.build`, which passes the starting index thresholds, a minimum split size of
100, every partition as a candidate, discounting once per pass, and a coverage penalty weight of
2.0. The parameter named `alpha` here is the exploration weight of LinUCB, written beta in the
thesis, and is unrelated to SPLIT_ACCESS_ALPHA. These options are off in the reported runs, the
query grounded reward, the do no harm margin, cooldowns, the recency feature, the potential based
reward and the warm start.
"""

from __future__ import annotations

from dataclasses import replace
from enum import IntEnum

import numpy as np

from src.bandit.context import CONTEXT_DIM, extract_context
from src.bandit.linucb import ARROW_LAMBDAS, LinUCBAgent, WarmLinUCB
from src.bandit.reward import (
    PartitionSnapshot,
    combine_snapshots,
    compute_reward,
    index_cost,
    snapshot,
)
from src.config import (
    BOUNDARY_REASSIGN_TOP_K,
    DEFAULT_NPROBE,
    DRIFT_FRACTION,
    LIRE_MERGE_CANDIDATES,
    MAINTENANCE_CHECK_INTERVAL,
    MAX_PARTITION_SIZE,
    MIN_PARTITION_SIZE,
    SPLIT_ACCESS_ALPHA,
)
from src.index import IVFIndex
from src.maintainers.base import (
    Maintainer,
    collect_empty_partitions,
    drift_limit,
    merge_partitions,
    nearest_partitions_in_order,
    reassign_boundary,
    split_partition_2means,
    top_k_nearest_partitions,
)
from src.metrics import query_scan_cost
from src.partition import Partition
from src.types import MaintenanceReport


class Action(IntEnum):
    """The five actions the bandit can take on a partition."""

    SPLIT = 0
    CENTROID_UPDATE = 1
    MERGE = 2
    COMPACT = 3
    NO_OP = 4


_N_ARMS: int = len(Action)

# the weight of each action on the vectors it touches, in the action cost, divided by the split
# threshold in _action_cost
_ACTION_COST_MULTIPLIER: dict[int, float] = {
    Action.SPLIT: 2.0,
    Action.MERGE: 1.5,
    Action.CENTROID_UPDATE: 1.0,
    Action.COMPACT: 1.0,
    Action.NO_OP: 0.0,
}


class BanditMaintainer(Maintainer):
    """A contextual bandit that learns which action to take on each partition."""

    name = "bandit"

    def __init__(
        self,
        alpha: float = 1.0,
        discount: float = 0.99,
        access_decay: float = 0.9,
        n_candidates: int = 20,
        n_explore: int = 200,
        drift_fraction: float = DRIFT_FRACTION,
        cost_model=None,
        max_partition_size: int = MAX_PARTITION_SIZE,
        min_partition_size: int = MIN_PARTITION_SIZE,
        min_split_size: int = 0,
        frag_penalty_weight: float = 2.0,
        check_interval: int = MAINTENANCE_CHECK_INTERVAL,
        seed: int = 42,
        eval_queries: np.ndarray | None = None,
        eval_k: int = 10,
        eval_nprobe: int = 10,
        recall_penalty_weight: float = 5.0,
        act_margin: float | None = None,
        min_partitions: int | None = None,
        max_partitions: int | None = None,
        coverage_penalty_weight: float = 0.0,
        cooldown_base: int = 0,
        cooldown_min: int = 1,
        cooldown_change_frac: float = 0.5,
        reversal_penalty_weight: float = 0.0,
        use_recency_feature: bool = False,
        drift_threshold: float | None = None,
        recency_window: int = 4,
        track_reversals: bool = False,
        potential_reward: bool = False,
        potential_cost_weight: float = 0.1,
        nprobe: int = DEFAULT_NPROBE,
        merge_candidates: int = LIRE_MERGE_CANDIDATES,
        reward_clip: float = 1.0,
        discount_mode: str = "arm",
        warm_start: bool = False,
    ) -> None:
        """Create the policy.

        The main parameters are `alpha`, the exploration weight, `discount`, the forgetting
        factor of the learner, `access_decay`, the factor the access counts are multiplied by each
        pass, `n_explore`, the number of random decisions at the start, `cost_model`, the scan
        cost curve the reward is priced with, the split and merge thresholds, `min_split_size`,
        and `coverage_penalty_weight`. `n_candidates` None makes every partition a candidate.
        """
        super().__init__(check_interval=check_interval)
        # the nprobe used to turn probe counts into the access fraction when no routed query
        # count exists. separate from eval_nprobe, which belongs to the optional query reward
        self.nprobe = nprobe
        self.use_recency_feature = use_recency_feature
        self.context_dim = CONTEXT_DIM + (1 if use_recency_feature else 0)
        # "pass" discounts every action once per maintenance pass, as D-LinUCB discounts every
        # round, and is what the reported runs use. "arm" with no warm start is the earlier agent
        self.discount_mode = discount_mode
        self.warm_start = warm_start
        self._warmed = False
        if discount_mode == "arm" and not warm_start:
            self.agent = LinUCBAgent(
                n_arms=_N_ARMS, context_dim=self.context_dim, alpha=alpha, discount=discount
            )
        else:
            self.agent = WarmLinUCB(
                n_arms=_N_ARMS, context_dim=self.context_dim, alpha=alpha, discount=discount,
                discount_mode=discount_mode,
                lambdas=ARROW_LAMBDAS if warm_start else (1.0,),
            )
        if warm_start:
            # a warm start replaces the random decisions at the start, and the confidence bonus
            # does the exploring from there
            n_explore = 0
        self.access_decay = access_decay
        self.n_candidates = n_candidates
        self.n_explore = n_explore
        self.drift_fraction = drift_fraction
        self.drift_threshold = drift_threshold
        # the cost model the reward prices actions with, Quake equation 1. without one the reward
        # uses access times size
        self.cost_model = cost_model
        self.max_partition_size = max_partition_size
        self.min_partition_size = min_partition_size
        # no split below this size, so the cost cannot be lowered by fragmenting the index
        self.min_split_size = min_split_size
        # an extra split cost once the partitions are on average smaller than min_split_size
        self.frag_penalty_weight = frag_penalty_weight
        self._frag_overhead: float = 0.0
        self._rng = np.random.default_rng(seed)
        self.action_counts: dict[str, int] = {a.name: 0 for a in Action}
        # the query grounded reward, off unless eval_queries is given
        self.eval_queries = eval_queries
        self.eval_k = eval_k
        self.eval_nprobe = eval_nprobe
        self.recall_penalty_weight = recall_penalty_weight
        # the do no harm margin, off unless given, acts only if the predicted reward exceeds it
        self.act_margin = act_margin
        self.min_partitions = min_partitions
        self.max_partitions = max_partitions
        # the penalty on splits that discourages fragmenting an index that is already fine
        self.coverage_penalty_weight = coverage_penalty_weight
        # cooldowns, off when cooldown_base is 0, block reversing a recent split or merge
        self.cooldown_base = cooldown_base
        self.cooldown_min = cooldown_min
        self.cooldown_change_frac = cooldown_change_frac
        # a softer alternative to the cooldown, which penalises a reversal instead of blocking it
        self.reversal_penalty_weight = reversal_penalty_weight
        self.recency_window = recency_window
        self._track_actions = bool(cooldown_base > 0 or reversal_penalty_weight > 0
                                   or use_recency_feature or track_reversals)
        self._now_ops = 0
        self._mean_access = 0.0
        self._action_log: dict[int, tuple[str, int, int]] = {}
        self._total_splits = 0
        self._total_merges = 0
        self._reversals = 0
        # the potential based reward, off unless asked for
        self.potential_reward = potential_reward
        self.potential_cost_weight = potential_cost_weight
        self._phi_scale = 1.0
        # the combined state of both partitions before a merge, set by _do_merge and read by
        # maintain
        self._merge_before: PartitionSnapshot | None = None
        # how many nearest partitions a merge looks through for a partner that fits
        self.merge_candidates = merge_candidates
        self.reward_clip = reward_clip
        # actions that turned out to change nothing are neither rewarded nor learned from, and
        # are counted here instead
        self.refused_counts: dict[str, int] = {a.name: 0 for a in Action}
        self._last_refused = False

    def should_maintain(self, index: IVFIndex, step: int) -> bool:
        """Return True if a partition is over the split threshold, under the merge threshold or
        empty, or if the mean centroid drift is over the drift limit.

        Reading the mean drift recomputes the mean of every changed partition, which is charged as
        work, directly to the running total since no report exists yet.
        """
        self._cumulative_work += sum(p.size() for p in index.partitions.values()
                                     if p._cached_true_mean is None or p._dirty)
        stats = index.stats()
        if stats.max_partition_size > self.max_partition_size:
            return True
        if stats.mean_drift > drift_limit(index, self.drift_fraction, self.drift_threshold):
            return True
        return any(p.size() < self.min_partition_size for p in index.partitions.values())

    def maintain(self, index: IVFIndex) -> MaintenanceReport:
        """Run one pass, choosing, applying and learning from one action per candidate partition."""
        report = MaintenanceReport(triggered=True)
        # first, so no decision is spent on an empty partition and the context is not computed
        # against dead centroids
        report.num_collected += len(collect_empty_partitions(index))
        for p in index.partitions.values():
            p.decay_access(self.access_decay)
        # the query count decays together with the access counts it divides
        index.queries_routed *= self.access_decay
        stats = index.stats()
        self._now_ops = index._total_inserts + index._total_deletes
        if self.potential_reward:
            self._phi_scale = max(1e-9, self._badness(index) / max(1, len(index.partitions)))
        if self._track_actions:
            _accs = [p.access_count for p in index.partitions.values()]
            self._mean_access = float(np.mean(_accs)) if _accs else 0.0
        mean_size = stats.live_vectors / max(1, stats.num_partitions)
        self._frag_overhead = (
            max(0.0, (self.min_split_size / max(1.0, mean_size)) - 1.0)
            if self.min_split_size > 0 else 0.0
        )
        if isinstance(self.agent, WarmLinUCB):
            self.agent.end_of_pass()
        if self.warm_start and not self._warmed:
            self._warm_start(index, stats, report)
            self._warmed = True
        candidates = self._select_candidates(index)

        grounded = self.eval_queries is not None

        for pid in candidates:
            if pid not in index.partitions:
                continue

            partition = index.partitions[pid]
            # earlier actions in this pass changed the index, so the averages the context divides
            # by are recomputed for each candidate
            _live = sum(p.size() for p in index.partitions.values())
            _n = len(index.partitions)
            cur_stats = replace(
                stats,
                num_partitions=_n,
                live_vectors=_live,
                mean_partition_size=(_live / _n) if _n else 0.0,
            )
            # the total access count, which an earlier split in this pass may have changed
            _total_access = sum(p.access_count for p in index.partitions.values())
            _drift_scale = drift_limit(index, self.drift_fraction, self.drift_threshold)
            # reading the drift of a changed partition recomputes its mean, one read of each of
            # its vectors, which is charged
            if partition._cached_true_mean is None or partition._dirty:
                report.vectors_processed += partition.size()
            context = extract_context(
                partition, cur_stats, size_cap=self.max_partition_size,
                total_access=_total_access, nprobe=self.nprobe, drift_scale=_drift_scale,
                queries=index.query_count(self.nprobe),
            )
            if self.use_recency_feature:
                context = np.append(context, self._signed_recency(pid, partition))

            # finding the allowed actions is charged too, see _available_arms
            available = self._available_arms(pid, partition, index, meter=report)
            if self.agent.total_updates < self.n_explore:
                arm = int(available[int(self._rng.integers(0, len(available)))])
            else:
                arm = self.agent.select_arm(context, available)
                if self.act_margin is not None and arm != int(Action.NO_OP):
                    predicted = float(self.agent.theta(arm) @ context)
                    if predicted <= self.act_margin:
                        arm = int(Action.NO_OP)

            action = Action(arm)
            before = snapshot(partition, self.cost_model)
            rev_penalty = self._reversal_penalty(pid, partition, action)
            self._merge_before = None

            if action == Action.NO_OP:
                # nothing changes, so the reward is zero in every form
                after_partitions = self._execute(action, pid, index, report)
                reward = 0.0
            elif self.potential_reward:
                b_before = self._badness(index)
                after_partitions = self._execute(action, pid, index, report)
                b_after = self._badness(index)
                cost = self._action_cost(action, self._merge_before or before)
                reward = ((b_before - b_after) / self._phi_scale
                          - self.potential_cost_weight * cost)
            elif grounded and action == Action.NO_OP:
                after_partitions = self._execute(action, pid, index, report)
                reward = 0.0
            elif grounded:
                cost_b, dist_b = self._eval_cost_quality(index)
                after_partitions = self._execute(action, pid, index, report)
                cost_a, dist_a = self._eval_cost_quality(index)
                cost = self._action_cost(action, self._merge_before or before)
                cost_reduction = (cost_b - cost_a) / max(cost_b, 1e-9)
                quality_penalty = self.recall_penalty_weight * max(
                    0.0, (dist_a - dist_b) / max(dist_b, 1e-9))
                reward = cost_reduction - quality_penalty - 0.1 * cost
            else:
                # the reported reward. the cost term is the change in the total Quake cost C
                # across the action, so the neighbours a split takes vectors from and the centroid
                # it adds are both priced, see reward.index_cost
                c_before = index_cost(index, self.cost_model, self.nprobe)
                scale = self._cost_scale(index, c_before)
                after_partitions = self._execute(action, pid, index, report)
                cost = self._action_cost(action, self._merge_before or before)
                reward = compute_reward(
                    self._merge_before or before, after_partitions,
                    cost_model=self.cost_model,
                    action_cost=cost,
                    drift_scale=_drift_scale,
                    delta_index_cost=index_cost(index, self.cost_model, self.nprobe) - c_before,
                    cost_scale=scale,
                    clip=self.reward_clip,
                    # only an action that keeps the same members can be credited for closing drift,
                    # see compute_reward
                    drift_credit=action in (Action.CENTROID_UPDATE, Action.COMPACT),
                )

            if self._last_refused:
                # a guard inside the action refused it, so nothing happened and there is nothing
                # to learn
                self.refused_counts[action.name] += 1
                continue

            if (self.coverage_penalty_weight > 0.0 and action == Action.SPLIT
                    and len(after_partitions) > 1):
                reward -= self.coverage_penalty_weight * self._frag_overhead
            reward -= rev_penalty
            # the penalties come after compute_reward clipped the reward, so it is clipped again
            if self.reward_clip is not None:
                reward = max(-self.reward_clip, min(self.reward_clip, reward))

            self.agent.update(arm, context, reward)
            self.action_counts[action.name] += 1

        self._total_splits += report.num_splits
        self._total_merges += report.num_merges
        return report

    def _select_candidates(self, index: IVFIndex) -> list[int]:
        """Return the partitions to decide on in this pass.

        With `n_candidates` None, which the reported runs use, that is every partition, as Quake
        considers every partition each pass. Otherwise it is the top ones by (1 + access) times
        size, plus a few of the smallest partitions under the merge threshold, so the merge action
        sees the partitions it is meant for.
        """
        pids = index.partition_ids()
        if not pids:
            return []
        urgency = np.array(
            [
                (1.0 + index.partitions[pid].access_count) * index.partitions[pid].size()
                for pid in pids
            ],
            dtype=np.float64,
        )
        n = len(pids) if self.n_candidates is None else min(self.n_candidates, len(pids))
        if n < len(pids):
            top_idx = np.argpartition(urgency, -n)[-n:]
        else:
            top_idx = np.arange(len(pids), dtype=np.intp)
        candidates = [pids[int(i)] for i in top_idx]

        if self.min_partition_size > 0 and self.n_candidates is not None:
            undersized = sorted(
                (pid for pid in pids
                 if 0 < index.partitions[pid].size() < self.min_partition_size),
                key=lambda pid: index.partitions[pid].size(),
            )
            extra = max(1, self.n_candidates // 4)
            seen = set(candidates)
            candidates.extend(pid for pid in undersized[:extra] if pid not in seen)
        return candidates

    def _eval_cost_quality(self, index: IVFIndex) -> tuple[float, float]:
        """Return the mean vectors scanned and the mean neighbour distance on the evaluation queries.

        Used only by the optional query grounded reward. Lower is better for both.
        """
        results = index.search_batch(
            self.eval_queries, self.eval_k, self.eval_nprobe, record_access=False)
        dists = [d for res in results for (_, d) in res]
        quality = float(np.mean(dists)) if dists else 0.0
        cost = query_scan_cost(index, self.eval_queries, self.eval_nprobe)
        return cost, quality

    def _warm_start(self, index: IVFIndex, stats, report) -> None:
        """Give the learner estimated rewards for every partition and action before it acts.

        This is the warm start of ARROW-CB, built from the running index and the cost model. For
        every partition and every action it could take, the reward is estimated without acting,
        on the same scale as the real reward, see `_whatif`. Not used by the reported runs.
        """
        n = max(1, len(index.partitions))
        live = sum(p.size() for p in index.partitions.values())
        cur = replace(stats, num_partitions=n, live_vectors=live, mean_partition_size=live / n)
        total_access = sum(p.access_count for p in index.partitions.values())
        queries = index.query_count(self.nprobe)
        drift_scale = drift_limit(index, self.drift_fraction, self.drift_threshold)
        c_before = index_cost(index, self.cost_model, self.nprobe)
        scale = self._cost_scale(index, c_before)
        fracs = {pid: (min(1.0, p.access_count / queries) if queries > 0 else 0.0)
                 for pid, p in index.partitions.items()}
        avg_access = sum(fracs.values()) / n
        avg_size = live // n
        xs: dict[int, list] = {int(a): [] for a in Action}
        rs: dict[int, list] = {int(a): [] for a in Action}
        for pid, p in index.partitions.items():
            if p.size() == 0:
                continue
            x = extract_context(p, cur, size_cap=self.max_partition_size,
                                total_access=total_access, nprobe=self.nprobe,
                                drift_scale=drift_scale, queries=queries)
            if self.use_recency_feature:
                x = np.append(x, 0.0)
            for arm in self._available_arms(pid, p, index, meter=report):
                r = self._whatif(Action(arm), pid, p, index, fracs[pid], scale,
                                 avg_access, avg_size, drift_scale, meter=report)
                xs[arm].append(x)
                rs[arm].append(r)
        for arm in xs:
            if xs[arm]:
                self.agent.add_warm(arm, np.array(xs[arm]), np.array(rs[arm]))

    def _whatif(self, action: Action, pid: int, p: Partition, index: IVFIndex, access: float,
                scale: float, avg_access: float, avg_size: int, drift_scale: float,
                meter=None) -> float:
        """Estimate the reward an action would earn, without taking it.

        Refresh, compaction and no action are estimated exactly. A split is priced with Quake
        equation 6 and a merge with Quake's uniform delete estimate, which are approximations.
        """
        if action == Action.NO_OP:
            return 0.0
        lam = self.cost_model.scan_latency if self.cost_model else float
        lam_c = self.cost_model.centroid_latency if self.cost_model else float
        k = len(index.partitions)
        tomb = len(p.tombstones) / max(p.total_size(), 1)
        snap = snapshot(p, self.cost_model)
        if action == Action.CENTROID_UPDATE:
            r = -0.1 * self._action_cost(action, snap) + 0.1 * p.centroid_drift() / max(
                drift_scale, 1e-6)
        elif action == Action.COMPACT:
            r = -0.1 * self._action_cost(action, snap) + 0.1 * tomb
        elif action == Action.SPLIT:
            s = p.size()
            dc = (lam_c(k + 1) - lam_c(k) - access * lam(s)
                  + 2.0 * SPLIT_ACCESS_ALPHA * access * lam(s // 2))
            r = -dc / max(scale, 1e-9) - 0.1 * self._action_cost(action, snap) + 0.1 * tomb
            r -= self.coverage_penalty_weight * self._frag_overhead
        else:
            nbr_pid = self._merge_target(pid, p, index, meter=meter)
            if nbr_pid is None:
                return 0.0
            both = combine_snapshots([snap, snapshot(index.partitions[nbr_pid], self.cost_model)])
            s = p.size()
            if k <= 1:
                dc = 0.0
            else:
                overhead = lam_c(k - 1) - lam_c(k)
                cost_old = (k - 1) * avg_access * lam(avg_size) + access * lam(s)
                m_access = avg_access + access / (k - 1)
                if s < k:
                    cost_new = (s * m_access * lam(avg_size + 1)
                                + (k - s - 1) * m_access * lam(avg_size))
                else:
                    cost_new = (k - 1) * m_access * lam(int(np.ceil(avg_size + s / (k - 1))))
                dc = overhead + cost_new - cost_old
            r = (-dc / max(scale, 1e-9) - 0.1 * self._action_cost(action, both)
                 + 0.1 * both.tombstone_count / max(both.total_size, 1))
        return max(-self.reward_clip, min(self.reward_clip, r))

    def _cost_scale(self, index: IVFIndex, c_before: float) -> float:
        """Return the unit the cost change is measured in, the mean scan cost per partition.

        That is the scan part of C divided by K. Dividing by the acted partition's own cost
        instead would divide by nearly zero for a partition queries rarely reach. Before any query
        has been counted the scan part is zero, so nprobe partitions of the mean size are used.
        """
        lam = self.cost_model.scan_latency if self.cost_model else float
        lam_c = self.cost_model.centroid_latency if self.cost_model else float
        k = max(1, len(index.partitions))
        scan = c_before - lam_c(len(index.partitions))
        if scan <= 0.0:
            mean = sum(p.size() for p in index.partitions.values()) / k
            scan = self.nprobe * lam(int(mean))
        return scan / k

    def _merge_target(self, pid: int, partition: Partition, index: IVFIndex,
                      meter=None) -> int | None:
        """Return the merge partner, the nearest partition whose combined size fits, or None.

        As in SPFresh section 3.2, the nearest partitions are tried in order. With cooldowns on, a
        partition that was just split off is skipped, since merging with it would undo that split.
        """
        for nbr_pid in nearest_partitions_in_order(
                index, partition.centroid, self.merge_candidates, exclude={pid},
                meter=meter):
            nbr = index.partitions[nbr_pid]
            if partition.size() + nbr.size() > self.max_partition_size:
                continue
            if self._in_cooldown(nbr_pid, nbr, "split"):
                continue
            return nbr_pid
        return None

    def _available_arms(self, pid: int, partition: Partition, index: IVFIndex,
                        meter=None) -> list[int]:
        """Return the actions whose guards allow them on this partition.

        No action is always allowed, a refresh needs live vectors, compaction needs tombstones,
        a split needs at least `min_split_size` vectors, and a merge needs a partner that fits.
        Finding a merge partner compares the partition with every centroid, which is charged as
        work, since a bandit that considers every partition would otherwise make about K squared
        distance computations per pass for free.
        """
        arms = [int(Action.NO_OP)]
        if partition.size() > 0:
            arms.append(int(Action.CENTROID_UPDATE))
        if partition.tombstones:
            arms.append(int(Action.COMPACT))
        if (partition.size() >= max(2, self.min_split_size)
                and (self.max_partitions is None or len(index.partitions) < self.max_partitions)
                and not self._in_cooldown(pid, partition, "merge")):
            arms.append(int(Action.SPLIT))
        if ((self.min_partitions is None or len(index.partitions) > self.min_partitions)
                and not self._in_cooldown(pid, partition, "split")
                and self._merge_target(pid, partition, index, meter=meter) is not None):
            arms.append(int(Action.MERGE))
        return sorted(arms)

    def _action_cost(self, action: Action, before) -> float:
        """Return the cost of an action, its weight times the vectors it touches over the split threshold.

        A split also pays the fragmentation overhead.
        """
        vectors = before.total_size if action == Action.COMPACT else before.size
        cost = _ACTION_COST_MULTIPLIER[action] * vectors / max(self.max_partition_size, 1)
        if action == Action.SPLIT:
            cost += self.frag_penalty_weight * self._frag_overhead
        return cost

    def _badness(self, index: IVFIndex) -> float:
        """Return the sum of (1 + access) times size squared, for the optional potential reward.

        It grows faster than linearly with size, so a split lowers it and undoing one raises it.
        """
        return float(sum((1.0 + p.access_count) * p.size() ** 2
                         for p in index.partitions.values()))

    def _op_horizon(self, base_intervals: float, partition: Partition) -> float:
        """Return how long, in updates, a cooldown lasts for this partition.

        A partition queried more often than average gets a shorter cooldown, so the policy can
        revisit it sooner.
        """
        rel_hot = partition.access_count / (self._mean_access + 1e-9)
        scale = self.check_interval
        return max(self.cooldown_min * scale, base_intervals * scale / (1.0 + rel_hot))

    def _in_cooldown(self, pid: int, partition: Partition, reverse_of: str) -> bool:
        """Return True if acting now would undo a recent `reverse_of` action on this partition.

        Always False when cooldowns are off, as in the reported runs. A partition whose size has
        changed a lot since that action is no longer held back.
        """
        if self.cooldown_base <= 0:
            return False
        rec = self._action_log.get(pid)
        if rec is None or rec[0] != reverse_of:
            return False
        _, birth_ops, birth_size = rec
        if (self._now_ops - birth_ops) >= self._op_horizon(self.cooldown_base, partition):
            return False
        changed = abs(partition.size() - birth_size) / max(1, birth_size)
        return changed <= self.cooldown_change_frac

    def _signed_recency(self, pid: int, partition: Partition) -> float:
        """Return the optional recency feature, positive after a recent split, negative after a merge.

        It fades from 1 to 0 over the recency window.
        """
        rec = self._action_log.get(pid)
        if rec is None:
            return 0.0
        op, birth_ops, _ = rec
        horizon = self._op_horizon(self.recency_window, partition)
        r = max(0.0, 1.0 - (self._now_ops - birth_ops) / horizon)
        return r * (1.0 if op == "split" else -1.0 if op == "merge" else 0.0)

    def _reversal_penalty(self, pid: int, partition: Partition, action: Action) -> float:
        """Return the optional penalty for undoing a recent split or merge, zero when it is off."""
        if self.reversal_penalty_weight <= 0.0:
            return 0.0
        reverse_of = ("split" if action == Action.MERGE
                      else "merge" if action == Action.SPLIT else None)
        rec = self._action_log.get(pid)
        if reverse_of is None or rec is None or rec[0] != reverse_of:
            return 0.0
        _, birth_ops, birth_size = rec
        if abs(partition.size() - birth_size) / max(1, birth_size) > self.cooldown_change_frac:
            return 0.0
        horizon = self._op_horizon(self.recency_window, partition)
        r = max(0.0, 1.0 - (self._now_ops - birth_ops) / horizon)
        return self.reversal_penalty_weight * r

    def _execute(
        self, action: Action, pid: int, index: IVFIndex, report: MaintenanceReport
    ) -> list[Partition]:
        """Apply an action to a partition and return the partitions that result.

        Sets `_last_refused` when the action changed nothing, judged by whether the report moved,
        so a refusal inside any of the helpers is detected the same way.
        """
        self._last_refused = False
        if pid not in index.partitions:
            self._last_refused = action != Action.NO_OP
            return []

        partition = index.partitions[pid]
        mark = (report.num_splits, report.num_merges, report.num_centroids_recomputed,
                report.vectors_processed)

        if action == Action.SPLIT:
            out = self._do_split(pid, partition, index, report)
        elif action == Action.CENTROID_UPDATE:
            out = self._do_centroid_update(pid, index, report)
        elif action == Action.MERGE:
            out = self._do_merge(pid, partition, index, report)
        elif action == Action.COMPACT:
            out = self._do_compact(pid, partition, index, report)
        else:
            return [partition]
        self._last_refused = mark == (report.num_splits, report.num_merges,
                                      report.num_centroids_recomputed, report.vectors_processed)
        return out

    def _do_split(
        self, pid: int, partition: Partition, index: IVFIndex, report: MaintenanceReport
    ) -> list[Partition]:
        """Split a partition with 2-means and reassign the vectors near the new boundary.

        The reassignment covers the two children and the 25 nearest partitions of the parent.
        Returns the children, or the partition itself if the split was refused.
        """
        if partition.size() < max(2, self.min_split_size):
            return [partition]
        if self.max_partitions is not None and len(index.partitions) >= self.max_partitions:
            return [partition]
        if self._in_cooldown(pid, partition, "merge"):
            return [partition]

        report.vectors_processed += partition.size()
        neighbors = top_k_nearest_partitions(
            index, partition.centroid, BOUNDARY_REASSIGN_TOP_K, exclude={pid}, meter=report
        )
        children = split_partition_2means(
            partition, random_state=int(self._rng.integers(0, 2**31)), meter=report
        )
        if len(children) < 2:
            return [partition]

        pids_before = set(index.partition_ids())
        index.apply_split(pid, children)
        new_pids = list(set(index.partition_ids()) - pids_before)
        for _np in new_pids:
            if self._track_actions and _np in index.partitions:
                self._action_log[_np] = ("split", self._now_ops, index.partitions[_np].size())

        moves = reassign_boundary(index, new_pids, neighbors, meter=report)
        report.num_reassigned += moves
        report.vectors_processed += moves
        report.num_splits += 1
        _rec = self._action_log.get(pid)
        if (_rec is not None and _rec[0] == "merge"
                and (self._now_ops - _rec[1]) < self._op_horizon(self.recency_window, partition)):
            self._reversals += 1
        report.partitions_touched.extend(new_pids)
        return [index.partitions[p] for p in new_pids if p in index.partitions]

    def _do_centroid_update(
        self, pid: int, index: IVFIndex, report: MaintenanceReport
    ) -> list[Partition]:
        """Move a partition's centroid to the mean of its live vectors."""
        partition = index.partitions[pid]
        if partition.size() == 0:
            return [partition]
        report.vectors_processed += partition.size()
        index.recompute_centroid(pid)
        report.num_centroids_recomputed += 1
        report.partitions_touched.append(pid)
        return [index.partitions[pid]]

    def _do_merge(
        self, pid: int, partition: Partition, index: IVFIndex, report: MaintenanceReport
    ) -> list[Partition]:
        """Join a partition with its merge partner into one centred on the mean of both.

        Unlike LIRE, where the surviving partition keeps its centroid. Both partitions are
        charged, as in every policy that merges. Returns the merged partition.
        """
        if self.min_partitions is not None and len(index.partitions) <= self.min_partitions:
            return [partition]
        if self._in_cooldown(pid, partition, "split"):
            return [partition]
        nbr_pid = self._merge_target(pid, partition, index, meter=report)
        if nbr_pid is None:
            return [partition]
        nbr = index.partitions[nbr_pid]

        report.vectors_processed += partition.size() + nbr.size()
        # the combined state of both partitions before the merge, see combine_snapshots
        self._merge_before = combine_snapshots(
            [snapshot(partition, self.cost_model), snapshot(nbr, self.cost_model)])
        merged = merge_partitions([partition, nbr])
        pids_before = set(index.partition_ids())
        index.apply_merge([pid, nbr_pid], merged)
        new_pids = list(set(index.partition_ids()) - pids_before)

        report.num_merges += 1
        for src_pid, src_part in ((pid, partition), (nbr_pid, nbr)):
            _rec = self._action_log.get(src_pid)
            if (_rec is not None and _rec[0] == "split"
                    and (self._now_ops - _rec[1])
                    < self._op_horizon(self.recency_window, src_part)):
                self._reversals += 1
                break
        if new_pids:
            report.partitions_touched.append(new_pids[0])
            if self._track_actions and new_pids[0] in index.partitions:
                self._action_log[new_pids[0]] = (
                    "merge", self._now_ops, index.partitions[new_pids[0]].size())
            return [index.partitions[new_pids[0]]] if new_pids[0] in index.partitions else []
        return []

    def _do_compact(
        self, pid: int, partition: Partition, index: IVFIndex, report: MaintenanceReport
    ) -> list[Partition]:
        """Compact a partition, charged its stored length. Does nothing if it has no tombstones."""
        if not partition.tombstones:
            return [partition]
        report.vectors_processed += partition.total_size()
        report.compaction_reads += partition.total_size()
        index.compact_partition(pid)
        report.partitions_touched.append(pid)
        return [partition]
