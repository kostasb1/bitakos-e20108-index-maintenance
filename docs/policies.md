# Maintenance policies

This document describes every policy compared in the thesis: when it acts, what one pass does, its parameters with their sources, and where it differs from the published method. The structures and shared operations it relies on are described in [architecture.md](architecture.md).

## Contents

1. [Overview](#overview)
2. [No maintenance](#no-maintenance)
3. [Global rebuild](#global-rebuild)
4. [LIRE, from SPFresh](#lire-from-spfresh)
5. [DeDrift Lazy, Split and Hybrid](#dedrift-lazy-split-and-hybrid)
6. [Quake](#quake)
7. [The bandit policy](#the-bandit-policy)
8. [Configurations outside the main comparison](#configurations-outside-the-main-comparison)
9. [Differences from the sources, summarised](#differences-from-the-sources-summarised)

## Overview

| Name in code | Thesis name | Class | Trigger | Reads query access |
|---|---|---|---|---|
| `no_op` | No maintenance | `NoOpMaintainer` | never | no |
| `global_rebuild` | Global rebuild | `GlobalRebuildMaintainer` | 2.5% of live vectors changed | no |
| `lire_lite` | LIRE | `LireLiteMaintainer` | a partition out of the size limits | no |
| `dedrift_lazy` | DeDrift Lazy | `DeDriftMaintainer` | 2.5% changed | no |
| `dedrift_split` | DeDrift Split | `DeDriftMaintainer` | 2.5% changed | no |
| `dedrift_hybrid` | DeDrift Hybrid | `DeDriftMaintainer` | 2.5% changed | no |
| `cost_driven_quake` | Quake | `CostDrivenQuakeMaintainer` | any access recorded | yes |
| `cost_driven_quake_tau50` | Quake, τ = 50 ns | `CostDrivenQuakeMaintainer` | any access recorded | yes |
| `bandit` | Bandit policy | `BanditMaintainer` | size out of limits or drift too high | yes |

Every policy is offered a pass once every 1,000 updates (`MAINTENANCE_CHECK_INTERVAL`). Quake runs maintenance after every operation, so this is a deliberate coarsening, applied to every policy equally. The size limits of the reported runs are those the starting index was grown with: split above 400, merge at 50 or below.

`scripts/final_sweep.py`, function `build`, is the single place where each name becomes a configured policy.

## No maintenance

`src/maintainers/no_op.py`. `should_maintain` always returns false, so the index changes only through inserts and deletes. Every other policy is reported relative to it on the same seed.

## Global rebuild

`src/maintainers/global_rebuild.py`. The rebuild baseline of Ada-IVF, section 5.1: "we trigger rebuilding after 2.5 percent of the total number of vectors in the workload have been modified".

**Trigger.** Inserts plus deletes since the last rebuild reach 2.5% of the current live count (`REBUILD_FRACTION_THRESHOLD`). The count starts when the stream takes over, so the growth of the starting index does not count. With 50,000 to 60,000 live vectors the threshold is 1,250 to 1,500 updates, and since it is checked every 1,000 updates a rebuild happens every 2,000 updates, 115 times per run.

**One pass.**

1. Collect every live vector and its id.
2. Clear the index and call `IVFIndex.build` with exact k-means at the current partition count (25 iterations, one initialisation, the run's seed).
3. Reset the update counter.

**Work.** One read of every live vector, plus every distance the k-means made, iterations plus one times vectors times centroids, counted as bulk k-means. This is by far the most expensive policy, about 7 billion scan distances per SIFT run against about 0.1 billion for the others.

**Parameter.** `n_partitions` fixes the partition count of every rebuild. The main runs leave it unset. The control run in `scripts/control_rebuild_at_k.py` sets it to the bandit's final partition count on each seed.

**Difference from the source.** Ada-IVF rebuilds to its own target partition size. With 50,000 to 60,000 live vectors the two choices give almost the same K.

## LIRE, from SPFresh

`src/maintainers/lire_lite.py`. The update protocol of SPFresh (SOSP 2023, sections 3.2 and 3.3), following the paper and the authors' C++ implementation.

| Parameter | Value | Source |
|---|---|---|
| split limit | 400 stored vectors | the starting index thresholds, the 8 to 1 ratio of SPFresh |
| merge limit | 50 live vectors | as above |
| merge candidates | 64 | `InternalResultNum` of the SPFresh SIFT1M configuration |
| reassignment radius | 25 partitions | Ada-IVF section 5.1, "a reindexing radius of 25" |

**Trigger.** Any partition whose stored length, tombstones included, exceeds 400, or whose live size is between 1 and 50, or which is empty. The split check reads the stored length because SPFresh checks the length of the posting, which still holds deleted vectors.

**One pass.**

1. **Collect empty partitions** first, so an empty partition is never a split neighbour or merge target. Counted as `num_collected`, no work.
2. **Long partitions**, each with stored length over 400:
   1. Compact it, removing tombstones. If it now has fewer than 400 live vectors, stop. SPFresh section 4.2.1 also collects deleted vectors before deciding to split.
   2. Split with scikit-learn 2-means, 3 starts.
   3. Find the 25 nearest partitions to the old centroid, then apply the split.
   4. Reassign with the two LIRE conditions. A vector of a new child is checked only if the old centroid was at least as close as every new one. A vector of a neighbour is checked only if some new centroid is at least as close as the old one was. A checked vector moves to the nearest centroid of the local set if it is strictly closer than its own.
3. **Short partitions**, each with 1 to 50 live vectors:
   1. Check the size again, since it may have grown earlier in the pass.
   2. Walk its 64 nearest partitions in order and take the first where this partition's live size plus the neighbour's stored size is below 400.
   3. Compact both, then delete the shorter one and append its vectors to the other, which keeps its own centroid (`merge_into_survivor`).
   4. Reassign a moved vector only if the surviving centroid is farther from it than its old centroid was, following the authors' implementation.

**Work.** The split partition is read once, k-means distances are counted, every reassignment check counts its distances and every move counts one vector. A merge charges both partitions. Compaction reads are counted separately and left out of the reported work.

**Role beyond the comparison.** The same class grows the starting index of every policy, see [architecture.md](architecture.md#the-starting-index).

## DeDrift Lazy, Split and Hybrid

`src/maintainers/dedrift.py`. The three strategies of DeDrift (Baranchuk et al., ICCV 2023, section 5.1).

**Trigger.** The same as the global rebuild, 2.5% of live vectors changed since the last step, so the three differ from the rebuild only in what they do. The paper runs them on a schedule of one to six months of content.

**Lazy.** For every non empty partition, move the centroid to the mean of its live vectors. No vector is reassigned. Work is one read of every live vector, about 17 million scan distances per run.

**Split.**

1. Count every partition, empty ones included. mu is the median partition size of the whole index, "the median cluster size of the whole IVF".
2. k = max(1, round(0.002 × K)), the largest clusters. DeDrift uses k = 8 at K = 4096 and k = 64 at K = 16384, so 0.20 to 0.39 percent of K. At K near 178 this rounds to 1.
3. B1 is the k largest partitions. k2 = ceil(|B1| / mu), at least k + 1.
4. B2 is the k2 − k smallest partitions, empty ones included, since an empty partition is a free slot.
5. Run scikit-learn k-means with k2 centres and 3 starts on the vectors of B1 and B2.
6. Replace the involved partitions with the k2 new ones, through `apply_merge` then `apply_split`, so K stays the same.

If more than half the index is empty, mu is zero and the step is skipped, a case DeDrift does not cover.

**Hybrid.** Lazy, then Split.

**What Split does here, measured.** Each pass takes one cluster of about 850 vectors as B1 and the two smallest, 126 to 147 vectors, as B2, wherever they lie. After a pass the share of B2 vectors whose own centroid is their nearest falls from about 74% to about 52%, while the mean squared distance of the whole index hardly moves. The damage is to routing rather than to the k-means objective, and on GIST it leaves Split 22% above no maintenance. DeDrift was designed for indexes of 4,096 partitions and more, where B2 is a much smaller share of the index.

## Quake

`src/maintainers/cost_driven_quake.py`. The maintenance loop of Quake (OSDI 2025, section 4.2), following the paper and the authors' C++ implementation (`maintenance_policies.cpp`, `maintenance_cost_estimator.cpp`, `partition_manager.cpp`).

| Parameter | Value | Source |
|---|---|---|
| τ, split and delete | 250 ns | Quake page 15, "We set tau = 250ns" |
| α | 0.86 | measured here, Quake publishes 0.9, see below |
| minimum partition size | 32 | `common.h` of the released code |
| forced delete threshold | 30% shrinkage | `common.h` |
| refinement radius | 50 partitions | Quake page 10 |
| refinement iterations | 1 | Quake page 10 |
| split k-means | Faiss, 5 iterations, at most 256 points per centre | `common.h` and Faiss defaults |
| access window | one maintenance interval | Quake page 15 |

### The cost model

The total cost the policy minimises is

```
C = Σ_i A_i · λ(s_i) + λ_c(K)
```

where A_i is the share of routed queries in the window that scanned partition i, λ(s) is the cost of scanning a partition of s vectors and λ_c(K) the cost of comparing a query with K centroids. Both come from the Faiss price curve, see [measurement.md](measurement.md#lambda-the-policy-side-curve), so the policy decides in the unit it is scored in.

The estimated change from splitting a partition of size s with access A (Quake equation 6) is

```
ΔC_split = [λ_c(K + 1) − λ_c(K)] − A·λ(s) + 2α·A·λ(s / 2)
```

Each child is assumed to keep α of the parent's access, so the two together hold 2α·A. The first term is the cost of the extra centroid, without which ΔC would be negative for every accessed partition and the policy would split without limit.

An action is taken only when its ΔC is below −τ, that is when it is predicted to save more than τ.

### One pass

1. **Measure shrinkage.** For every partition, the share of its previous size it has lost since the last pass.
2. **Refresh centroids.** For every partition that changed, shift its centroid by the change: new centroid = (previous size × centroid + sum of vectors now − sum then) / size now. An unchanged partition is skipped, and the work is the number of changed vectors.
3. **Snapshot** sizes and vector sums, and compute A for every partition.
4. **Decide everything from that snapshot.** For each partition:
   - **Delete estimate.** `delta_merge_estimate_uniform` prices spreading the partition over the other T − 1 partitions, all taken as the average partition, exactly as the released code's `compute_delete_delta` does.
   - If that is below −τ and the partition has more than 32 vectors, **check it exactly** with `delta_merge_exact`, equation 5 over the partitions its vectors would really move to. A partition of 32 or fewer is deleted on the estimate alone, as in the released loop.
   - Otherwise, if it has more than 32 vectors, try the **split estimate** of equation 6.
   - A partition no action chose that lost more than 30% since the last pass is deleted regardless, the **forced delete**.
5. **Delete together.** All chosen centroids are removed first, then every vector of the deleted partitions goes to its nearest remaining partition.
6. **Split and verify.** Each chosen partition is split with the Faiss style 2-means (`faiss_split`). The split is kept only if the exact ΔC over the real children (equation 4) is still below −τ.
7. **Refine.** The 50 nearest partitions of every new child get one assignment pass against unchanged centroids (`reassign_boundary`).
8. **Close the window.** Remove empty partitions and reset every access count and the routed query count.

### Why α is 0.86

`IVFIndex.apply_split` credits each child 0.86 of the parent's access, the value measured on this data by `scripts/measure_split_alpha.py` (0.862, 0.862 and 0.824 on SIFT, 0.880 on GIST). If the policy predicted with 0.9 while the index assigned 0.86, the check of a split would compare its estimate with a number the index never produces. The same value is therefore used in the estimate and in the index.

### Behaviour under the Faiss prices

One SIFT vector costs about 12 ns to scan, so τ = 250 ns asks a split to save about 21 vectors of expected scanning per query. Few partitions qualify, deletes outweigh splits, and K falls from about 176 to about 79. On GIST one vector costs about 120 ns and the same τ is easily cleared. The τ = 50 ns configuration is the value the Quake authors use for Quake in their released SIFT1M maintenance experiment.

## The bandit policy

`src/maintainers/bandit.py`, with `src/bandit/context.py`, `src/bandit/linucb.py` and `src/bandit/reward.py`. The contribution of the thesis: a contextual bandit that chooses one of five actions for each partition, learning from the change in Quake's total cost.

| Parameter | Value |
|---|---|
| actions | split, centroid refresh, merge, compaction, no action |
| exploration weight β (`alpha` in the code) | 1.0 |
| discount per pass | 0.99, D-LinUCB, every action every pass |
| random decisions at the start | 200 |
| access decay per pass | 0.9 |
| minimum split size | 100 vectors (the target average) |
| coverage penalty weight | 2.0 |
| reward clip | [−1, 1] |
| candidates per pass | every partition |
| drift limit | 1.29% of the data scale, about 5 on SIFT |

### Trigger

A pass runs when the largest partition exceeds 400 live vectors, or any partition has fewer than 50 (empty ones included), or the mean centroid drift exceeds the drift limit.

### One pass

1. Remove empty partitions.
2. Multiply every access count and the routed query count by 0.9, so recent queries weigh more.
3. Discount every action's evidence by 0.99 (`WarmLinUCB.end_of_pass`).
4. For every partition, in turn:
   1. **Context.** Six numbers, each clamped to a range (`extract_context`).
   2. **Allowed actions.** No action always. Refresh if it has live vectors. Compaction if it has tombstones. Split if it has at least 100 vectors. Merge if one of its 64 nearest partitions fits with it, at most 400 live vectors together.
   3. **Choose.** Random among the allowed actions for the first 200 decisions, then the highest LinUCB score.
   4. **Act** and compute the **reward**.
   5. **Learn.** Update the chosen action's model with the context and reward.

### The context

| Feature | Formula | Range |
|---|---|---|
| size | live size / 400 | 0 to 2 |
| drift | centroid drift / drift limit | 0 to 2 |
| access | A, the share of routed queries that scanned it | 0 to 1 |
| tombstones | tombstoned / stored | 0 to 1 |
| relative size | live size / mean partition size | 0 to 4 |
| bias | 1 | 1 |

The ranges keep the context bounded, which the LinUCB analysis assumes.

### The actions

| Action | What it does | Action cost weight |
|---|---|---|
| split | 2-means, then every vector of the two children and their 25 nearest partitions moves to its nearest centroid among them (`reassign_boundary`) | 2.0 |
| merge | joins the partition with the first of its 64 nearest that fits, into one partition centred on the mean of both | 1.5 |
| refresh | moves the centroid to the mean of its vectors | 1.0 |
| compaction | removes tombstones | 1.0 |
| no action | nothing | 0 |

Unlike LIRE, the bandit's merge recentres the result on the mean of both partitions, and the fit is checked on live sizes.

### The reward

```
r = −ΔC / unit  −  0.1 · cost(a)  +  0.1 · drift closed / drift limit  +  0.1 · tombstone share removed
```

- ΔC is the change in the total Quake cost C of the whole index across the action (`index_cost`), so the neighbours a split takes vectors from and the centroid it adds are both priced.
- unit is the scan part of C divided by K, the mean scan cost of one partition.
- cost(a) is the action's weight times the vectors it touches, divided by 400.
- The drift term is credited only for refresh and compaction. A split or merge changes which vectors a centroid serves, so drift before and after would describe different sets of vectors.
- The result is clipped to [−1, 1]. For a split, a coverage penalty of 2 × max(0, 100 / mean size − 1) is then subtracted and the reward clipped again. The penalty is zero while partitions hold more than 100 vectors on average and discourages splitting an index that is already finely partitioned.

### The learner

D-LinUCB with disjoint models. Each action a keeps V_a, Ṽ_a and b_a, starting at the identity, the identity and zero.

```
θ_a = V_a⁻¹ b_a
score_a = θ_a · x + β · sqrt(xᵀ V_a⁻¹ Ṽ_a V_a⁻¹ x)
update after action a with reward r:  V_a += x xᵀ,  Ṽ_a += x xᵀ,  b_a += r x
before each pass:  V_a = I + 0.99 (V_a − I),  Ṽ_a = I + 0.99² (Ṽ_a − I),  b_a = 0.99 b_a
```

The discount fades old evidence and never the identity term, so the ridge regularisation stays constant. Discounting every action every pass means an action left unchosen regains uncertainty and is tried again.

### Two design properties

- After a split each child is credited 0.86 of the parent's access, not an observed value, so the reward of a split rests partly on the same assumption as Quake's estimate.
- Each decision is treated as affecting only its immediate reward, although splits and merges change the future index. This is a deliberate simplification, since the immediate change in modelled cost is available at once.

### Options present but off

The constructor also supports a reward measured on real queries, a do no harm margin, cooldowns against reversing a recent split or merge, a recency feature, a potential based reward and a warm start in the style of ARROW-CB. None is used in the reported runs. A warm start from another seed was rejected because it would give the bandit information the other policies do not have.

## Configurations outside the main comparison

| Name | What it is | Status |
|---|---|---|
| `cost_driven_quake_tau50` | Quake with τ = 50 ns for split and delete | the declared sensitivity configuration, reported but never as "Quake" |
| `bandit_linear` | the bandit without the cost model, its reward uses access times size | not reported |
| `lire_lite_no_merge` | LIRE with merging switched off | not reported |
| `global_rebuild_at_bandit_k` | the rebuild at the bandit's partition count per seed | the control run, added by `scripts/control_rebuild_at_k.py` |

Three policies from earlier stages, `cost_driven_old`, `drift_aware` and `hybrid`, are not included. The first combined Quake's cost model with a trigger of our own and was replaced by the faithful Quake port. The other two had parameters with no published source.

## Differences from the sources, summarised

| Policy | Difference | Reason |
|---|---|---|
| all | maintenance every 1,000 updates | affordable for a stream of about 230,000 updates, the same for every policy |
| Quake | α = 0.86 instead of 0.9 | measured on this data, so the estimate predicts what the index assigns |
| Quake | recall matched nprobe per window instead of per query adaptive scanning | the same recall target, one calibration per window |
| Quake | λ by size only | k is fixed at 10 |
| Quake | the delete check follows equation 5 of the paper | the released check adds the deleted partition's cost where the equation subtracts it, and would reject nearly every checked delete |
| LIRE | plain 2-means instead of balanced clustering | a balanced version was measured to do worse here and is switched off |
| LIRE | reassignment radius 25 instead of 64 | the published value of Ada-IVF |
| LIRE | merges checked every pass, and a partner may itself be waiting to merge | changing it would change the shared starting index, so it is recorded as a limitation |
| Global rebuild | rebuild at the current partition count | almost the same as Ada-IVF's target size at this scale |
| Rebuild, DeDrift | 2.5% of live vectors rather than of the whole workload | recorded, effect not measured |
