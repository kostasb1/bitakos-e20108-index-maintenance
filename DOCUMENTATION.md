# Documentation

This file goes through the codebase with a brief explanation of every code file,files are also commented.

## High level of the system

One run takes one dataset, one seed and one maintenance policy. It grows a starting index of 50,000
vectors, then plays a stream of 100 time steps in which new vectors arrive in one region of the data
and older ones expire. After every insert or delete, 8 queries are routed through the index so the
policies can see which partitions are read. Every 1,000 updates the policy gets a chance to
maintain the index. The index is scored at the start, at time steps 25, 50 and 75, and at the end: we find the
nprobe that reaches recall 0.9 on held-out queries and price what such a query costs. The result is
one row in a CSV file.

The code is divided into two folders. `src/` holds the index, the policies and the scoring. `scripts/` holds
the programs that run the experiments and turn the results into the tables and figures of the
thesis.

## The index, in src/

### src/vector.py

This file has the distance calculations. The most important function scans one block of vectors and returns the
distance of each to a query. Both the search and the scoring use this same function, so they always
compute distances the same way.

### src/partition.py

One partition of the index. It stores its vectors in a single block of memory that doubles in size
when it fills up, and keeps a table from vector id to row. A delete does not move anything, it only
marks the vector as deleted (a tombstone). The vector stays stored until a compaction rewrites the
partition. So a partition has a live size, the vectors that count, and a stored length, which also
includes the deleted ones.

A partition also keeps its centroid, which is what queries and inserts are routed by, the mean of
its live vectors, which is recomputed only after a change, and an access count that goes up every
time a query from the stream reads it. The distance between the centroid and the mean is the
centroid drift.

### src/index.py

The IVF index, a set of partitions with one centroid each. `IVFIndex.build` clusters a set of
vectors with exact k-means. `search` answers a query, `route` only finds the partitions a query
would read and increases their access counts, and `insert` and `delete` add and remove vectors.

Routing has two steps. A fast step ranks all centroids and keeps a few more candidates than needed,
and an exact step recomputes the distance to those candidates and picks the nearest. The exact step
always makes the choice, and the fast step is there only to save time.

### src/types.py

Small records shared by the rest of the code. The one that matters most is `MaintenanceReport`, where
every maintenance operation writes how many vectors it read and how many distances it computed. The
maintenance work in the thesis is computed from these counts.

### src/config.py

The constants used across the code, for example the 1,000 updates between maintenance passes, the
2.5 percent rebuild trigger and the split share 0.86, each with a comment that says where the value
comes from.

### src/dataset.py

Reads the SIFT1M and GIST1M files and computes exact nearest neighbours. The exact neighbours use a
Faiss flat index, which compares a query with every vector, or scikit-learn if Faiss is not installed.

## The policies, in src/maintainers/

### src/maintainers/base.py

It has the `Maintainer` class that every policy extends. A policy says when it wants to
act (`should_maintain`) and what it does (`maintain`). It also has the operations that change the index,
shared by all policies: splitting a partition in two with 2-means, merging two partitions, deleting
a partition and reinserting its vectors, and moving boundary vectors to a nearer partition after a
split or merge. Because every policy uses these same operations, the same action always costs the
same.

### src/maintainers/no_op.py

The policy that does no maintenance at all. Every other policy is compared with this one on the same seed.

### src/maintainers/global_rebuild.py

The periodic rebuild used as a baseline by Ada-IVF. Once inserts plus deletes since the last rebuild
reach 2.5 percent of the live vectors, it rebuilds the whole index with k-means at the current number
of partitions. Since the check runs every 1,000 updates, this happens every 2,000 updates in practice.

### src/maintainers/lire_lite.py

LIRE, the update protocol of SPFresh. A partition with more than 400 stored vectors is compacted and,
if it is still too big, split. A partition with fewer than 50 live vectors is merged into a nearby
one. After each split or merge the vectors near the boundary are checked against the 25 nearest
partitions and moved if another centroid is closer. The same class is used to grow the starting
index.

### src/maintainers/dedrift.py

The three DeDrift strategies. Lazy recomputes every centroid as the mean of its partition. Split takes
the largest partitions together with some of the smallest and clusters them again into the same
number of partitions. Hybrid does Lazy and then Split. They run on the same 2.5 percent trigger as the
rebuild.

### src/maintainers/cost_driven_quake.py

Quake's maintenance loop. For each partition it estimates, from how often the partition is read and
from a curve of scan cost against partition size, how much a split or a delete would change the
expected query cost. It acts only when the predicted saving is larger than a threshold τ (250 ns as
published, 50 ns in the second configuration), and then checks the prediction again on the real
result before keeping the change. Its decisions rest only on the access counts and a fixed threshold, without learning from earlier
outcomes.

### src/maintainers/bandit.py

The bandit policy. For each partition it chooses one of five actions, do nothing, refresh the
centroid, compact, split or merge, using the LinUCB learner in `src/bandit/`. An action is offered only
when it can run, for example a split only for partitions of at least 100 vectors. After each action it
computes the reward and updates the learner. The first 200 decisions are random, to give the learner
something to start from.

## Parts of the bandit policy, in src/bandit/

### src/bandit/context.py

The six numbers that describe a partition to the bandit: its size, its centroid drift, how often it
is read, its share of deleted vectors, its size compared with the average, and a constant 1.

### src/bandit/linucb.py

LinUCB is the learner. For each action it fits a linear estimate of the reward from the six numbers, adds a
bonus for actions it knows little about, and picks the action with the highest total. Older
observations are discounted so recent ones count more. The reported runs use `WarmLinUCB` with its
warm start switched off.

### src/bandit/reward.py

The reward is computed with Quake's cost model. It estimates the total query cost of the index before
and after an action, and the reward is the saving minus a small charge for the action.

## The stream and the scoring, in src/

### src/staple.py

Grows the starting index, called the staple in the code. The dataset is first divided into 64
workload clusters with k-means. Vectors then arrive cluster by cluster following the schedule of the
Big ANN streaming benchmark (ported from its published generator, with deletes scaled to one for
every four inserts), while LIRE keeps the index in shape, until 50,000 vectors are live. Growing the
index this way avoids the near-empty partitions that k-means leaves on GIST.

### src/stream.py

The stream every result is measured on. `run_stream` plays the 100 time steps: about 1,200 inserts
per step around a workload cluster that moves over time, every batch expiring 50 steps after it
arrived, 8 routed queries after every update, and the policy called every 1,000 updates.
`split_query_pools` divides the queries into a routed half and a held-out half for scoring, and
`measure` takes one reading of the scores.

### src/metrics.py

`compute_live_ground_truth` finds the true 10 nearest neighbours among the vectors live
at that moment. `cost_at_target_recall` finds the nprobe that reaches recall 0.9, by doubling nprobe
and then narrowing down, and prices a query at that point. `stored_cost_at` does the same under the
SPFresh delete model, where deleted vectors are still scanned. Nothing here records access, so
scoring never changes what a policy sees.

### src/calibration.py

The prices that turn counts into one cost unit. They were measured once on Faiss and are read from
`results/cost_model/query_cost_calibration.yaml`. `query_cost` prices a query from the lists it
opened and the centroids it compared, and `work_cost` prices maintenance work from the counts in a
`MaintenanceReport`.

### src/cost_model.py

The scan cost curve λ(s) that the Quake and bandit policies use to make decisions, built from the
Faiss prices by `lambda_from_prices`. The file also has older functions that timed the Python scan.
The reported runs do not use them.

### src/provenance.py

A fingerprint of the code, the data and the settings, written next to every result row. It lets a
long set of runs be stopped and resumed, and it refuses to mix rows produced by different versions
of the code.

## scripts/

### scripts/final_sweep.py

Runs the experiments, every combination of dataset, seed and policy, and writes one row per run to a
CSV file. `build` creates a policy with the exact settings of the reported runs.

### scripts/analyse_final.py

Reads the result files and produces the tables of Chapter 6 and Appendix I: each policy against no
maintenance on the same seed, the confidence intervals, the work, and the comparisons at the same
number of partitions.

### scripts/calibrate_query_cost.py

Measures the prices of the cost unit on Faiss: the cost of opening and scanning a list of each length,
the cost of comparing with a centroid, and the cost of k-means for the maintenance work. Writes them to
`results/cost_model/`.

### scripts/control_rebuild_at_k.py

The control run of Section 6.5: the global rebuild, run with the number of partitions the bandit
policy ended at on the same seed.

### scripts/k_curve.py

Builds fresh indexes at several numbers of partitions and measures their query cost (Figure 6.2).

### scripts/rebuild_noise.py

Rebuilds the same vectors several times and measures how much one reading varies from build to
build (Section 6.2).

### scripts/dedrift_split_trace.py

Records what DeDrift Split does to the index in each pass of one run (Section 6.8).

### scripts/measure_split_alpha.py

Measures how much of a partition's access each half keeps after a split, which gave the value 0.86
used by Quake and the bandit (Section 4.6).

### scripts/make_figures.py

Draws the figures of the thesis, in English and Greek.

### scripts/download_datasets.py

Downloads SIFT1M and GIST1M into `data/`.

## results/

The result files of every run reported in the thesis, the tables made from them, and the measured
prices. Each result file has a `.provenance.json` file next to it with the fingerprint of the code
and settings that produced it.
