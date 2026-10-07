# Glossary

Terms used in the code and the documentation, what each means here, and where it comes from. Names in `code` are identifiers in the code.

## Index and data

| Term | Meaning | Source |
|---|---|---|
| IVF, inverted file index | An index that groups vectors around centroids and scans only the groups nearest a query. | Jégou et al., Faiss |
| partition | One group of vectors with its centroid. Called a list in Faiss and a posting in SPFresh. Class `Partition`. | general |
| centroid | The vector a partition is routed by, normally the mean of its vectors. | general |
| K | The number of partitions. Not to be confused with k. | general |
| k | The number of neighbours a query asks for, 10 throughout (`DEFAULT_K`). | general |
| nprobe | How many partitions a query opens. | Faiss |
| recall at 10 | The share of a query's 10 true nearest neighbours that the search returns. | general |
| live vector, live size | A vector not deleted, and a partition's count of them, `size()`. | this code |
| stored size | All vectors a partition stores, deleted ones included, `total_size()`. | this code |
| tombstone | The mark on a deleted vector that is still stored. | SPFresh |
| compaction | Rewriting a partition with its live vectors only. | SPFresh |
| collection | Removing an empty partition. Not counted as a merge. | this code |
| data scale | The norm of the per dimension standard deviation of the first build, about 387 on SIFT and 1.417 on GIST. Drift limits are fractions of it. | this code |
| drift, centroid drift | The distance between a partition's centroid and the mean of its live vectors. | DeDrift |
| routing | Choosing the nprobe partitions with the nearest centroids. | general |

## Workload

| Term | Meaning | Source |
|---|---|---|
| starting index, staple | The index every run starts from, grown under LIRE to 50,000 live vectors. `staple` is its name in the code. | this code |
| workload cluster | One of 64 regions of the dataset, from k-means, used to group arrivals. Not a partition. | Big ANN streaming track |
| runbook | The schedule of clustered insert and delete bursts used to grow the starting index. | Big ANN streaming track |
| retention window | The stream in which every batch expires a fixed number of steps after arriving. | Big ANN streaming track |
| time step | One of 100 steps of the stream, about 1,200 inserts each. | this code |
| run, cell | One policy on one dataset under one seed. Produces one result row. | this code |
| seed | The integer every random choice of a run is derived from. Five per dataset: 42, 1, 7, 13, 23. | general |
| routed query | A query from the stream, which only counts access. 8 follow every update. | this code |
| routed pool | The 4,000 queries the stream draws routed queries from. | this code |
| held out query | A query used only for scoring, never routed. 400 per reading. | this code |
| served sample | 400 queries from the routed pool, scored beside the held out ones. | this code |
| query region | The 10% of queries nearest one centre query, from which 80% of draws are taken. | Quake, Ada-IVF |
| access count | How many routed queries probed a partition in the current window. | Quake |
| access fraction, A | Access count divided by routed queries, the share of queries that scanned a partition. | Quake section 4.1 |

## Maintenance

| Term | Meaning | Source |
|---|---|---|
| policy, maintainer | A rule for when and how to restructure the index. `Maintainer` in the code. | this code |
| pass | One call of a policy's `maintain`. | this code |
| check interval | Updates between offers to act, 1,000. | this code |
| split | Replacing one partition with two, by 2-means. | general |
| merge | Joining a partition with another. LIRE keeps the survivor's centroid, the bandit recentres. | SPFresh |
| partition delete | Quake's merge, removing a partition and moving each vector to its nearest remaining partition. | Quake section 4.2.1 |
| reassignment | Moving single vectors to a nearer centroid after a split or merge. | SPFresh |
| LIRE conditions | The two necessary conditions that decide which vectors are checked after a split. | SPFresh section 3.3 |
| refinement | One assignment pass over the partitions near a new child. | Quake |
| repartition | Moving many vectors at once, in a rebuild or a DeDrift Split. | this code |
| LIRE | SPFresh's update protocol, `lire_lite`. | SPFresh |
| Lazy, Split, Hybrid | The three DeDrift strategies. | DeDrift section 5.1 |
| B1, B2, k2, mu | In DeDrift Split, the largest clusters, the smallest clusters added to keep K constant, the new cluster count, and the median cluster size. | DeDrift section 5.1 |
| global rebuild | Rebuilding the whole index with k-means after 2.5% of vectors changed. | Ada-IVF section 5.1 |

## Cost

| Term | Meaning | Source |
|---|---|---|
| scan distance | The cost unit, the time one more vector adds to scanning a long Faiss list. 11.94 ns on SIFT, 119.72 ns on GIST. Not a geometric distance. | this code |
| list price | The Faiss cost of opening and scanning one list of a given length, in scan distances. | this code |
| query cost | The priced cost of one query at recall 0.9, the sum of the list prices of the probed lists plus the centroid price times K. | this code |
| fixed recall | Reading cost at the nprobe that reaches a recall target, rather than at a fixed nprobe. | Quake |
| maintenance work | The priced cost of everything a policy did over a run. | this code |
| ingestion | The cost of taking in updates, centroid comparisons per insert and one copy per delete. | this code |
| total compute | Queries served times mean query cost, plus work, plus ingestion. | this code |
| efficiency | The query cost saved against no maintenance over the run, divided by work. | this code |
| break even reads | Queries per update needed before a policy's saving repays its work. | this code |
| λ(s), lambda | The cost of scanning a partition of s vectors, as the policies see it. | Quake section 4.1 |
| λ_c(K) | The cost of comparing a query with K centroids. | Quake equations 4 and 5 |
| C | Quake's total cost of the index, the sum of A times λ(s) plus λ_c(K). | Quake equation 1 |
| ΔC | The change in C an action is predicted or measured to cause. | Quake section 4.2 |
| τ, tau | The saving an action must exceed before Quake takes it, 250 ns. | Quake page 15 |
| α, alpha | The share of the parent's access each child of a split keeps, 0.86 here. | Quake section 4.2.2 |
| forced delete | Quake's delete of a partition that lost more than 30% since the last pass, without consulting ΔC. | Quake implementation |
| delete model | Whether a deleted vector is scanned until compaction (SPFresh) or gone at once (Quake, Faiss). | this code |
| fresh build at own K | A new k-means of the final live vectors at the policy's own partition count. | this code |

## Bandit

| Term | Meaning | Source |
|---|---|---|
| contextual bandit | A learner that chooses an action from a context and learns from the reward it gets. | LinUCB |
| context | The six numbers describing a partition. | LinUCB |
| action, arm | One of split, refresh, merge, compaction, no action. | LinUCB |
| reward | The score of one action, mainly the drop in C. | LinUCB |
| LinUCB | Linear upper confidence bound learning, one ridge regression per action plus an optimism term. | Li et al., 2010 |
| D-LinUCB | LinUCB with discounted evidence, for a changing environment. | Russac, Vernade and Cappé, 2019 |
| β | The weight of the optimism term, 1.0. Called `alpha` in the code. | LinUCB |
| coverage penalty | A reward penalty on splits once partitions average fewer than 100 vectors. | this code |

## Reproducibility

| Term | Meaning | Source |
|---|---|---|
| fingerprint | A hash of the code, prices, machine, versions, threads and parameters, stored with every row. | this code |
| provenance file | The `.provenance.json` beside a result file, with the details behind each fingerprint. | this code |
| resume | Rerunning only the runs missing from a result file, or produced by other code. | this code |
| sweep | A set of runs, for example `final_2709`. | this code |
| `final_2709` | The main runs, nine configurations, two datasets, five seeds. | this code |
| `sens_ta50_2709` | The half size check, a target of 50 vectors per partition on SIFT. | this code |
| `control_2809` | The rebuild at the bandit's partition count. | this code |
| `diag_2809` | The K curves and the DeDrift Split trace. | this code |
