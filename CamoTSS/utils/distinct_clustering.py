"""Average-linkage clustering of read start positions on ALL reads, computed on distinct positions.

CamoTSS finds TSS clusters by clustering the start positions of a gene's reads with
    AgglomerativeClustering(n_clusters=None, linkage='average', distance_threshold=InnerDistance)
"Average linkage" = start with every read as its own cluster; repeatedly merge the two clusters whose AVERAGE
read-to-read distance is smallest; stop when that smallest average distance is >= InnerDistance.
Run on every read, sklearn needs memory and time ~ (number of reads)^2, so it cannot cluster large genes
(a 50,000-read gene needs ~10 GB; an 850,000-read gene ~3 TB). The fork therefore used to cluster a 20,000-read
sample and add the other reads back, which hid and trimmed TSS. This module gives the SAME clusters as running
sklearn on every read, without a sample. Three facts make that exact:

  1. Reads on the same base are 0 bp apart, so full clustering always merges them first. After those merges
     there is one cluster per distinct position holding w reads -- which is where this algorithm starts.
  2. On a line (1-D), if every read of cluster A lies left of every read of cluster B, every read-to-read distance
     is (b - a), so the AVERAGE distance is simply mean(B) - mean(A) (means weighted by read counts).
     Example: A = {10, 10, 12}, B = {40}: pairs 30, 30, 28 -> average 29.33 = 40 - mean(10, 10, 12).
  3. On a line the closest pair of clusters is always a pair of NEIGHBOURS: if A < B < C, every A-to-C distance
     is larger than every A-to-B distance. So clusters stay contiguous runs of positions, and only neighbouring
     runs ever need comparing.

So: put the distinct positions in order, keep merging the neighbouring pair whose weighted means are closest
(a heap finds it quickly), stop when the closest pair is >= InnerDistance apart. Cost ~ m log m for m distinct
positions, independent of how many reads sit on each position.

Ties: positions are whole numbers, so two merges are often EXACTLY equally close (dozens of times per gene), and which
one goes first can change the final clusters. sklearn settles ties by the order the reads are listed in (BAM order),
so its own answer can change if the same reads are shuffled. This function compares distances exactly (integer
arithmetic for near-equal values) and settles exact ties left to right, so it is deterministic. Tests (2026-11-23,
human fetch_reads.pkl): an independent brute-force average linkage on every read, exact fractions, ties -> leftmost
pair, reproduces this function exactly; sklearn differs from it only where exact ties occur.
"""
import heapq
from fractions import Fraction
import numpy as np


class _Distance:
    """Average distance between two neighbouring clusters, (S_j*N_i - S_i*N_j) / (N_i*N_j), ordered EXACTLY.

    Floating point alone cannot order exact ties correctly (two equal averages can round to slightly different floats),
    so when two distances are within 1e-6 bp of each other they are compared exactly with integers; exact ties go to
    the pair whose left cluster starts further left (the stated tie rule).
    """
    __slots__ = ("f", "num", "den", "i")

    def __init__(self, num, den, i):
        self.num, self.den, self.i = num, den, i
        self.f = num / den

    def __lt__(self, other):
        if abs(self.f - other.f) > 1e-6:
            return self.f < other.f
        c = self.num * other.den - other.num * self.den
        if c != 0:
            return c < 0
        return self.i < other.i


def cluster_distinct_positions(positions, threshold=100):
    """Cluster read start positions exactly like average linkage on every read.

    positions : 1-D array of read start positions (one entry per read; repeats allowed)
    threshold : InnerDistance; clusters are merged only while their average distance is < threshold
                (sklearn's rule: a merge at or above distance_threshold is not made)
    Returns (uniq, counts, labels):
        uniq   = sorted distinct positions
        counts = reads at each distinct position
        labels = cluster number of each distinct position (0, 1, 2, ... from left to right)
    """
    uniq, counts = np.unique(np.asarray(positions, dtype=np.int64), return_counts=True)
    m = len(uniq)
    if m == 1:
        return uniq, counts, np.zeros(1, dtype=np.int64)

    # Each live cluster is a contiguous run of distinct positions, stored at the index of its leftmost position.
    # Positions are shifted to start at 0 (distances do not change) so the sums stay small and floats stay accurate.
    base = int(uniq[0])
    S = [(int(p) - base) * int(c) for p, c in zip(uniq, counts)]   # sum of read positions in the cluster (exact ints)
    N = [int(c) for c in counts]                          # number of reads in the cluster
    right = list(range(1, m)) + [-1]                      # index of the neighbouring cluster to the right (-1 = none)
    left = [-1] + list(range(0, m - 1))                   # index of the neighbouring cluster to the left
    alive = [True] * m
    version = [0] * m                                     # bumped when a cluster changes, to skip stale heap entries
    heap = []
    thr = Fraction(threshold)                             # exact threshold (100.0 -> 100)

    def push(i):
        j = right[i]
        if j != -1:
            # average read-to-read distance between neighbours = difference of their mean positions (fact 2),
            # kept as the exact fraction (S_j*N_i - S_i*N_j) / (N_i*N_j)
            d = _Distance(S[j] * N[i] - S[i] * N[j], N[i] * N[j], i)
            heapq.heappush(heap, (d, i, j, version[i], version[j]))

    for i in range(m - 1):
        push(i)

    while heap:
        d, i, j, vi, vj = heapq.heappop(heap)
        if not (alive[i] and alive[j] and right[i] == j and version[i] == vi and version[j] == vj):
            continue                                       # stale entry: one of the two clusters has changed
        if d.num * thr.denominator >= thr.numerator * d.den:
            break                                          # closest pair is >= threshold apart (exact): stop merging
        S[i] += S[j]                                       # merge cluster j (the right neighbour) into cluster i
        N[i] += N[j]
        alive[j] = False
        right[i] = right[j]
        if right[j] != -1:
            left[right[j]] = i
        version[i] += 1
        if left[i] != -1:
            push(left[i])                                  # new distance to the left neighbour
        push(i)                                            # new distance to the right neighbour

    labels = np.empty(m, dtype=np.int64)
    lbl = -1
    for k in range(m):
        if alive[k]:
            lbl += 1                                       # a live index starts a new run (cluster)
        labels[k] = lbl
    return uniq, counts, labels


def cluster_gene_reads(readinfo, inner_distance, min_count):
    """Cluster ALL reads of one gene; return CamoTSS's cluster list (clusters with >= min_count reads).

    readinfo : list of (TSS position, cell barcode, CIGAR) tuples, as stored in fetch_reads.pkl
    Returns a list of [positions (n,1), barcodes (n,1), CIGARs (n,1)] numpy arrays, one entry per cluster,
    clusters ordered left to right, reads inside a cluster in their original order -- the same structure the
    sklearn-based code produced (altTSSls), so everything downstream is unchanged.
    """
    pos = np.array([t[0] for t in readinfo], dtype=np.int64)
    uniq, counts, labels = cluster_distinct_positions(pos, inner_distance)
    n_reads = np.bincount(labels, weights=counts).astype(np.int64)        # reads per cluster
    keep = np.flatnonzero(n_reads >= min_count)
    if len(keep) == 0:
        return []
    read_label = labels[np.searchsorted(uniq, pos)]                       # cluster of every read
    sel = np.flatnonzero(np.isin(read_label, keep))                       # reads in kept clusters, original order
    order = sel[np.argsort(read_label[sel], kind="stable")]               # grouped by cluster, order kept inside
    bounds = np.cumsum(n_reads[keep])[:-1]
    CB = np.array([readinfo[i][1] for i in order]).reshape(-1, 1)
    cigar = np.array([readinfo[i][2] for i in order]).reshape(-1, 1)
    posi = pos[order].reshape(-1, 1)
    return [[p, c, g] for p, c, g in zip(np.split(posi, bounds), np.split(CB, bounds), np.split(cigar, bounds))]
