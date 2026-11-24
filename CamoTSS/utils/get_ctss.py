import numpy as np
import pandas as pd
import os
import pickle
import anndata as ad
import time
import logging
import warnings
from scipy import sparse as sp



warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=Warning)


# ----------------------------------------------------------------------------------------------------------------------
# Shared CTSS (single-base TSS) machinery, used by BOTH modes:
#   --mode CTSS     -> get_CTSS_count.produce_CTSS_adata()                   (this file)
#   --mode TC+CTSS  -> get_TSS_count.produce_CTSS_adata() in get_counts.py
# Changes 2026-11-23:
#   * ONE sliding window for both modes: window_sliding() below = the old --mode CTSS window (every base counts, bases
#     without reads count 0, so a window is windowSize consecutive bases) with one fix: the cluster's LAST base is now
#     included (the old code used range(TSS_start, TSS_end), which stopped one base short; a cluster's edge base always
#     carries reads). The old --mode TC+CTSS window used only bases that HAVE reads, so its 15 "positions" could span
#     hundreds of bp and its last 14 read-bearing bases could never be tested; the two modes called different CTSS from
#     the same data (human: 805 vs 341 CTSS; with the last-base fix 849).
#   * each kept cluster's reads (every read of its gene inside [TSS_start, TSS_end], the same reads the old code
#     selected from fetch_reads.pkl) are read one cluster at a time from the read store (count/fetch_reads_store,
#     read_store.py) instead of loading the whole fetch_reads.pkl. For output folders made by earlier versions of the
#     fork (no store), keepdict.pkl is used instead.
#     NOTE: for an output folder made by the OLD sampled clustering, keepdict.pkl misses some reads inside the bounds
#     (the 20,000-read sample + add-back), so CTSS mode on such a folder can give lower counts than before.
#   * cells at each CTSS are found with one numpy comparison per cluster, not a Python loop over every read of the gene
#     for every CTSS.
#   * the cell x CTSS matrix is built sparse (CSR, float32), like scTSS_count_all.h5ad, instead of a dense pandas
#     DataFrame filled one column at a time. Same numbers; cells (obs) are sorted instead of in random set order.
#   * CTSS_foldchange.pkl is reused on a rerun (resume after a crash) ONLY if CTSS_foldchange_params.txt shows it was
#     made with the same window settings; otherwise it is recomputed.
#   * var no longer gets a meaningless 'index' column (generefdf.reset_index() was copied into var on every run).
# ----------------------------------------------------------------------------------------------------------------------

WINDOW_VERSION = "consecutive-bases-incl-last-base-v1"     # written next to CTSS_foldchange.pkl; change if the window changes


def window_sliding(genereads, TSS_start, TSS_end, strand, windowSize):
    """Fold change of every base of one TSS cluster against the windowSize bases downstream of it.

    genereads : list of (position, barcode, CIGAR) reads of the cluster; only reads with an unencoded G (CIGAR soft clip
                14S/15S/16S) are used here
    Every base from TSS_start to TSS_end INCLUSIVE is a position; bases without unencoded-G reads count 0. Bases are
    ordered along transcription (ascending on '+', descending on '-'), and for each base
        fold change = (reads at the base + 1) / (mean reads over the base and the next windowSize-1 bases + 1)
    Returns [[base, unencoded-G reads at the base, fold change], ...] sorted by fold change, highest first. As before,
    the last windowSize-1 bases of the cluster (downstream end) have no full window and are not scored.
    """
    leftIndex = 0
    filterls = [i for i in genereads if ('14S' in i[2]) or ('15S' in i[2]) or ('16S' in i[2])]

    # unencoded-G read count at each base
    promoterTSS = [read[0] for read in filterls if (read[0] >= TSS_start) and (read[0] <= TSS_end)]
    TSS, count = np.unique(promoterTSS, return_counts=True)

    originaldf = pd.DataFrame({'count': count}, index=TSS)
    desireddf = pd.DataFrame({'count': 0}, index=[i for i in range(int(TSS_start), int(TSS_end) + 1)])   # + 1: last base
    desireddf['count'] = desireddf.index.map(originaldf['count']).fillna(0)
    desireddf.reset_index(inplace=True)
    nonzeroarray = np.array(desireddf)

    if strand == '+':
        sortfinalarray = nonzeroarray[nonzeroarray[:, 0].argsort()]
    elif strand == '-':
        sortfinalarray = nonzeroarray[nonzeroarray[:, 0].argsort()[::-1]]

    TSS = sortfinalarray.T[0]
    count = sortfinalarray.T[1]

    # sliding window
    storels = []
    for i in range(len(TSS) - windowSize + 1):
        onewindow = TSS[i: i + windowSize]
        correspondingcount = count[i: i + windowSize]
        middlecount = correspondingcount[leftIndex]
        foldchange = (middlecount + 1) / (sum(correspondingcount) / len(correspondingcount) + 1)
        storels.append([onewindow[leftIndex], correspondingcount[leftIndex], foldchange])

    foldchangels = [i[2] for i in storels]
    sortindex = sorted(range(len(foldchangels)), key=lambda k: foldchangels[k], reverse=True)
    allsortls = [storels[i] for i in sortindex]

    return allsortls


def load_kept_cluster_reads(count_out_dir):
    """Reads of every kept TSS cluster, keyed by the cluster name used in afterfiltered.csv ('gene*start_end').

    keepdict.pkl (written by _filter_false_positive) maps gene_id -> list of kept clusters, each
    [positions (n,1), barcodes (n,1), CIGARs (n,1)]; the name is rebuilt exactly as _filter_false_positive builds it.
    Returns {cluster name: (positions 1-D, barcodes 1-D, CIGARs 1-D)}.
    """
    with open(os.path.join(count_out_dir, 'keepdict.pkl'), 'rb') as f:
        keepdict = pickle.load(f)
    clusters = {}
    for geneid, clusterls in keepdict.items():
        for c in clusterls:
            name = str(geneid) + '*' + str(np.min(c[0])) + '_' + str(np.max(c[0]))
            clusters[name] = (c[0].ravel(), c[1].ravel(), c[2].ravel())
    del keepdict
    return clusters


class StoreClusterReads:
    """{cluster name: (positions, barcodes, CIGARs)} for kept TSS clusters, read from the read store on demand.

    The cluster named 'gene*start_end' holds every read of the gene whose position lies in [start, end] (all-reads
    clustering), so its reads are taken straight from the store -- nothing is held in memory between clusters.
    """

    def __init__(self, count_out_dir):
        from .read_store import ReadStore, store_dir_of
        self.store = ReadStore(store_dir_of(count_out_dir))

    def __getitem__(self, name):
        geneid, se = name.split('*')
        start, end = (int(x) for x in se.split('_'))
        pos, cb, cig = self.store.get(geneid)
        m = (pos >= start) & (pos <= end)
        return pos[m], self.store.barcodes[cb[m]], self.store.cigars[cig[m]]


def cluster_reads_source(count_out_dir):
    """Where CTSS gets each kept cluster's reads: the read store (runs made with the low-memory code), or keepdict.pkl
    (output folders made by earlier versions of the fork)."""
    from .read_store import has_store
    if has_store(count_out_dir):
        return StoreClusterReads(count_out_dir)
    return load_kept_cluster_reads(count_out_dir)


def pick_ctss(ctssls, minCTSSCount, minFC):
    """Keep CTSS with more than minCTSSCount unencoded-G reads and fold change above minFC (unchanged rule)."""
    return [ele for ele in ctssls if (ele[1] > minCTSSCount) & (ele[2] > minFC)]


def build_ctss_adata(clusters, alloneclusterdf, generefdf, windowSize, minCTSSCount, minFC, ctss_out_dir):
    """Detect CTSS inside every kept TSS cluster and write all_ctss.h5ad and all_ctss_two.h5ad.

    clusters        : output of load_kept_cluster_reads()
    alloneclusterdf : afterfiltered.csv read with pd.read_csv (cluster name in column 'Unnamed: 0')
    generefdf       : ref_gene.tsv as a DataFrame (needs 'gene_id' and 'Strand')
    windowSize      : bases per sliding window (--windowSize, default 15)
    """
    ctime = time.time()
    df = alloneclusterdf.copy()
    df['gene_id'] = df['Unnamed: 0'].str.split('*', expand=True)[0]
    df['TSS_start'] = df['Unnamed: 0'].str.split('*', expand=True)[1].str.split('_', expand=True)[0].astype('float')
    df['TSS_end'] = df['Unnamed: 0'].str.split('_', expand=True)[1].astype('float')
    df = df.merge(generefdf[['Strand', 'gene_id']], on='gene_id').reset_index(drop=True)

    # --- step 1: sliding-window fold change inside every kept cluster ---
    # On a rerun the saved CTSS_foldchange.pkl is reused only if it was made with the same window settings.
    ctssOutPath = os.path.join(ctss_out_dir, 'CTSS_foldchange.pkl')
    paramsPath = os.path.join(ctss_out_dir, 'CTSS_foldchange_params.txt')
    params = f"window={WINDOW_VERSION} windowSize={windowSize} clusters={len(df)}"
    saved = open(paramsPath).read().strip() if os.path.exists(paramsPath) else None
    if os.path.exists(ctssOutPath) and saved == params:
        logging.warning("[CTSS] Found CTSS_foldchange.pkl made with the same settings -- resuming from it.")
        with open(ctssOutPath, 'rb') as f:
            allsortfddict = pickle.load(f)
    else:
        start_time = time.time()
        allsortfddict = {}
        for i in range(len(df)):
            clusterID = df['Unnamed: 0'][i]
            pos, cb, cigar = clusters[clusterID]
            genereads = list(zip(pos.tolist(), cb.tolist(), cigar.tolist()))
            allsortfddict[clusterID] = window_sliding(genereads, df['TSS_start'][i], df['TSS_end'][i], df['Strand'][i],
                                                      windowSize)
        print('window sliding Time elapsed', int(time.time() - start_time), 'seconds.')
        with open(ctssOutPath, 'wb') as f:
            pickle.dump(allsortfddict, f)
        with open(paramsPath, 'w') as f:
            f.write(params + '\n')

    # --- step 2: keep CTSS passing the count / fold-change filters; collect the cells of each kept CTSS ---
    col_ids, col_cells, col_counts = [], [], []
    for clusterID, ctssls in allsortfddict.items():
        pos, cb, _ = clusters[clusterID]
        for j in pick_ctss(ctssls, minCTSSCount, minFC):
            newid = clusterID + '#' + str(j[0]) + '@' + str(j[1]) + '$' + str(j[2])
            cellID, count = np.unique(cb[pos == j[0]], return_counts=True)   # every read (any CIGAR) at this base
            col_ids.append(newid); col_cells.append(cellID); col_counts.append(count)

    # --- step 3: sparse cell x CTSS matrix ---
    final_index = pd.Index(sorted(set(c for cells in col_cells for c in cells)))
    rows = [final_index.get_indexer(cells) for cells in col_cells]
    cols = [np.full(len(cells), k, dtype=np.int64) for k, cells in enumerate(col_cells)]
    X = sp.coo_matrix((np.concatenate(col_counts).astype(np.float32) if col_counts else np.zeros(0, np.float32),
                       (np.concatenate(rows) if rows else np.zeros(0, np.int64),
                        np.concatenate(cols) if cols else np.zeros(0, np.int64))),
                      shape=(len(final_index), len(col_ids)), dtype=np.float32).tocsr()

    ctssvardf = pd.DataFrame({'clusterID': col_ids})
    ctssvardf['gene_id'] = ctssvardf['clusterID'].str.split('*', expand=True)[0]
    ctssvardf['CTSS'] = ctssvardf['clusterID'].str.split('#', expand=True)[1].str.split('@', expand=True)[0]
    ctssvardf['counts_dropped_UnencodedG'] = ctssvardf['clusterID'].str.split('@', expand=True)[1].str.split('$', expand=True)[0]
    ctssvardf['fold_change'] = ctssvardf['clusterID'].str.split('$', expand=True)[1]
    ctssvardf = ctssvardf.merge(generefdf, on='gene_id', how='left')        # left: keeps column order and length
    ctssvardf.set_index('clusterID', drop=True, inplace=True)

    ctssadata = ad.AnnData(X=X, obs=pd.DataFrame(index=final_index), var=ctssvardf)
    logging.warning(f"[CTSS] {ctssadata.n_vars} CTSS x {ctssadata.n_obs} cells, {X.nnz} nonzeros.")
    ctssadata.write(os.path.join(ctss_out_dir, 'all_ctss.h5ad'))

    twoctssselect = ctssadata.var[ctssadata.var.duplicated('gene_id', keep=False)].index
    twoctssadata = ctssadata[:, twoctssselect]
    twoctssadata.write(os.path.join(ctss_out_dir, 'all_ctss_two.h5ad'))

    print('produce CTSS h5ad Time elapsed', int(time.time() - ctime), 'seconds.')
    return twoctssadata


class get_CTSS_count():


    def __init__(self,out_dir,minCTSSCount,minFC,n_proc,windowSize):
        self.out_dir=out_dir
        generefPath=os.path.join(out_dir,'ref_file','ref_gene.tsv')
        afterfiltered_path=os.path.join(out_dir,'count','afterfiltered.csv')

        self.generefdf=pd.read_csv(generefPath,delimiter='\t')
        self.alloneclusterdf=pd.read_csv(afterfiltered_path)

        self.minCTSSCount=minCTSSCount
        self.minFC=minFC
        self.n_proc=n_proc
        self.windowSize=windowSize



    def produce_CTSS_adata(self):
        self.ctss_out_dir=os.path.join(self.out_dir+'/CTSS/')
        if not os.path.exists(self.ctss_out_dir):
            os.mkdir(self.ctss_out_dir)
        clusters = cluster_reads_source(os.path.join(self.out_dir, 'count'))
        return build_ctss_adata(clusters, self.alloneclusterdf, self.generefdf, self.windowSize,
                                self.minCTSSCount, self.minFC, self.ctss_out_dir)
