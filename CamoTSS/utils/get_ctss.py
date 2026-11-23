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
# Each mode keeps its own window_sliding() unchanged (the two differ; see the notes on each), so the CTSS each mode
# reports is the same as before. What changed (2026-11-23), all without changing which CTSS are called or their counts:
#   * reads come from keepdict.pkl (the reads of each kept TSS cluster) instead of loading the whole fetch_reads.pkl.
#     Clustering now uses ALL reads (distinct_clustering.py), so a kept cluster holds exactly every read of its gene
#     inside [TSS_start, TSS_end] -- the same reads the old code selected from fetch_reads.pkl -- at a fraction of
#     the memory (human: 8% of reads are in kept clusters).
#     NOTE: for an output folder made by the OLD sampled clustering, keepdict.pkl misses some reads inside the bounds
#     (the 20,000-read sample + add-back), so CTSS mode on such a folder can give lower counts than before.
#   * cells at each CTSS are found with one numpy comparison per cluster, not a Python loop over every read of the gene
#     for every CTSS.
#   * the cell x CTSS matrix is built sparse (CSR, float32), like scTSS_count_all.h5ad, instead of a dense pandas
#     DataFrame filled one column at a time. Same numbers; cells (obs) are sorted instead of in random set order.
#   * CTSS_foldchange.pkl is reused when present (resume after a crash), like the cluster-mode checkpoints.
#   * var no longer gets a meaningless 'index' column (generefdf.reset_index() was copied into var on every run).
# ----------------------------------------------------------------------------------------------------------------------

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


def pick_ctss(ctssls, minCTSSCount, minFC):
    """Keep CTSS with more than minCTSSCount unencoded-G reads and fold change above minFC (unchanged rule)."""
    return [ele for ele in ctssls if (ele[1] > minCTSSCount) & (ele[2] > minFC)]


def build_ctss_adata(clusters, alloneclusterdf, generefdf, window_sliding, minCTSSCount, minFC, ctss_out_dir):
    """Detect CTSS inside every kept TSS cluster and write all_ctss.h5ad and all_ctss_two.h5ad.

    clusters        : output of load_kept_cluster_reads()
    alloneclusterdf : afterfiltered.csv read with pd.read_csv (cluster name in column 'Unnamed: 0')
    generefdf       : ref_gene.tsv as a DataFrame (needs 'gene_id' and 'Strand')
    window_sliding  : the mode's sliding-window function, called as window_sliding(reads, TSS_start, TSS_end, strand)
                      with reads = list of (position, barcode, CIGAR); returns [[CTSS, unencoded-G count, fold change]]
                      sorted by fold change
    """
    ctime = time.time()
    df = alloneclusterdf.copy()
    df['gene_id'] = df['Unnamed: 0'].str.split('*', expand=True)[0]
    df['TSS_start'] = df['Unnamed: 0'].str.split('*', expand=True)[1].str.split('_', expand=True)[0].astype('float')
    df['TSS_end'] = df['Unnamed: 0'].str.split('_', expand=True)[1].astype('float')
    df = df.merge(generefdf[['Strand', 'gene_id']], on='gene_id').reset_index(drop=True)

    # --- step 1: sliding-window fold change inside every kept cluster (reuse the saved result on a rerun) ---
    ctssOutPath = os.path.join(ctss_out_dir, 'CTSS_foldchange.pkl')
    if os.path.exists(ctssOutPath):
        logging.warning("[CTSS] Found existing CTSS_foldchange.pkl -- resuming from it.")
        with open(ctssOutPath, 'rb') as f:
            allsortfddict = pickle.load(f)
    else:
        start_time = time.time()
        allsortfddict = {}
        for i in range(len(df)):
            clusterID = df['Unnamed: 0'][i]
            pos, cb, cigar = clusters[clusterID]
            genereads = list(zip(pos.tolist(), cb.tolist(), cigar.tolist()))
            allsortfddict[clusterID] = window_sliding(genereads, df['TSS_start'][i], df['TSS_end'][i], df['Strand'][i])
        print('window sliding Time elapsed', int(time.time() - start_time), 'seconds.')
        with open(ctssOutPath, 'wb') as f:
            pickle.dump(allsortfddict, f)

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



    def window_sliding(self,genereads,TSS_start,TSS_end,strand):
        # --mode CTSS window (unchanged): every base from TSS_start to TSS_end - 1 is a position, including bases with
        # no read (count 0), so a window is windowSize consecutive BASES. Note: range(TSS_start, TSS_end) stops one base
        # short of TSS_end, so reads on the cluster's last base are never considered.


        leftIndex=0
        filterls=[i for i in genereads if ('14S' in i[2]) or ('15S' in i[2]) or ('16S' in i[2])  ]
        #print("the pid of current process：",os.getpid())

        #calculate the TSS position and corresponding counts
        promoterTSS = [read[0] for read in filterls if (read[0] >= TSS_start) and (read[0] <= TSS_end)]
        TSS,count=np.unique(promoterTSS,return_counts=True)



        originaldf=pd.DataFrame({'count':count},index=TSS)
        desireddf=pd.DataFrame({'count':0},index=[i for i in range(int(TSS_start),int(TSS_end))])
        desireddf['count']=desireddf.index.map(originaldf['count']).fillna(0)
        desireddf.reset_index(inplace=True)
        nonzeroarray=np.array(desireddf)


        if strand=='+':
            sortfinalarray=nonzeroarray[nonzeroarray[:, 0].argsort()]
        elif strand=='-':
            sortfinalarray=nonzeroarray[nonzeroarray[:, 0].argsort()[::-1]]

        TSS=sortfinalarray.T[0]
        count=sortfinalarray.T[1]


        #do something with sliding windows algorithm
        storels=[]
        for i in range(len(TSS) - self.windowSize + 1):
            #print(i)
            onewindow=TSS[i: i + self.windowSize]
            correspondingcount=count[i: i + self.windowSize]
            middlecount=correspondingcount[leftIndex]
            foldchange=(middlecount+1)/(sum(correspondingcount)/len(correspondingcount)+1)
            storels.append([onewindow[leftIndex],correspondingcount[leftIndex],foldchange])


        foldchangels=[i[2] for i in storels]
        sortindex=sorted(range(len(foldchangels)), key=lambda k: foldchangels[k],reverse=True)
        allsortls=[storels[i] for i in sortindex]

        return allsortls




    def produce_CTSS_adata(self):
        self.ctss_out_dir=os.path.join(self.out_dir+'/CTSS/')
        if not os.path.exists(self.ctss_out_dir):
            os.mkdir(self.ctss_out_dir)
        clusters = load_kept_cluster_reads(os.path.join(self.out_dir, 'count'))
        return build_ctss_adata(clusters, self.alloneclusterdf, self.generefdf, self.window_sliding,
                                self.minCTSSCount, self.minFC, self.ctss_out_dir)
