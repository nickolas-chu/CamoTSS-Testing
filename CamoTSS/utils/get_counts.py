import numpy as np
import pandas as pd
import os
import pickle
from functools import reduce
import anndata as ad
import multiprocessing 
import pysam
from scipy.optimize import linear_sum_assignment
from scipy import sparse as sp
import time
import random
import statistics
import math
import editdistance
import warnings
from pathlib import Path
from .toolbox import check_pysam_chrom,fetch_reads
from .distinct_clustering import cluster_distinct_positions
from .read_store import (ReadStore, ReadStoreWriter, store_dir_of, has_store,
                         convert_fetch_reads_pickle)
from multiprocessing import get_context
import logging
import signal
import sys
from sklearn.neighbors import NearestNeighbors
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed
from collections import defaultdict, Counter
class TimeoutException(Exception):
    pass






warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=Warning)

def build_transcript_column_from_path(path, final_index):
    import pickle
    import numpy as np
    import pandas as pd
    import os
    import logging

    try:
        with open(path, 'rb') as f:
            extend_entry = pickle.load(f)

        transcriptid = extend_entry[0]
        cellID, count = np.unique(extend_entry[1][1], return_counts=True)
        transcriptdf = pd.DataFrame({'cell_id': cellID, transcriptid: count})
        transcriptdf.set_index('cell_id', inplace=True)
        result = transcriptid, final_index.map(transcriptdf[transcriptid]).fillna(0)

        logging.warning(f"[SCLEVEL] Processed {transcriptid} from {os.path.basename(path)}")
        return result

    except Exception as e:
        logging.error(f"[SCLEVEL] Failed to process {path}: {type(e).__name__}: {e}")
        return None

def get_fastq_file(fastqFilePath):
    fastqFile=pysam.FastaFile(fastqFilePath)
    return fastqFile

    


_FASTA_CACHE = {}


def _cached_fasta(path):
    """one open FASTA per worker process (was reopened for every gene)"""
    if path not in _FASTA_CACHE:
        _FASTA_CACHE[path] = get_fastq_file(path)
    return _FASTA_CACHE[path]


def _keep_read1(r, geneid, row, cellBarcode, fastqFile):
    """The per-read filters of fetch_gene_reads() for one read 1, in the same order; returns its (TSS position,
    barcode, CIGAR) if it passes, else None."""
    if r.get_tag('GX') != geneid:
        return None
    if r.get_tag('CB') not in cellBarcode:
        return None
    chrom = 'chr' + str(row['Chromosome'])
    if row['Strand'] == '+':
        if not editdistance.eval(fastqFile.fetch(start=r.reference_start-14, end=r.reference_start-1, region=chrom), 'TTTCTTATATGGG') > 3:
            return None
        if r.is_reverse != False:
            return None
        if not editdistance.eval(r.query_sequence[9:14], 'ATGGG') <= 4:
            return None
        if not len(r.cigartuples) >= 2:
            return None
        if not ((r.cigartuples[0][0] == 4) & (6 < r.cigartuples[0][1] < 20) & (r.cigartuples[1][0] == 0) & (r.cigartuples[1][1] > 5)):
            return None
        return (r.reference_start, r.get_tag('CB'), r.cigarstring)
    elif row['Strand'] == '-':
        if not editdistance.eval(fastqFile.fetch(start=r.reference_end, end=r.reference_end+13, region=chrom), 'CCCATATAAGAAA') > 3:
            return None
        if r.is_reverse != True:
            return None
        if not editdistance.eval(r.query_sequence[-13:-8], 'CCCAT') <= 4:
            return None
        if not len(r.cigartuples) >= 2:
            return None
        if not ((r.cigartuples[-2][0] == 0) & (r.cigartuples[-2][1] > 5) & (r.cigartuples[-1][0] == 4) & (6 < r.cigartuples[-1][1] < 20)):
            return None
        return (r.reference_end, r.get_tag('CB'), r.cigarstring)
    return None


def fetch_gene_reads_streaming(bamfilePath, fastqFilePath, geneid, row, cellBarcode):
    """Same result as fetch_gene_reads() (same reads, same order), without holding the gene's BAM records in memory.

    fetch_gene_reads() first collects every BAM record of the gene region (toolbox.fetch_reads: read 1 and read 2,
    with sequences, qualities and tags -- 17.7 GB for human MALAT1), pairs read 1 with read 2 by read name, and only
    then filters. Here every record is filtered as it is read and only small things are kept: each read 1's name and,
    if it passes all filters, its (position, barcode, CIGAR); each read 2's name. The pairing is then done on the names
    with exactly the same steps as toolbox.fetch_reads (duplicate skip, region/length filters, the read-name suffix
    check, the sorted-name merge), so the kept reads and their order are identical.
    Defaults of toolbox.fetch_reads as called by CamoTSS: rm_duplicate=True, inner_only=True, mapq_min=0,
    trimLen_max=100, rlen_min=1, is_mated=True.
    """
    samfile, chrom = check_pysam_chrom(bamfilePath, str(row['Chromosome']))
    start, end = row['Start'], row['End']
    fastqFile = _cached_fasta(fastqFilePath)
    # --- toolbox.fetch_reads part 1: chromosome name and the region's records ---
    chrom = str(chrom)
    if chrom not in samfile.references:
        chrom_parts = chrom.split("chr")
        chrom = chrom_parts[0] if len(chrom_parts) <= 1 else chrom_parts[1]
    try:
        reads = samfile.fetch(chrom, start, end)
    except ValueError:
        reads = []
        print("Cannot fetch reads in region: %s:%d-%d" % (chrom, start, end))
    except AssertionError:
        reads = []
        print("AssertionError in region: %s:%d-%d" % (chrom, start, end))
        print(" - Check that your BAM file is indexed!")
    # --- part 2: same filters, streaming; keep names (+ the CamoTSS tuple of passing read 1s) ---
    qname1, keep1, qname2 = [], [], []
    r_prev = None
    for r in reads:
        if (r_prev is not None and r_prev.qname == r.qname and r_prev.positions == r.positions):
            r_prev = r; continue                           # filter 4: only keep the first one of duplicates
        r_prev = r
        if (r.pos is None or r.pos < start or r.aend is None or r.aend > end): continue   # filter 1
        if r.mapq < 0: continue                                                            # filter 2
        if r.rlen - len(r.positions) > 100: continue                                       # filter 3
        if len(r.positions) < 1: continue                                                  # filter 5
        if r.is_read2:
            qname2.append(r.qname)
        else:
            qname1.append(r.qname)
            keep1.append(_keep_read1(r, geneid, row, cellBarcode, fastqFile))
    # --- part 2.1: the read-name suffix check (as written in toolbox.fetch_reads) ---
    FLAG = True
    if len(qname1) > 0:
        for i in range(len(qname1) - 1):
            if qname1[i][-1] != qname1[i + 1][-1]:
                FLAG = False
                break
    if FLAG:
        qname1 = [q[:-1] for q in qname1]
        qname2 = [q[:-1] for q in qname2]
    # --- part 3: pair read 1 with read 2 by sorted name; keep paired read 1s in that order ---
    idx1 = sorted(range(len(qname1)), key=qname1.__getitem__)
    idx2 = sorted(range(len(qname2)), key=qname2.__getitem__)
    out = []
    i1, i2 = 0, 0
    while i1 < len(idx1) and i2 < len(idx2):
        a, b = qname1[idx1[i1]], qname2[idx2[i2]]
        if a == b:
            t = keep1[idx1[i1]]
            if t is not None:
                out.append(t)
            i1, i2 = i1 + 1, i2 + 1
        elif a < b:
            i1 += 1
        else:
            i2 += 1
    return out


def fetch_gene_reads(bamfilePath, fastqFilePath, geneid, row, cellBarcode):
    """The reads CamoTSS keeps for one gene: [(TSS position, cell barcode, CIGAR), ...].

    Same filters, same order as get_TSS_count._getreads before (moved here so worker processes do not need the
    whole get_TSS_count object). row = the gene's row of the gene table (Chromosome, Start, End, Strand);
    cellBarcode = set of cell barcodes to keep.
    """
    #fetch reads1 in gene 
    samFile, _chrom = check_pysam_chrom(bamfilePath, str(row['Chromosome']))

    reads = fetch_reads(samFile, _chrom,  row['Start'] , row['End'],  trimLen_max=100)
    reads1_umi = reads["reads1"]



    #select according to GX tag and CB (filter according to user owned cell)
    reads1_umi=[r for r in reads1_umi if r.get_tag('GX')==geneid]
    # print("first")
    # print(reads1_umi)
    reads1_umi=[r for r in reads1_umi if r.get_tag('CB') in cellBarcode]
    # print("second")
    # print(reads1_umi)


    #filter strand invasion
    fastqFile=_cached_fasta(fastqFilePath)
    if row['Strand']=='+':
        reads1_umi=[r for r in reads1_umi if editdistance.eval(fastqFile.fetch(start=r.reference_start-14, end=r.reference_start-1, region='chr'+str(row['Chromosome'])),'TTTCTTATATGGG') >3 ]
    elif row['Strand']=='-':
        reads1_umi=[r for r in reads1_umi if editdistance.eval(fastqFile.fetch(start=r.reference_end, end=r.reference_end+13, region='chr'+str(row['Chromosome'])),'CCCATATAAGAAA') >3 ]

    #print(reads1_umi)
    reads_info=[]
    #filter according to the cateria of SCAFE
    if row['Strand']=='+':
        reads1_umi=[r for r in reads1_umi if r.is_reverse==False]

        reads1_umi=[r for r in reads1_umi if editdistance.eval(r.query_sequence[9:14],'ATGGG')<=4]
        reads1_umi=[r for r in reads1_umi if len(r.cigartuples)>=2]
        #print([i.cigarstring for i in reads1_umi])
        reads1_umi = [r for r in reads1_umi if (r.cigartuples[0][0] == 4) & (6 < r.cigartuples[0][1] < 20) & (r.cigartuples[1][0] == 0) & (r.cigartuples[1][1] > 5)]
        #print(reads1_umi)
        reads_info=[(r.reference_start,r.get_tag('CB'),r.cigarstring) for r in reads1_umi]

    elif row['Strand']=='-':
        reads1_umi=[r for r in reads1_umi if r.is_reverse==True]

        reads1_umi=[r for r in reads1_umi if editdistance.eval(r.query_sequence[-13:-8],'CCCAT')<=4]
        reads1_umi=[r for r in reads1_umi if len(r.cigartuples)>=2]
        #print([i.cigarstring for i in reads1_umi])
        reads1_umi = [r for r in reads1_umi if (r.cigartuples[-2][0] == 0) & (r.cigartuples[-2][1] > 5) & (r.cigartuples[-1][0] == 4) & (6 < r.cigartuples[-1][1] < 20)]
        #print(reads1_umi)
        reads_info=[(r.reference_end,r.get_tag('CB'),r.cigarstring) for r in reads1_umi]

    #print(reads_info)



    return reads_info


def annotate_gene_clusters(geneid, altTSSitemdict, temprefdf):
    """Assign each kept TSS cluster of one gene to a reference transcript and name it (unchanged code, moved here
    from get_TSS_count._do_anno_and_filter). altTSSitemdict = the gene's kept clusters, each
    [positions (n,1), second array, third array]; temprefdf = the gene's rows of ref_TSS.tsv.
    Returns {transcript_id: (positions, second array, third array)}.
    """
    #get gene ID
    #get list of TSS for gene ID
    #get rows of reference for gene

    #print(geneid)
    #print(altTSSdict)

    #preparing for cases where there are more TSS clusters than transcripts

    num_clusters = len(altTSSitemdict)
    num_transcripts = temprefdf.shape[0]

    if num_clusters > num_transcripts:
        # Determine the maximum number of TSS clusters that can be assigned to a single transcript
        max_clusters_per_transcript = num_clusters

        # Create duplicates of the transcripts
        duplicated_transcripts = pd.concat([temprefdf]*max_clusters_per_transcript, ignore_index=True)

        # Calculate cost matrix with duplicated transcripts
        cost_mtx = np.zeros((len(altTSSitemdict), len(duplicated_transcripts)))
        for i in range(len(altTSSitemdict)):
            for j in range(len(duplicated_transcripts)):
                cluster_val = altTSSitemdict[i][0]
                position, count = np.unique(cluster_val, return_counts=True)
                mode_position = position[np.argmax(count)]
                cost_mtx[i, j] = np.absolute(np.sum(mode_position - duplicated_transcripts.iloc[j, 5]))

        # Use linear_sum_assignment with duplicated transcripts
        row_ind, col_ind = linear_sum_assignment(cost_mtx)
        #creating a list of transcript IDs corresponding to the optimal assignment
        transcriptls=list(duplicated_transcripts.iloc[col_ind,:]['transcript_id'])
        # tssls[i] is the position of the ref TSS assigned to the transcript at index col_ind[i]
        tssls=list(duplicated_transcripts.iloc[col_ind,:]['TSS'])

    else:
        #use Hungarian algorithm to assign cluster to corresponding transcript
        # array filled with zeros. each row corresponds to a TSS, and each column corresponds to a transcript
        cost_mtx=np.zeros((len(altTSSitemdict),temprefdf.shape[0]))
        #For each TSS
        for i in range(len(altTSSitemdict)):
        #for each row in reference    
            for j in range(temprefdf.shape[0]):
                cluster_val=altTSSitemdict[i][0]

                #this cost matrix should be corrected
                #finding the unique values in cluster_val and their counts
                position,count=np.unique(cluster_val,return_counts=True)
                #gettting most frequent position
                mode_position=position[np.argmax(count)]
                # absolute difference between mode_position and all ref TSS positions
                #assigned value to corresponding position in the cost matrix
                cost_mtx[i,j]=np.absolute(np.sum(mode_position-temprefdf.iloc[j,5]))
        #2 arrays,assignment of the TSS at index row_ind[i] to the transcript at index col_ind[i]
        row_ind, col_ind = linear_sum_assignment(cost_mtx)

        #creating a list of transcript IDs corresponding to the optimal assignment
        #transcriptls[i] is the ID of the transcript assigned to the TSS at index row_ind[i]
        transcriptls=list(temprefdf.iloc[col_ind,:]['transcript_id'])
        # tssls[i] is the position of the ref TSS assigned to the transcript at index col_ind[i]
        tssls=list(temprefdf.iloc[col_ind,:]['TSS'])

    transcriptdict={}
    #intermediate_dicts = []
    # initialize the counters
    count1 = 0
    count2 = 0

    #Counters do not reset for when switching to the next transcript ID
    for i in range(0,len(tssls)):
        #setting the minimum and maximum read positions for a cluster to be the bounds
        #check if the reference tss positions that the tss cluster was assigned to
        #falls within it. Name based on results
        if (tssls[i]>=np.min(altTSSitemdict[i][0])) & (tssls[i]<=np.max(altTSSitemdict[i][0])):
            # increment count1
            count1 += 1
            #name1=str(geneid)+'_'+str(transcriptls[i])
            name1=f"{geneid}_{transcriptls[i]}_{count1}"
            transcriptdict[name1]=(altTSSitemdict[row_ind[i]][0],altTSSitemdict[row_ind[i]][1],altTSSitemdict[row_ind[i]][2])
        else:
            # increment count2
            count2 += 1
            #newname1=str(geneid)+'_newTSS'
            newname1=f"{geneid}_newTSS_{count2}" 
            transcriptdict[newname1]=(altTSSitemdict[row_ind[i]][0],altTSSitemdict[row_ind[i]][1],altTSSitemdict[row_ind[i]][2])

        #intermediate_dicts.append(transcriptdict.copy())
    #if len(altTSSitemdict) > 1:
        #with open(f'intermediate_dicts_{geneid}.pkl', 'wb') as f:
            #pickle.dump(intermediate_dicts, f)

    return transcriptdict


# ======================================================================================================================
# LOW-MEMORY, ONE-GENE-AT-A-TIME PIPELINE (2026-11-23)
#
# Before: the main process held every read of every gene (fetch_reads.pkl in memory), then every clustered read
# (before_cluster_peak.pkl, keepdict.pkl, extendls.pkl), and each gene was sent to a worker as a bound method, which
# copied the whole get_TSS_count object (reference tables, cell list; 12.5 MB for human) with EVERY gene.
# Now: reads live in a compact on-disk store (read_store.py); workers get only a gene id, read that gene from disk,
# and return small results. The algorithms and their outputs are unchanged:
#   phase 1 (per gene): all-reads clustering -> clusters with >= minCount reads -> CamoTSS's 4 features
#   main:               fourFeature.csv, CamoTSS's logistic classifier -> afterfiltered.csv          (same code as before)
#   phase 2 (per gene): transcript assignment of the kept clusters (same code as before) + counts per cell
#   main:               regiondf.csv, scTSS_count_all.h5ad, scTSS_count_two.h5ad                    (same code as before)
# ======================================================================================================================

_W = {}            # per-worker context, filled once per worker process by _init_worker (not sent with every gene)


def _init_worker(ctx):
    _W.clear()
    _W.update(ctx)
    if 'store_dir' in ctx:
        _W['store'] = ReadStore(ctx['store_dir'])


def _fetch_worker(geneid):
    """BAM step for one gene; returns (geneid, reads) or (geneid, error)."""
    try:
        return geneid, fetch_gene_reads_streaming(_W['bam'], _W['fasta'], geneid, _W['generows'][geneid],
                                                  _W['cellBarcode']), None
    except Exception as e:
        return geneid, None, f"{type(e).__name__}: {e}"


def _exact_stdev(pos):
    """statistics.stdev(positions) exactly as the previous code computed it.

    The previous code called statistics.stdev on a numpy int64 array; Python's statistics module then converts the
    variance back to the data's integer type, i.e. ROUNDS THE VARIANCE DOWN to a whole number before the square root.
    Reproduced here with exact integer arithmetic (positions shifted to the cluster's minimum so sums stay small).
    """
    n = len(pos)
    if n < 2:
        return statistics.stdev(pos)                      # raises StatisticsError exactly as before
    d = (pos - pos.min()).astype(np.int64)
    span = int(d.max())
    if n * span * span < 2 ** 62:                         # int64 sums are exact here
        s1 = int(d.sum(dtype=np.int64)); s2 = int((d * d).sum(dtype=np.int64))
    else:                                                 # very wide / very large cluster: python ints, exact
        d = d.astype(object); s1 = int(d.sum()); s2 = int((d * d).sum())
    return math.sqrt((n * s2 - s1 * s1) // (n * (n - 1)))


def _cluster_features_worker(geneid):
    """Phase 1 for one gene: clusters with >= minCount reads and CamoTSS's 4 features for each.

    Returns (geneid, rows, error); rows = [(clustername, UMI_count, SD, summit_UMI_count, unencoded_G_percent,
    NO.TSS, gene_id, summit_position), ...] in the gene's cluster order (left to right), as _filter_false_positive
    built them.
    """
    def handler(signum, frame):
        raise TimeoutException("Clustering timed out")
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(600)
    try:
        store = _W['store']
        pos, cb, cig = store.get(geneid)
        uniq, counts, labels = cluster_distinct_positions(pos, _W['InnerDistance'])
        n_reads = np.bincount(labels, weights=counts).astype(np.int64)
        keep = np.flatnonzero(n_reads >= _W['minCount'])             # clusters with >= minCount reads, left to right
        read_label = labels[np.searchsorted(uniq, pos)]
        order = np.argsort(read_label, kind='stable')                # reads grouped by cluster, read order kept inside
        first = np.r_[0, np.cumsum(n_reads)]
        rows = []
        for j, c in enumerate(keep):
            m = order[first[c]:first[c + 1]]                         # the cluster's reads = all reads inside its bounds
            p = pos[m]
            startpos, stoppos = np.min(p), np.max(p)
            count = len(p)
            std = _exact_stdev(p)
            tempposi, tempposicount = np.unique(p, return_counts=True)
            summit_count = np.max(tempposicount)
            summitpos = tempposi[np.argmax(tempposicount)]
            unencoded_G_percent = int(store.cigar_g[cig[m]].sum()) / count
            rows.append((str(geneid) + '*' + str(startpos) + '_' + str(stoppos), count, std, summit_count,
                         unencoded_G_percent, j, geneid, summitpos))
        signal.alarm(0)
        return geneid, rows, None
    except Exception as e:
        signal.alarm(0)
        return geneid, None, f"{type(e).__name__}: {e}"


def _tss_counts_worker(args):
    """Phase 2 for one gene: name the kept clusters (transcript assignment, unchanged) and count reads per cell.

    args = (geneid, [(start, end), ...] of the gene's kept clusters in afterfiltered.csv order)
    Returns (geneid, [(transcript_id, TSS_start, TSS_end, cell codes, counts), ...], error).
    """
    geneid, bounds = args
    try:
        store = _W['store']
        pos, cb, _ = store.get(geneid)
        altTSSitemdict = []
        for lo, hi in bounds:
            m = (pos >= lo) & (pos <= hi)                            # = the kept cluster's reads, in read order
            altTSSitemdict.append([pos[m].reshape(-1, 1), cb[m].reshape(-1, 1), None])
        transcriptdict = annotate_gene_clusters(geneid, altTSSitemdict, _W['tssref'].get(geneid, _W['tssref_empty']))
        out = []
        for name, (p, c, _) in transcriptdict.items():
            cells, n = np.unique(c.ravel(), return_counts=True)
            out.append((name, np.min(p), np.max(p), cells.astype(np.int32), n.astype(np.int64)))
        return geneid, out, None
    except Exception as e:
        return geneid, None, f"{type(e).__name__}: {e}"


def _append_checkpoint(path, obj):
    with open(path, 'ab') as f:
        pickle.dump(obj, f)


def _read_checkpoint(path):
    out = []
    if os.path.exists(path):
        with open(path, 'rb') as f:
            while True:
                try:
                    out += pickle.load(f)
                except EOFError:
                    break
                except Exception:                        # a batch cut short by a crash: keep what was complete
                    break
    return out


def _run_genes(nproc, ctx, worker, items, label, checkpoint, key=lambda x: x):
    """Run worker over items in a pool whose workers get ctx ONCE; resume from / append to checkpoint.

    Returns {gene: result}. Genes that fail are retried once, one at a time, in this process; if any still fails the
    run stops (as before).
    """
    done = {g: r for g, r in _read_checkpoint(checkpoint)}
    todo = [it for it in items if key(it) not in done]
    if done:
        logging.warning(f"[{label}] resuming: {len(done)} genes already done, {len(todo)} to go")
    failed = []
    t0 = time.time(); batch = []
    if todo:
        with get_context('fork').Pool(nproc, initializer=_init_worker, initargs=(ctx,)) as pool:
            for g, res, err in pool.imap_unordered(worker, todo, chunksize=1):
                if err is not None:
                    logging.warning(f"[{label}] gene {g} failed: {err}")
                    failed.append(g)
                    continue
                done[g] = res; batch.append((g, res))
                if len(batch) >= 500:
                    _append_checkpoint(checkpoint, batch); batch = []
                    logging.warning(f"[{label}] {len(done)} genes done ({int(time.time() - t0)} s)")
        if batch:
            _append_checkpoint(checkpoint, batch)
    if failed:
        logging.warning(f"[{label}] retrying {len(failed)} failed genes one at a time")
        _init_worker(ctx)
        by_key = {key(it): it for it in todo}
        still = []
        for g in failed:
            g2, res, err = worker(by_key[g])
            if err is not None:
                logging.error(f"[{label}] gene {g} failed again: {err}")
                still.append(g)
            else:
                done[g] = res
                _append_checkpoint(checkpoint, [(g, res)])
        if still:
            with open(os.path.join(os.path.dirname(checkpoint), 'failed_genes.txt'), 'w') as f:
                f.write('\n'.join(still))
            print("Stopped: genes failed twice. See log.txt and failed_genes.txt.")
            sys.exit(1)
    logging.warning(f"[{label}] finished {len(done)} genes in {int(time.time() - t0)} s")
    return done


class get_TSS_count():



    def __init__(self,generefPath,tssrefPath,bamfilePath,fastqFilePath,outdir,cellBarcodePath,nproc,minCount,maxReadCount,clusterDistance,InnerDistance,windowSize,minCTSSCount,minFC):
        self.generefdf=pd.read_csv(generefPath,delimiter='\t')
        #self.generefdf.set_index('gene_id',inplace=True)
        self.generefdf['len']=self.generefdf['End']-self.generefdf['Start']
        self.tssrefdf=pd.read_csv(tssrefPath,delimiter='\t')
        self.bamfilePath=bamfilePath
        self.outdir=outdir
        self.count_out_dir=str(outdir)+'/count/'
        if not os.path.exists(self.count_out_dir):
            os.mkdir(self.count_out_dir)




        self.minCount=minCount
        # a set: checking a read's barcode is then one lookup, not a scan of the whole cell list (same answer)
        self.cellBarcode=set(pd.read_csv(cellBarcodePath,delimiter='\t')['cell_id'].values)
        self.nproc=nproc
        self.maxReadCount=maxReadCount
        self.clusterDistance=clusterDistance
        self.fastqFilePath=fastqFilePath
        self.InnerDistance=InnerDistance
        self.windowSize=windowSize
        self.minCTSSCount=minCTSSCount
        self.minFC=minFC



    def _getreads(self,bamfilePath,fastqFilePath,geneid,mergedf):
        # kept for anyone calling it directly; the pipeline uses fetch_gene_reads() in worker processes
        return fetch_gene_reads(bamfilePath, fastqFilePath, geneid, mergedf.loc[geneid], self.cellBarcode)



    def _get_read_store(self):
        """The gene reads, as a compact on-disk store (read_store.py). Built once, then reused on every rerun.

        - store already complete (count/fetch_reads_store/COMPLETE): reuse it
        - only an older fetch_reads.pkl present: convert it once, streaming (memory ~ one gene)
        - neither: read the BAM (unchanged filters), writing each gene to the store as soon as it is done
        """
        store_dir = store_dir_of(self.count_out_dir)
        if has_store(self.count_out_dir):
            print("Resuming from existing fetch_reads_store...")
            return store_dir
        pkl = self.count_out_dir + 'fetch_reads.pkl'
        if os.path.exists(pkl):
            print("Converting existing fetch_reads.pkl to fetch_reads_store (one time)...")
            t0 = time.time()
            convert_fetch_reads_pickle(pkl, store_dir)
            logging.warning(f"[STORE] converted fetch_reads.pkl in {int(time.time() - t0)} s")
            return store_dir
        self._get_gene_reads(store_dir)
        return store_dir



    def _get_gene_reads(self, store_dir):
        """Read every gene's reads from the BAM (same filters as before) straight into the read store."""
        t0 = time.time()
        getreadsFile=pysam.AlignmentFile(self.bamfilePath,'rb')
        # gene ids in order of first appearance in the BAM (as drop_duplicates gave), without a list of every read
        seen = {}
        for read in getreadsFile.fetch(until_eof = True):
            seen.setdefault(read.get_tag('GX'), None)
        geneid_uniqdf=pd.DataFrame(list(seen),columns=['gene_id'])
        mergedf=geneid_uniqdf.merge(self.generefdf,on='gene_id')
        mergedf.set_index('gene_id',inplace=True)
        generows = {g: r for g, r in mergedf.iterrows()}
        order = {g: i for i, g in enumerate(mergedf.index)}
        logging.warning(f"[FETCH] {len(generows)} genes; reading them with {self.nproc} processes")

        ctx = {'bam': self.bamfilePath, 'fasta': self.fastqFilePath, 'generows': generows, 'cellBarcode': self.cellBarcode}
        w = ReadStoreWriter(store_dir)
        n_done = 0
        with get_context('fork').Pool(self.nproc, initializer=_init_worker, initargs=(ctx,)) as pool:
            for geneid, resls, err in pool.imap_unordered(_fetch_worker, list(mergedf.index), chunksize=4):
                if err is not None:
                    raise RuntimeError(f"reading gene {geneid} from the BAM failed: {err}")
                # same post-processing as before: sample down to maxReadCount, drop genes with < 2 reads
                if len(resls) > self.maxReadCount:
                    resls = random.sample(resls, self.maxReadCount)
                if len(resls) >= 2:
                    w.add_gene(geneid, resls)
                n_done += 1
                if n_done % 2000 == 0:
                    logging.warning(f"[FETCH] {n_done} genes read ({int(time.time() - t0)} s)")
        # keep genes in the BAM's first-appearance order, as the old fetch_reads.pkl dict had them
        w.genes.sort(key=lambda t: order[t[0]])
        w.close()
        print('Hello, we finished to get the reads')
        logging.warning(f"[FETCH] read store written in {int(time.time() - t0)} s")



    def _filter_false_positive(self):
        """Clustering (phase 1, per gene) + CamoTSS's classifier. Writes fourFeature.csv and afterfiltered.csv.

        Returns afterfiltereddf. Resumes: if both CSVs exist they are read back instead of recomputed.
        """
        ff_path = self.count_out_dir+'fourFeature.csv'
        af_path = self.count_out_dir+'afterfiltered.csv'
        if os.path.exists(ff_path) and os.path.exists(af_path):
            logging.warning("[RECOVERY] Found fourFeature.csv and afterfiltered.csv -- resuming from them.")
            return pd.read_csv(af_path, index_col=0)

        store_dir = self._get_read_store()
        store = ReadStore(store_dir)
        genes = sorted(store.genes, key=lambda g: -store.n_reads(g))       # biggest first, for load balance
        print(f"Clustering {len(genes)} genes with {self.nproc} processes...")
        start_time = time.time()
        ctx = {'store_dir': store_dir, 'InnerDistance': self.InnerDistance, 'minCount': self.minCount}
        res = _run_genes(self.nproc, ctx, _cluster_features_worker, genes, 'CLUSTER',
                         self.count_out_dir + 'cluster_features.partial.pkl')
        logging.warning(f"do clustering Time elapsed, {int(time.time() - start_time)} seconds.")

        print("Filtering...")
        clusterdict={}
        for g in store.genes:                                              # genes in their original order
            for row in res.get(g, []):
                clusterdict[row[0]]=row[1:]
        fourfeaturedf=pd.DataFrame(clusterdict).T
        fourfeaturedf.columns=['UMI_count','SD','summit_UMI_count','unencoded_G_percent','NO.TSS','gene_id','summit_position']
        fourfeaturedf.to_csv(ff_path)

        print('one_gene_with_two_TSS_fourfeature : %i'%(len(fourfeaturedf)))
        test_X=fourfeaturedf.iloc[:,0:4]
        pathstr=str(Path(os.path.dirname(os.path.abspath(__file__))).parents[0])+'/model/logistic_4feature_model.sav'
        loaded_model = pickle.load(open(pathstr, 'rb'))
        test_Y=loaded_model.predict(test_X.values)

        #do filtering, the result of this step should be output as final h5ad file display at single cell level.
        afterfiltereddf=fourfeaturedf[test_Y==1]
        afterfiltereddf.columns=['UMI_count','SD','summit_UMI_count','unencoded_G_percent','NO.TSS','gene_id','summit_position']
        afterfiltereddf.to_csv(af_path)
        os.remove(self.count_out_dir + 'cluster_features.partial.pkl')
        print("Finished filtering.")
        return pd.read_csv(af_path, index_col=0)



    def _TSS_annotation(self):
        """Phase 2: transcript assignment of the kept clusters (unchanged code) + reads per cell for every kept TSS.

        Returns (entries, regiondf); entries = [(transcript_id, TSS_start, TSS_end, cell codes, counts), ...] in the
        order the old extendls had them (genes in afterfiltered.csv order, transcripts in assignment order).
        """
        start_time = time.time()
        afterfiltereddf = self._filter_false_positive()
        store_dir = store_dir_of(self.count_out_dir)
        bounds = {}
        for name in afterfiltereddf.index:                                 # afterfiltered.csv order
            g, se = name.split('*')
            s, e = se.split('_')
            bounds.setdefault(g, []).append((int(s), int(e)))
        tssref = {g: d for g, d in self.tssrefdf.groupby('gene_id', sort=False)}
        ctx = {'store_dir': store_dir, 'tssref': tssref, 'tssref_empty': self.tssrefdf.iloc[0:0]}
        res = _run_genes(self.nproc, ctx, _tss_counts_worker, list(bounds.items()), 'ANNOTATE',
                         self.count_out_dir + 'tss_counts.partial.pkl', key=lambda it: it[0])
        entries = [t for g in bounds for t in res[g]]
        regiondf = pd.DataFrame({'transcript_id': [t[0] for t in entries],
                                 'TSS_start': [t[1] for t in entries],
                                 'TSS_end': [t[2] for t in entries]})
        print('do annotation Time elapsed', int(time.time() - start_time), 'seconds.')
        regiondf.to_csv(os.path.join(self.count_out_dir, 'regiondf.csv'))
        return entries, regiondf



    def produce_sclevel(self):
        ctime = time.time()
        entries, regiondf = self._TSS_annotation()
        store = ReadStore(store_dir_of(self.count_out_dir))

        # cells = every cell with a read in some kept TSS (as before; now in a fixed order instead of random set order)
        cell_codes = np.unique(np.concatenate([t[3] for t in entries])) if entries else np.zeros(0, np.int32)
        final_index = pd.Index(store.barcodes[cell_codes].astype(str))
        row_of = np.full(len(store.barcodes), -1, dtype=np.int64); row_of[cell_codes] = np.arange(len(cell_codes))
        n_cells = len(final_index)

        # Build the cell x transcript count matrix directly as a sparse (CSR) matrix (numbers as before).
        logging.warning(f"[SCLEVEL] Building sparse transcript matrix for {n_cells} cells x {len(entries)} transcripts.")
        rows = [row_of[t[3]] for t in entries]
        cols = [np.full(len(t[3]), j, dtype=np.int64) for j, t in enumerate(entries)]
        data = [t[4].astype(np.float32) for t in entries]
        transcript_ids = [t[0] for t in entries]
        rows = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
        cols = np.concatenate(cols) if cols else np.zeros(0, dtype=np.int64)
        data = np.concatenate(data) if data else np.zeros(0, dtype=np.float32)
        X = sp.coo_matrix((data, (rows, cols)), shape=(n_cells, len(transcript_ids)), dtype=np.float32).tocsr()

        adata = ad.AnnData(
            X=X,
            obs=pd.DataFrame(index=final_index),
            var=pd.DataFrame(index=pd.Index(transcript_ids)),
        )
        logging.warning(f"[SCLEVEL] Finished sparse transcript matrix with {adata.n_vars} columns, {X.nnz} nonzeros.")

        vardf = pd.DataFrame(adata.var.copy())
        vardf.reset_index(inplace=True)
        vardf.columns = ['transcript_id']
        vardf = vardf.join(regiondf.set_index('transcript_id'), on='transcript_id')
        vardf['gene_id'] = vardf['transcript_id'].str.split('_', expand=True)[0]
        vardf = vardf.merge(self.generefdf, on='gene_id')
        vardf.set_index('transcript_id', drop=True, inplace=True)
        adata.var = vardf

        sc_output_h5ad = os.path.join(self.count_out_dir, 'scTSS_count_all.h5ad')
        adata.write(sc_output_h5ad)

        # Filter by cluster distance
        newdf = adata.var.copy()
        newdf.reset_index(inplace=True)
        selectedf = newdf[newdf.duplicated('gene_id', keep=False)]
        geneID = selectedf['gene_id'].unique()

        keepdfls = []
        for i in geneID:
            tempdf = selectedf[selectedf['gene_id'] == i]
            tempdf = tempdf.sort_values('transcript_id', ascending=False)
            tempdf['diff'] = tempdf['TSS_start'].diff()
            keepdf = tempdf[tempdf['diff'].isna() | tempdf['diff'].abs().ge(self.clusterDistance)]
            keepdfls.append(keepdf)

        allkeepdf = reduce(lambda x, y: pd.concat([x, y]), keepdfls)
        finaltwodf = allkeepdf[allkeepdf.duplicated('gene_id', keep=False)]
        finaltwoadata = adata[:, adata.var.index.isin(finaltwodf['transcript_id'])]

        sc_output_h5ad = os.path.join(self.count_out_dir, 'scTSS_count_two.h5ad')
        finaltwoadata.write(sc_output_h5ad)
        if os.path.exists(self.count_out_dir + 'tss_counts.partial.pkl'):
            os.remove(self.count_out_dir + 'tss_counts.partial.pkl')

        print('produce h5ad Time elapsed', int(time.time() - ctime), 'seconds.')
        return adata



    def produce_CTSS_adata(self):
        """--mode TC+CTSS: CTSS inside every kept TSS cluster.

        Same code as --mode CTSS (utils/get_ctss.py build_ctss_adata + window_sliding): one sliding window for both
        modes (windowSize consecutive bases, cluster's last base included); each kept cluster's reads are read from
        the read store on demand; cells per CTSS found with numpy; sparse matrix.
        """
        from .get_ctss import cluster_reads_source, build_ctss_adata
        self.ctss_out_dir=str(self.outdir)+'/CTSS/'
        if not os.path.exists(self.ctss_out_dir):
            os.mkdir(self.ctss_out_dir)
        clusters = cluster_reads_source(self.count_out_dir)
        alloneclusterdf = pd.read_csv(self.count_out_dir+'afterfiltered.csv')
        return build_ctss_adata(clusters, alloneclusterdf, self.generefdf, self.windowSize,
                                self.minCTSSCount, self.minFC, self.ctss_out_dir)
