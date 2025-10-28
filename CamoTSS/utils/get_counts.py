import numpy as np
import pandas as pd
import os
import pickle
from functools import reduce
import anndata as ad
import multiprocessing 
import pysam
from sklearn.cluster import AgglomerativeClustering
from scipy.optimize import linear_sum_assignment
import time
import random
import statistics
import editdistance
import warnings
from pathlib import Path
from .toolbox import check_pysam_chrom,fetch_reads
from multiprocessing import get_context
import logging
import signal
import sys
from sklearn.neighbors import NearestNeighbors
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed
from collections import defaultdict
class TimeoutException(Exception):
    pass






warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=Warning)





def get_fastq_file(fastqFilePath):
    fastqFile=pysam.FastaFile(fastqFilePath)
    return fastqFile

    
def _assign_leftover_chunk_bounded(chunk, cluster_bounds, cluster_centroids):
    results = []
    for read in chunk:
        read_pos = read[0]
        eligible = [
            (idx, abs(read_pos - cluster_centroids[idx]))
            for idx, (min_, max_) in enumerate(cluster_bounds)
            if min_ <= read_pos <= max_
        ]
        if eligible:
            best_cluster = min(eligible, key=lambda x: x[1])[0]
            results.append((read, best_cluster))
    return results


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
        self.cellBarcode=pd.read_csv(cellBarcodePath,delimiter='\t')['cell_id'].values
        self.nproc=nproc
        self.maxReadCount=maxReadCount
        self.clusterDistance=clusterDistance
        self.fastqFilePath=fastqFilePath
        self.InnerDistance=InnerDistance
        self.windowSize=windowSize
        self.minCTSSCount=minCTSSCount
        self.minFC=minFC

        

    def _getreads(self,bamfilePath,fastqFilePath,geneid,mergedf):
        #print(self.generefdf)
        #fetch reads1 in gene 
        samFile, _chrom = check_pysam_chrom(bamfilePath, str(mergedf.loc[geneid]['Chromosome']))
        
        reads = fetch_reads(samFile, _chrom,  mergedf.loc[geneid]['Start'] , mergedf.loc[geneid]['End'],  trimLen_max=100)
        reads1_umi = reads["reads1"]



        #select according to GX tag and CB (filter according to user owned cell)
        reads1_umi=[r for r in reads1_umi if r.get_tag('GX')==geneid]
        # print("first")
        # print(reads1_umi)
        reads1_umi=[r for r in reads1_umi if r.get_tag('CB') in self.cellBarcode]
        # print("second")
        # print(reads1_umi)


        #filter strand invasion
        fastqFile=get_fastq_file(fastqFilePath)
        if mergedf.loc[geneid]['Strand']=='+':
            reads1_umi=[r for r in reads1_umi if editdistance.eval(fastqFile.fetch(start=r.reference_start-14, end=r.reference_start-1, region='chr'+str(mergedf.loc[geneid]['Chromosome'])),'TTTCTTATATGGG') >3 ]
        elif mergedf.loc[geneid]['Strand']=='-':
            reads1_umi=[r for r in reads1_umi if editdistance.eval(fastqFile.fetch(start=r.reference_end, end=r.reference_end+13, region='chr'+str(mergedf.loc[geneid]['Chromosome'])),'CCCATATAAGAAA') >3 ]

        #print(reads1_umi)
        reads_info=[]
        #filter according to the cateria of SCAFE
        if mergedf.loc[geneid]['Strand']=='+':
            reads1_umi=[r for r in reads1_umi if r.is_reverse==False]
            
            reads1_umi=[r for r in reads1_umi if editdistance.eval(r.query_sequence[9:14],'ATGGG')<=4]
            reads1_umi=[r for r in reads1_umi if len(r.cigartuples)>=2]
            #print([i.cigarstring for i in reads1_umi])
            reads1_umi = [r for r in reads1_umi if (r.cigartuples[0][0] == 4) & (6 < r.cigartuples[0][1] < 20) & (r.cigartuples[1][0] == 0) & (r.cigartuples[1][1] > 5)]
            #print(reads1_umi)
            reads_info=[(r.reference_start,r.get_tag('CB'),r.cigarstring) for r in reads1_umi]
        
        elif mergedf.loc[geneid]['Strand']=='-':
            reads1_umi=[r for r in reads1_umi if r.is_reverse==True]
            
            reads1_umi=[r for r in reads1_umi if editdistance.eval(r.query_sequence[-13:-8],'CCCAT')<=4]
            reads1_umi=[r for r in reads1_umi if len(r.cigartuples)>=2]
            #print([i.cigarstring for i in reads1_umi])
            reads1_umi = [r for r in reads1_umi if (r.cigartuples[-2][0] == 0) & (r.cigartuples[-2][1] > 5) & (r.cigartuples[-1][0] == 4) & (6 < r.cigartuples[-1][1] < 20)]
            #print(reads1_umi)
            reads_info=[(r.reference_end,r.get_tag('CB'),r.cigarstring) for r in reads1_umi]

        #print(reads_info)


        
        return reads_info
    

    

        
    def _get_gene_reads(self):

        pool = multiprocessing.Pool(processes=self.nproc)


        bamfilePath=self.bamfilePath
        fastqFilePath=self.fastqFilePath


        getreadsFile=pysam.AlignmentFile(bamfilePath,'rb')

        geneidls=[]
        for read in getreadsFile.fetch(until_eof = True):
            geneid=read.get_tag('GX')
            geneidls.append(geneid)
        geneiddf=pd.DataFrame(geneidls,columns=['gene_id'])
        geneid_uniqdf=geneiddf.drop_duplicates('gene_id')

        mergedf=geneid_uniqdf.merge(self.generefdf,on='gene_id')
        mergedf.set_index('gene_id',inplace=True)

        # print(mergedf)
        # print(self.generefdf)



        readinfodict={}
        results=[]

        #get reads because pysam object cannot be used for multiprocessing so inputting bam file path 
        for i in mergedf.index:
            #print(i)
            results.append(pool.apply_async(self._getreads,(bamfilePath,fastqFilePath,i,mergedf)))
        pool.close()
        pool.join()
        results=[res.get() for res in results]

        print('Hello, we finished to get the reads')

        for geneid,resls in zip(mergedf.index,results):
            readinfodict[geneid]=resls  


        #delete gene whose reads length is larger than maxReadCount
        for i in list(readinfodict.keys()):
            if len(readinfodict[i])>self.maxReadCount:
                readinfodict[i]=random.sample(readinfodict[i],self.maxReadCount)
            if len(readinfodict[i])<2:
                del readinfodict[i] 

        #print('hello,we finish get readinfodict')
        #store reads fetched
        outfilename=self.count_out_dir+'fetch_reads.pkl'
        with open(outfilename,'wb') as f:
            pickle.dump(readinfodict,f)


        return readinfodict
    

    def _do_clustering(self, args):
        geneid, readinfo_full = args
        logging.warning(f"Starting clustering for {geneid}")
        MAX_CLUSTER_READS = 20000
    
        # Set a hard timeout
        def handler(signum, frame):
            raise TimeoutException("Clustering timed out")
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(600)
    
        try:
            downsampled = False
    
            # --- Downsampling logic ---
            if len(readinfo_full) > MAX_CLUSTER_READS:
                logging.warning(f"Downsampling gene {geneid} from {len(readinfo_full)} to {MAX_CLUSTER_READS} reads")
    
                # Step 1: Group reads by condition and replicate
                condition_groups = defaultdict(lambda: defaultdict(list))
                for read in readinfo_full:
                    barcode = read[1]
                    suffix = barcode.split('-')[-1] if '-' in barcode else 'UNKNOWN'
                    condition = suffix.split('_')[0]
                    condition_groups[condition][suffix].append(read)
    
                # Step 2: Determine sample size per condition
                num_conditions = len(condition_groups)
                reads_per_condition = MAX_CLUSTER_READS // max(num_conditions, 1)
    
                # Step 3: Sample evenly across replicates
                readinfo_sample = []
                for condition, replicates in condition_groups.items():
                    num_replicates = len(replicates)
                    reads_per_replicate = reads_per_condition // max(num_replicates, 1)
                    for replicate_id, reads in replicates.items():
                        if len(reads) <= reads_per_replicate:
                            readinfo_sample.extend(reads)
                            logging.warning(f"Gene {geneid}: using all {len(reads)} reads from replicate {replicate_id} (condition {condition})")
                        else:
                            sampled = np.random.choice(len(reads), reads_per_replicate, replace=False)
                            readinfo_sample.extend([reads[i] for i in sampled])
                            logging.warning(f"Gene {geneid}: sampled {reads_per_replicate} reads from replicate {replicate_id} (condition {condition})")
    
                # Step 4: Track leftovers
                sampled_set = set(tuple(r) for r in readinfo_sample)
                readinfo_leftover = [r for r in readinfo_full if tuple(r) not in sampled_set]
                downsampled = True
            else:
                readinfo_sample = readinfo_full
                readinfo_leftover = []
                downsampled = False
    
            # --- Clustering on sample ---
            posi_sample = np.array([t[0] for t in readinfo_sample]).reshape(-1, 1)
            CB_sample = np.array([t[1] for t in readinfo_sample]).reshape(-1, 1)
            cigar_sample = np.array([t[2] for t in readinfo_sample]).reshape(-1, 1)
    
            model = AgglomerativeClustering(n_clusters=None, linkage='average', distance_threshold=self.InnerDistance)
            labels_sample = model.fit(posi_sample).labels_
    
            # --- Build initial clusters ---
            altTSSls_raw = []
            for lbl in np.unique(labels_sample):
                altTSSls_raw.append([
                    posi_sample[labels_sample == lbl],
                    CB_sample[labels_sample == lbl],
                    cigar_sample[labels_sample == lbl]
                ])
    
            # --- Assign leftover reads to nearest cluster ---
            if downsampled and len(altTSSls_raw) > 0 and len(readinfo_leftover) > 0:
    
                # Compute cluster bounds
                cluster_bounds = []
                for cluster in altTSSls_raw:
                    positions = cluster[0].flatten()
                    cluster_min = np.min(positions)
                    cluster_max = np.max(positions)
                    cluster_bounds.append((cluster_min, cluster_max))
                
                # Prepare leftover reads
                filtered_reads = []
                for read in readinfo_leftover:
                    read_pos = read[0]
                    eligible_clusters = [
                        idx for idx, (min_, max_) in enumerate(cluster_bounds)
                        if min_ <= read_pos <= max_
                    ]
                    if eligible_clusters:
                        # Assign to the closest eligible cluster by centroid
                        centroid_dists = [
                            (idx, abs(read_pos - altTSSls_raw[idx][0].mean()))
                            for idx in eligible_clusters
                        ]
                        best_cluster = min(centroid_dists, key=lambda x: x[1])[0]
                        filtered_reads.append((read, best_cluster))
                
                # Assign reads
                for read, lbl_idx in filtered_reads:
                    altTSSls_raw[lbl_idx][0] = np.vstack((altTSSls_raw[lbl_idx][0], read[0]))
                    altTSSls_raw[lbl_idx][1] = np.vstack((altTSSls_raw[lbl_idx][1], read[1]))
                    altTSSls_raw[lbl_idx][2] = np.vstack((altTSSls_raw[lbl_idx][2], read[2]))

    
            # --- Filter clusters by minCount ---
            altTSSls = [c for c in altTSSls_raw if c[0].shape[0] >= self.minCount]
            signal.alarm(0)
            logging.warning(f"Gene {geneid}: {len(altTSSls_raw)} raw clusters → {len(altTSSls)} after minCount={self.minCount}")
            return (geneid, altTSSls)
    
        except TimeoutException:
            logging.error(f"TIMEOUT: Gene {geneid} exceeded 600s")
            return (geneid, None)
        except Exception as e:
            logging.error(f"FAILED: Gene {geneid} - {type(e).__name__}: {e}")
            return (geneid, None)

    # New helper: sample-only clustering (no inner Pool)
    def _do_clustering_sample(self, args):
        geneid, readinfo_full = args
        logging.warning(f"[SAMPLE] Starting sample clustering for {geneid} with {len(readinfo_full)} reads")
        try:
            MAX_CLUSTER_READS = 20000
            from collections import defaultdict
    
            # --- Downsampling logic (same as before) ---
            if len(readinfo_full) > MAX_CLUSTER_READS:
                condition_groups = defaultdict(lambda: defaultdict(list))
                for read in readinfo_full:
                    barcode = read[1]
                    suffix = barcode.split('-')[-1] if '-' in barcode else 'UNKNOWN'
                    condition = suffix.split('_')[0]
                    condition_groups[condition][suffix].append(read)
    
                num_conditions = len(condition_groups)
                reads_per_condition = MAX_CLUSTER_READS // max(num_conditions, 1)
    
                readinfo_sample = []
                for condition, replicates in condition_groups.items():
                    num_replicates = len(replicates)
                    reads_per_replicate = reads_per_condition // max(num_replicates, 1)
                    for replicate_id, reads in replicates.items():
                        if len(reads) <= reads_per_replicate:
                            readinfo_sample.extend(reads)
                        else:
                            sampled = np.random.choice(len(reads), reads_per_replicate, replace=False)
                            readinfo_sample.extend([reads[i] for i in sampled])
    
                sampled_set = set(tuple(r) for r in readinfo_sample)
                readinfo_leftover = [r for r in readinfo_full if tuple(r) not in sampled_set]
                downsampled = True
            else:
                readinfo_sample = readinfo_full
                readinfo_leftover = []
                downsampled = False
    
            # --- Clustering on sample (same as before) ---
            posi_sample = np.array([t[0] for t in readinfo_sample]).reshape(-1, 1)
            CB_sample = np.array([t[1] for t in readinfo_sample]).reshape(-1, 1)
            cigar_sample = np.array([t[2] for t in readinfo_sample]).reshape(-1, 1)
    
            model = AgglomerativeClustering(n_clusters=None, linkage='average', distance_threshold=self.InnerDistance)
            labels_sample = model.fit(posi_sample).labels_
    
            altTSSls_raw = []
            for lbl in np.unique(labels_sample):
                altTSSls_raw.append([
                    posi_sample[labels_sample == lbl],
                    CB_sample[labels_sample == lbl],
                    cigar_sample[labels_sample == lbl]
                ])
    
            # compute cluster metadata and leftover-chunk descriptors that the main process will use
            cluster_bounds = []
            cluster_centroids = []
            for cluster in altTSSls_raw:
                positions = cluster[0].flatten()
                cluster_bounds.append((np.min(positions), np.max(positions)))
                cluster_centroids.append(np.mean(positions))
    
            # chunk leftovers into descriptors (do not assign here)
            CHUNK_SIZE = 10000
            chunks = [readinfo_leftover[i:i + CHUNK_SIZE] for i in range(0, len(readinfo_leftover), CHUNK_SIZE)]
            logging.warning(f"[SAMPLE] Finished sample clustering for {geneid} with {len(readinfo_full)} reads")
            # return everything needed by main thread to submit chunk tasks
            return {
                "geneid": geneid,
                "altTSSls_raw": altTSSls_raw,
                "downsampled": downsampled,
                "cluster_bounds": cluster_bounds,
                "cluster_centroids": cluster_centroids,
                "chunks": chunks
            }
    
        except Exception as e:
            logging.error(f"[SAMPLE] FAILED: Gene {geneid} - {type(e).__name__}: {e}")
            return {"geneid": geneid, "error": True}
    
    
    # Updated recovery that uses a single shared chunk Pool
    def _recover_failed_genes_parallel(self, failed_genes, readinfodict, altTSSdict):
        recovered = 0
        gene_nproc = 2  # how many sample-clustering gene workers run concurrently
        total_cores = max(1, int(self.nproc))
        # allocate remaining cores to the shared chunk pool (at least 1)
        chunk_pool_procs = max(1, total_cores - gene_nproc)
    
        args = [(geneid, readinfodict.get(geneid)) for geneid in failed_genes if readinfodict.get(geneid)]
        skipped = [geneid for geneid in failed_genes if not readinfodict.get(geneid)]
        for geneid in skipped:
            logging.warning(f"[RECOVERY] No readinfo found for {geneid}, skipping.")
    
        # create a shared chunk Pool in the main process (spawn start)
        ctx = get_context("spawn")
        chunk_pool = ctx.Pool(processes=chunk_pool_procs)
    
        try:
            # Phase A: run sample clustering in parallel (no nested pools)
            with ProcessPoolExecutor(max_workers=gene_nproc) as executor:
                sample_futures = {executor.submit(self._do_clustering_sample, arg): arg[0] for arg in args}
    
                # as each gene sample finishes, submit its chunk tasks to the shared chunk_pool
                for future in as_completed(sample_futures):
                    geneid = sample_futures[future]
                    logging.warning(f"Returned sample clustering for {geneid}")
                    try:
                        sample_res = future.result(timeout=1200)
                    except Exception as e:
                        logging.error(f"[RECOVERY] Sample clustering failed for {geneid}: {type(e).__name__}: {e}")
                        continue
    
                    if sample_res.get("error"):
                        logging.warning(f"[RECOVERY] Sample clustering returned error for {geneid}, skipping.")
                        continue
    
                    # get items returned
                    geneid = sample_res["geneid"]
                    altTSSls_raw = sample_res["altTSSls_raw"]
                    downsampled = sample_res["downsampled"]
                    cluster_bounds = sample_res["cluster_bounds"]
                    cluster_centroids = sample_res["cluster_centroids"]
                    chunks = sample_res["chunks"]
    
                    # if nothing to do, finalize gene immediately
                    if not (downsampled and len(altTSSls_raw) > 0 and len(chunks) > 0):
                        # just filter clusters by minCount and save
                        altTSSls = [c for c in altTSSls_raw if c[0].shape[0] >= self.minCount]
                        altTSSdict[geneid] = altTSSls
                        recovered += 1
                        if geneid in failed_genes:
                            failed_genes.remove(geneid)
                        logging.warning(f"[RECOVERY] Gene {geneid} required no chunk assignment, saved {len(altTSSls)} clusters.")
                        continue
    
                    # submit chunk tasks to shared pool
                    logging.warning(f"[RECOVERY] Submitting {len(chunks)} chunks for gene {geneid} to shared chunk pool "
                                    f"(bounds={len(cluster_bounds)}, procs={chunk_pool_procs})")
    
                    # use starmap_async to assign each chunk; pass cluster_bounds + centroids as immutable arguments
                    task_iter = [(chunk, cluster_bounds, cluster_centroids) for chunk in chunks]
                    async_result = chunk_pool.starmap_async(_assign_leftover_chunk_bounded, task_iter)
    
                    # wait for chunk assignment to complete (optionally add a timeout)
                    try:
                        logging.warning(f"[RECOVERY] Waiting for chunk assignment to complete for {geneid}")
                        assigned_chunks = async_result.get(timeout=1800)  # adjust timeout as needed
                        logging.warning(f"[RECOVERY] Chunk assignment completed for {geneid}")
                    except Exception as e:
                        logging.error(f"[RECOVERY] Chunk assignment failed/timeout for gene {geneid}: {type(e).__name__}: {e}")
                        continue
    
                    # merge assigned_chunks into altTSSls_raw
                    # Step 1: collect reads per cluster label
                    cluster_batches = defaultdict(list)
                    for chunk_res in assigned_chunks:
                        for read, lbl_idx in chunk_res:
                            cluster_batches[lbl_idx].append(read)

                    # Step 2: stack all reads per cluster in one go
                    logging.warning(f"[MERGE] Begin chunk stacking for {geneid}")
                    for lbl_idx, reads in cluster_batches.items():
                        try:
                            posi_stack = np.vstack([r[0] for r in reads])
                            CB_stack = np.vstack([r[1] for r in reads])
                            cigar_stack = np.vstack([r[2] for r in reads])

                            altTSSls_raw[lbl_idx][0] = np.vstack((altTSSls_raw[lbl_idx][0], posi_stack))
                            altTSSls_raw[lbl_idx][1] = np.vstack((altTSSls_raw[lbl_idx][1], CB_stack))
                            altTSSls_raw[lbl_idx][2] = np.vstack((altTSSls_raw[lbl_idx][2], cigar_stack))
                        except Exception as e:
                            logging.error(f"[MERGE] Failed to stack cluster {lbl_idx}: {type(e).__name__}: {e}")
    
                    # Filter clusters by minCount and save
                    altTSSls = [c for c in altTSSls_raw if c[0].shape[0] >= self.minCount]
                    altTSSdict[geneid] = altTSSls
                    recovered += 1
                    if geneid in failed_genes:
                        failed_genes.remove(geneid)
                    logging.warning(f"[RECOVERY] Successfully recovered gene {geneid}: {len(altTSSls)} clusters saved.")
    
                    # periodic checkpointing can be done here if desired
                    # self._save_recovery_checkpoint(altTSSdict, failed_genes)  # optionally call
    
        finally:
            # ensure we always close the shared chunk pool
            try:
                chunk_pool.close()
                chunk_pool.join()
            except Exception:
                pass
    
        # final checkpoint and failure handling (preserve your original behavior)
        self._save_recovery_checkpoint(altTSSdict, failed_genes, final=True)
    
        if failed_genes:
            logging.warning(f"Still failed after recovery: {len(failed_genes)} genes")
            for gid in failed_genes:
                logging.warning(f"  - {gid}")
            print("Clustering halted due to unrecoverable genes. See log.txt for details.")
            sys.exit(1)
        else:
            logging.warning(f"Recovered {recovered} genes using parallel recovery.")
    
        return altTSSdict


    def _save_recovery_checkpoint(self, altTSSdict, failed_genes, final=False):
        suffix = 'after_recovery' if final else 'recovery_hourly'
        with open(os.path.join(self.count_out_dir, f'altTSSdict_{suffix}.pkl'), 'wb') as f:
            pickle.dump(altTSSdict, f)
        with open(os.path.join(self.count_out_dir, 'failed_genes.txt'), 'w') as f:
            f.write('\n'.join(failed_genes))
        if not final:
            logging.warning(f"[RECOVERY] Checkpoint saved to altTSSdict_recovery_hourly.pkl")

    def _do_hierarchial_cluster(self):
        
        start_time = time.time()
        last_save_time = start_time
        failed_genes = []
        
        # First get all reads for all genes, either load previously saved reads or get from BAM
        fetch_path = self.count_out_dir + 'fetch_reads.pkl'
        if os.path.exists(fetch_path):
            print("Resuming from existing fetch_reads.pkl...")
            with open(fetch_path, 'rb') as f:
                readinfodict = pickle.load(f)
        else:
            readinfodict = self._get_gene_reads()

        # Try resuming from before_cluster_peak.pkl first, then altTSSdict_hourly.pkl
        altTSSdict = {}
        recovery_hourly_path = os.path.join(self.count_out_dir, 'altTSSdict_recovery_hourly.pkl')
        peak_path = os.path.join(self.count_out_dir, 'before_cluster_peak.pkl')
        hourly_path = os.path.join(self.count_out_dir, 'altTSSdict_hourly.pkl')
        
        # --- Priority: resume from recovery checkpoint if present ---
        if os.path.exists(recovery_hourly_path):
            print("Resuming from altTSSdict_recovery_hourly.pkl...")
            with open(recovery_hourly_path, 'rb') as f:
                altTSSdict = pickle.load(f)
        
            failed_genes_path = os.path.join(self.count_out_dir, 'failed_genes.txt')
            if os.path.exists(failed_genes_path):
                with open(failed_genes_path, 'r') as f:
                    failed_genes = f.read().splitlines()
                logging.warning(f"[RECOVERY] Attempting to re-cluster {len(failed_genes)} failed genes using chunked multiprocessing.")
            else:
                failed_genes = [gid for gid in readinfodict if gid not in altTSSdict]
                logging.warning(f"[RECOVERY] Reconstructed {len(failed_genes)} failed genes from readinfodict and altTSSdict.")
        
            altTSSdict = self._recover_failed_genes_parallel(failed_genes, readinfodict, altTSSdict)
        
            # Final save
            final_path = os.path.join(self.count_out_dir, 'before_cluster_peak_Final.pkl')
            with open(final_path, 'wb') as f:
                pickle.dump(altTSSdict, f)
            logging.warning("[RECOVERY] Final clustering results saved to before_cluster_peak_Final.pkl.")
            return altTSSdict
        
        # --- Otherwise resume from before_cluster_peak.pkl ---
        elif os.path.exists(peak_path):
            print("Resuming from before_cluster_peak.pkl...")
            with open(peak_path, 'rb') as f:
                altTSSdict = pickle.load(f)
        
            failed_genes_path = os.path.join(self.count_out_dir, 'failed_genes.txt')
            if os.path.exists(failed_genes_path):
                with open(failed_genes_path, 'r') as f:
                    failed_genes = f.read().splitlines()
                logging.warning(f"[RECOVERY] Attempting to re-cluster {len(failed_genes)} failed genes using chunked multiprocessing.")
            else:
                failed_genes = [gid for gid in readinfodict if gid not in altTSSdict]
                logging.warning(f"[RECOVERY] Reconstructed {len(failed_genes)} failed genes from readinfodict and altTSSdict.")
        
            altTSSdict = self._recover_failed_genes_parallel(failed_genes, readinfodict, altTSSdict)
        
            final_path = os.path.join(self.count_out_dir, 'before_cluster_peak_Final.pkl')
            with open(final_path, 'wb') as f:
                pickle.dump(altTSSdict, f)
            logging.warning("[RECOVERY] Final clustering results saved to before_cluster_peak_Final.pkl.")
            return altTSSdict

        #otherwise start clustering from a checkpoint or from the begining        
        elif os.path.exists(hourly_path):
            print("Resuming from altTSSdict_hourly.pkl...")
            with open(hourly_path, 'rb') as f:
                altTSSdict = pickle.load(f)
            
        readls = list(readinfodict.keys())
        args = [(gid, readinfodict[gid]) for gid in readls if gid not in altTSSdict]
        skipped_genes = [gid for gid in readls if gid in altTSSdict]
        logging.warning(f"Skipping {len(skipped_genes)} genes already clustered.")
        print(f"Clustering {len(args)} genes with {self.nproc} processes...")

        with ProcessPoolExecutor(max_workers=self.nproc) as executor:
            futures = {executor.submit(self._do_clustering, arg): arg[0] for arg in args}
        
            for future in as_completed(futures):
                geneid = futures[future]
                try:
                    geneid, reslsSec = future.result()
                    futures.pop(future)
                    
                    if reslsSec is not None:
                        altTSSdict[geneid] = reslsSec
                        if not reslsSec:
                            logging.warning(f"Gene {geneid} clustered but returned no valid TSSs (all clusters < minCount)")
                    else:
                        logging.warning(f"Gene {geneid} failed (timeout or exception)")
                        failed_genes.append(geneid)
                        
                except Exception as e:
                    futures.pop(future)
                    logging.error(f"Gene {geneid} crashed: {type(e).__name__}: {e}")
                    failed_genes.append(geneid)
                    
                # Save intermediate results every hour
                current_time = time.time()
                if current_time - last_save_time >= 3600:
                    checkpoint_path = os.path.join(self.count_out_dir, 'altTSSdict_hourly.pkl')
                    with open(checkpoint_path, 'wb') as f:
                        pickle.dump(altTSSdict, f)
                    logging.warning(f"Checkpoint saved to altTSSdict_hourly.pkl at {int((current_time - start_time) / 60)} min")
                    last_save_time = current_time
    
        # Save before recovery attempts
        tss_output = os.path.join(self.count_out_dir, 'before_cluster_peak.pkl')
        with open(tss_output, 'wb') as f:
            pickle.dump(altTSSdict, f)
    
        logging.warning(f"do clustering Time elapsed, {int(time.time() - start_time)} seconds.")

        if failed_genes:
            logging.warning(f"Clustering failed for {len(failed_genes)} genes:")
            for gid in failed_genes:
                logging.warning(f"  - {gid}")
            logging.warning("Clustering halted due to failed genes. See log.txt for details.")
            with open(os.path.join(self.count_out_dir, 'failed_genes.txt'), 'w') as f:
                f.write('\n'.join(failed_genes))

            #  Retry failed genes one-by-one using chunked multiprocessing ---
            altTSSdict = self._recover_failed_genes_parallel(failed_genes, readinfodict, altTSSdict)

        # Final save
        tss_output = os.path.join(self.count_out_dir, 'before_cluster_peak_Final.pkl')
        with open(tss_output, 'wb') as f:
            pickle.dump(altTSSdict, f)
                                        
        return altTSSdict



    def _filter_false_positive(self):

        altTSSdict=self._do_hierarchial_cluster()      
        #print(altTSSdict)
        #get testX
        ## get RNA-seq X
        #make a new dictionary
        print("Filtering...")
        clusterdict={}
        for i in altTSSdict.keys():
            for j in range(0,len(altTSSdict[i])):
                #print(altTSSdict[i][j])
                startpos=np.min(altTSSdict[i][j][0])
                stoppos=np.max(altTSSdict[i][j][0])
                clustername=str(i)+'*'+str(startpos)+'_'+str(stoppos)
                

                count=len(altTSSdict[i][j][0])
                std=statistics.stdev(altTSSdict[i][j][0].flatten())
                summit_count=np.max(np.unique(altTSSdict[i][j][0].flatten(),return_counts=True)[1])
                unencoded_G_percent=sum([('14S' in ele)or('15S' in ele)or('16S' in ele) for ele in altTSSdict[i][j][2].flatten()])/count
                
                #summit position
                tempposi,tempposicount=np.unique(altTSSdict[i][j][0].flatten(),return_counts=True)
                maxpos=np.argmax(tempposicount)
                summitpos=tempposi[maxpos]   

                clusterdict[clustername]=(count,std,summit_count,unencoded_G_percent,j,i,summitpos)    
                #summitpos,altTSSdict[i][j][0],altTSSdict[i][j][1],altTSSdict[i][j][2]
                

        # cluster_output=self.count_out_dir+'before_filter_cluster.pkl'
        # with open(cluster_output,'wb') as f:
        #     pickle.dump(clusterdict,f)

        
        fourfeaturedf=pd.DataFrame(clusterdict).T 
        fourfeaturedf.columns=['UMI_count','SD','summit_UMI_count','unencoded_G_percent','NO.TSS','gene_id','summit_position']
        fourfeature_output=self.count_out_dir+'fourFeature.csv'
        fourfeaturedf.to_csv(fourfeature_output)

        print('one_gene_with_two_TSS_fourfeature : %i'%(len(fourfeaturedf)))
        test_X=fourfeaturedf.iloc[:,0:4]
        # print('hello')

        # print(os.path.abspath(__file__))

        # print(os.path.dirname(os.path.abspath(__file__)))

        # print(Path(os.path.dirname(os.path.abspath(__file__))))

        # print(Path(os.path.dirname(os.path.abspath(__file__))).parents[1])

        pathstr=str(Path(os.path.dirname(os.path.abspath(__file__))).parents[0])+'/model/logistic_4feature_model.sav'
        loaded_model = pickle.load(open(pathstr, 'rb'))
        test_Y=loaded_model.predict(test_X.values)

        #do filtering, the result of this step should be output as final h5ad file display at single cell level. 
        afterfiltereddf=fourfeaturedf[test_Y==1]
        afterfiltereddf.columns=['UMI_count','SD','summit_UMI_count','unencoded_G_percent','NO.TSS','gene_id','summit_position']

        afterfilter_output=self.count_out_dir+'afterfiltered.csv'
        afterfiltereddf.to_csv(afterfilter_output)


        allgeneID=afterfiltereddf['gene_id'].unique()
        keepdict={}
        for i in allgeneID:
            selectgeneiddf=afterfiltereddf[afterfiltereddf['gene_id']==i]
            keeptranscriptls=[]
            for j in selectgeneiddf.index:
                index=afterfiltereddf.loc[j]['NO.TSS']
                keeptranscriptls.append(altTSSdict[i][index])
            keepdict[i]=keeptranscriptls
            

        print("Finished filtering, saveing keepdict.pkl...")

        tss_output=self.count_out_dir+'keepdict.pkl'
        with open(tss_output,'wb') as f:
            pickle.dump(keepdict,f)

        
        return keepdict


    def _do_anno_and_filter(self,inputpar):
        #get gene ID
        geneid=inputpar[0]
        #get list of TSS for gene ID
        altTSSitemdict=inputpar[1]
        #get rows of reference for gene
        temprefdf=self.tssrefdf[self.tssrefdf['gene_id']==geneid]

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

    def _load_and_annotate(self, inputpair):
        geneid, filepath = inputpair
        with open(filepath, "rb") as f:
            altTSSitemdict = pickle.load(f)

        result = self._do_anno_and_filter((geneid, altTSSitemdict))

        # Save result to disk
        outpath = os.path.join(self.count_out_dir, f"anno_{geneid}.pkl")
        with open(outpath, "wb") as f:
            pickle.dump(result, f)
        return outpath


    def _TSS_annotation(self):
        
        start_time = time.time()
    
        keepdict_path = os.path.join(self.count_out_dir, 'keepdict.pkl')
    
        # Step 1: Load or compute keepdict
        if os.path.exists(keepdict_path):
            logging.warning("[RECOVERY] Loading existing keepdict.pkl")
            with open(keepdict_path, "rb") as f:
                keepdict = pickle.load(f)
        else:
            logging.warning("[RECOVERY] Computing keepdict from scratch")
            keepdict = self._filter_false_positive()
            with open(keepdict_path, "wb") as f:
                pickle.dump(keepdict, f)
    
        keepIDls = list(keepdict.keys())
    
        # Step 2: Save each gene's cluster data to disk and prepare input parameters
        # Step 2: Save each gene's cluster data to disk if not already present
        inputpar = []
        for geneid in keepIDls:
            filepath = os.path.join(self.count_out_dir, f"cluster_{geneid}.pkl")
            if os.path.exists(filepath):
                logging.warning(f"[RECOVERY] Found existing cluster file for {geneid}, skipping write.")
            else:
                with open(filepath, "wb") as f:
                    pickle.dump(keepdict[geneid], f)
                logging.warning(f"[RECOVERY] Saved cluster file for {geneid}")
            inputpar.append((geneid, filepath))
    
        # Step 3: Annotate in parallel using file paths
        with multiprocessing.Pool(self.nproc) as pool:
            result_paths = pool.map(self._load_and_annotate, inputpar)

        # Step 4: Load results from disk and delete result files
        transcriptdictls = []
        for path in result_paths:
            try:
                with open(path, "rb") as f:
                    transcriptdictls.append(pickle.load(f))
                os.remove(path)
            except Exception as e:
                logging.warning(f"[RECOVERY] Failed to load/delete result file {path}: {type(e).__name__}: {e}")

        # Step 5: Delete temporary cluster files
        for _, filepath in inputpar:
            try:
                os.remove(filepath)
            except Exception as e:
                logging.warning(f"[RECOVERY] Failed to delete temp file {filepath}: {type(e).__name__}: {e}")
        
        # Step 6: Save annotation results
        tss_output = os.path.join(self.count_out_dir, 'temp_tss.pkl')
        with open(tss_output, 'wb') as f:
            pickle.dump(transcriptdictls, f)
    
        # Step 7: Flatten and organize results
        extendls = []
        for d in transcriptdictls:
            extendls.extend(list(d.items()))
    
        d = {
            'transcript_id': [transcript[0] for transcript in extendls],
            'TSS_start': [np.min(transcript[1][0]) for transcript in extendls],
            'TSS_end': [np.max(transcript[1][0]) for transcript in extendls]
        }
        regiondf = pd.DataFrame(d)
    
        print('do annotation Time elapsed', int(time.time() - start_time), 'seconds.')
    
        # Save CSV outputs
        extendls_df = pd.DataFrame(extendls, columns=['transcript_id', 'details'])
        extendls_df.to_csv(os.path.join(self.count_out_dir, 'extendls.csv'), index=False)
        extendls_path = os.path.join(self.count_out_dir, 'extendls.pkl')
        with open(extendls_path, 'wb') as f:
            pickle.dump(extendls, f)

        regiondf.to_csv(os.path.join(self.count_out_dir, 'regiondf.csv'))

        return extendls, regiondf

        



    def _build_transcript_column(self, extend_entry, final_index):
        transcriptid = extend_entry[0]
        cellID, count = np.unique(extend_entry[1][1], return_counts=True)
        transcriptdf = pd.DataFrame({'cell_id': cellID, transcriptid: count})
        transcriptdf.set_index('cell_id', inplace=True)
        return transcriptid, final_index.map(transcriptdf[transcriptid]).fillna(0)



    def produce_sclevel(self):
        ctime=time.time()
        extendls_path = os.path.join(self.count_out_dir, 'extendls.pkl')
        regiondf_path = os.path.join(self.count_out_dir, 'regiondf.csv')

        if not (os.path.exists(extendls_path) and os.path.exists(regiondf_path)):
            extendls, regiondf = self._TSS_annotation()
        else:
            logging.warning("[SCLEVEL] Found existing extendls.csv and regiondf.csv — resuming from saved annotation results.")
            regiondf = pd.read_csv(regiondf_path)
            extendls_path = os.path.join(self.count_out_dir, 'extendls.pkl')
            with open(extendls_path, 'rb') as f:
                extendls = pickle.load(f)


        #transcriptdfls=[]

        cellIDls=[]
        for i in range(0,len(extendls)):
            cellID=np.unique(extendls[i][1][1])
            cellIDls.append(list(cellID))
        cellIDset = set([item for sublist in cellIDls for item in sublist])


        finaldf = pd.DataFrame(index=list(cellIDset))
        args = [(extendls[i], finaldf.index) for i in range(len(extendls))]

        with Pool(self.nproc) as pool:
            results = pool.starmap(self._build_transcript_column, args)

        for transcriptid, col in results:
            finaldf[transcriptid] = col

        logging.warning(f"[SCLEVEL] Finished building transcript matrix with {len(results)} columns.")

        finaldf.fillna(0,inplace=True)
        finaldf.to_csv('finaldf.csv')
        adata=ad.AnnData(finaldf)
        adata.write('adata.h5ad')
        vardf=pd.DataFrame(adata.var.copy())
        vardf.reset_index(inplace=True)
        vardf.columns=['transcript_id']
        vardf=vardf.join(regiondf.set_index('transcript_id'), on='transcript_id')
        vardf['gene_id']=vardf['transcript_id'].str.split('_',expand=True)[0]
        vardf=vardf.merge(self.generefdf,on='gene_id')
        vardf.set_index('transcript_id',drop=True,inplace=True)

        adata.var=vardf
        sc_output_h5ad=self.count_out_dir+'scTSS_count_all.h5ad'
        adata.write(sc_output_h5ad)

        #filter according to user' defined distance
        newdf=adata.var.copy()
        newdf.reset_index(inplace=True)
        selectedf=newdf[newdf.duplicated('gene_id',keep=False)]  #get data frame which includes two transcript for one gene
        geneID=selectedf['gene_id'].unique()

        keepdfls=[]
        for i in geneID:
            tempdf=selectedf[selectedf['gene_id']==i]

            tempdf=tempdf.sort_values('transcript_id',ascending=False)
            tempdf['diff']=tempdf['TSS_start'].diff()
            keepdf=tempdf[tempdf['diff'].isna()|tempdf['diff'].abs().ge(self.clusterDistance)]    #want to get TSS whose cluster distance is more than user defined.
            #keepdf=keepdf.iloc[:2,:]
            keepdfls.append(keepdf)

        #print(keepdfls)


        allkeepdf=reduce(lambda x,y:pd.concat([x,y]),keepdfls)
        finaltwodf=allkeepdf[allkeepdf.duplicated('gene_id',keep=False)]
        finaltwoadata=adata[:,adata.var.index.isin(finaltwodf['transcript_id'])]

        sc_output_h5ad=self.count_out_dir+'scTSS_count_two.h5ad'
        finaltwoadata.write(sc_output_h5ad)


        print('produce h5ad Time elapsed',int(time.time()-ctime),'seconds.')


        return adata









    def window_sliding(self,genereads,TSS_start,TSS_end,strand):

        leftIndex=0

        # do filtering; drop reads which does not include unencoded G
        filterls=[]
        for i in genereads:
            if ('14S' in i[2]) or ('15S' in i[2]) or ('16S' in i[2]):
                filterls.append(i)


        #calculate the TSS position and corresponding counts
        promoterTSS=[]
        for read in filterls:
            tss=read[0]
            if (tss>=TSS_start)&(tss<=TSS_end):
                promoterTSS.append(tss)
        TSS,count=np.unique(promoterTSS,return_counts=True)

        nonzeroarray=np.asarray((TSS, count)).T


        if strand=='+':
            sortfinalarray=nonzeroarray[nonzeroarray[:, 0].argsort()]
            TSS=sortfinalarray.T[0]
            count=sortfinalarray.T[1]
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



    def _get_CTSS(self,fetchadata):
        oneclusterfilePath=self.count_out_dir+'afterfiltered.csv'
        alloneclusterdf=pd.read_csv(oneclusterfilePath)
        alloneclusterdf['gene_id']=alloneclusterdf['Unnamed: 0'].str.split('*',expand=True)[0]
        alloneclusterdf['TSS_start']=alloneclusterdf['Unnamed: 0'].str.split('*',expand=True)[1].str.split('_',expand=True)[0].astype('float')
        alloneclusterdf['TSS_end']=alloneclusterdf['Unnamed: 0'].str.split('_',expand=True)[1].astype('float')

        self.generefdf.reset_index(inplace=True)
        

        #print(self.generefdf)
        stranddf=self.generefdf[['Strand','gene_id']]
        alloneclusterdf=alloneclusterdf.merge(stranddf,on='gene_id')




        start_time=time.time()

        allsortfddict={}

        for i in range(0,len(alloneclusterdf)):
            geneID=alloneclusterdf['gene_id'][i]
            # print(geneID)
            genereads=fetchadata[geneID]
            clusterID=alloneclusterdf['Unnamed: 0'][i]
            TSS_start=alloneclusterdf['TSS_start'][i]
            TSS_end=alloneclusterdf['TSS_end'][i]
            strand=alloneclusterdf['Strand'][i]
            windowreturn=self.window_sliding(genereads,TSS_start,TSS_end,strand)
            allsortfddict[clusterID]=windowreturn


        print('window sliding Time elapsed',int(time.time()-start_time),'seconds.')

        
        ctssOutPath=self.ctss_out_dir+'CTSS_foldchange.pkl'
        with open(ctssOutPath,'wb') as f:
            pickle.dump(allsortfddict,f)

        return allsortfddict


    def pickCTSS(self,ctssls):

        keepCTSS=[]
        for ele in ctssls:
            if (ele[1]>self.minCTSSCount)&(ele[2]>self.minFC):
                keepCTSS.append(ele)
        return keepCTSS
    





    def produce_CTSS_adata(self):
        ctime=time.time()


        self.ctss_out_dir=str(self.outdir)+'/CTSS/'
        if not os.path.exists(self.ctss_out_dir):
            os.mkdir(self.ctss_out_dir)






        readspath=self.count_out_dir+'fetch_reads.pkl'
        with open(readspath,'rb') as f:
            fetchadata=pickle.load(f)

        allsortfddict=self._get_CTSS(fetchadata)
        keepdict={}
        for ctssid in allsortfddict.keys():
            keepdict[ctssid]=self.pickCTSS(allsortfddict[ctssid])
        
        #print(keepdict)


        #get the cellID meeting our requirement
        cellIDdict={}
        for i in keepdict.keys():
            
            for j in keepdict[i]:
                geneid=i.split('*')[0]
                newid=i+'#'+str(j[0])+'@'+str(j[1])+'$'+str(j[2])
                cellIDls=[]
                for ele in fetchadata[geneid]:
                    if j[0]==ele[0]:
                        cellIDls.append(ele[1])
                cellIDdict[newid]=cellIDls

        #print(len(cellIDdict))


        #create a big matrix including cell ID
        cellidls=list(cellIDdict.values())
        cellidset = list(set([item for sublist in cellidls for item in sublist]))
        ctssfinaldf=pd.DataFrame(index=cellidset)



        
        for clusterID in cellIDdict.keys():
            cellID,count=np.unique(cellIDdict[clusterID],return_counts=True)
            CTSSdf=pd.DataFrame({'cell_id':cellID,clusterID:count})
            CTSSdf.set_index('cell_id',inplace=True)
            ctssfinaldf[clusterID]=ctssfinaldf.index.map(CTSSdf[clusterID])


        ctssfinaldf.fillna(0,inplace=True)
        #print(ctssfinaldf)
        ctssadata=ad.AnnData(ctssfinaldf)

        ctssvardf=pd.DataFrame(ctssadata.var.copy())
        ctssvardf.reset_index(inplace=True)
        ctssvardf.columns=['clusterID']
        ctssvardf['gene_id']=ctssvardf['clusterID'].str.split('*',expand=True)[0]
        ctssvardf['CTSS']=ctssvardf['clusterID'].str.split('#',expand=True)[1].str.split('@',expand=True)[0]
        ctssvardf['counts_dropped_UnencodedG']=ctssvardf['clusterID'].str.split('@',expand=True)[1].str.split('$',expand=True)[0]
        ctssvardf['fold_change']=ctssvardf['clusterID'].str.split('$',expand=True)[1]


        ctssvardf=ctssvardf.merge(self.generefdf,on='gene_id')
        ctssvardf.set_index('clusterID',drop=True,inplace=True)
        ctssadata.var=ctssvardf

        ctss_output_h5ad=self.ctss_out_dir+'all_ctss.h5ad'
        ctssadata.write(ctss_output_h5ad)


        twoctssselect=ctssadata.var[ctssadata.var.duplicated('gene_id',keep=False)].index
        twoctssadata=ctssadata[:,twoctssselect]

        sc_output_h5ad=self.ctss_out_dir+'all_ctss_two.h5ad'
        twoctssadata.write(sc_output_h5ad)

        print('produce CTSS h5ad Time elapsed',int(time.time()-ctime),'seconds.')


        return twoctssadata



