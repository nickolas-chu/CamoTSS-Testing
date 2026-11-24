"""Compact on-disk store of the reads CamoTSS fetches per gene (replaces holding fetch_reads.pkl in memory).

Why: fetch_reads.pkl is one pickled dict {gene_id: [(TSS position, cell barcode, CIGAR), ...]}. Loading it builds a
Python tuple + int + 2 strings per read (~200+ bytes each; human 3 GB file -> ~20 GB in RAM, mouse 43.5 GB -> ~300 GB),
and the whole dict had to stay in the main process for the whole run. The store keeps the same reads, in the same order,
as three flat binary arrays plus two small lookup tables, and any one gene is read straight from disk (memory-mapped).

Layout of <count_dir>/fetch_reads_store/ :
  pos.int64      read TSS position (CamoTSS's raw coordinate, as in fetch_reads.pkl)
  cb.int32       index into barcodes.txt (the read's cell barcode)
  cigar.int32    index into cigars.txt (the read's CIGAR string)
  genes.tsv      gene_id, first read (offset), number of reads -- genes in the original order
  barcodes.txt, cigars.txt   one string per line, line number = code
  COMPLETE       written last; a store without it is unfinished and is rebuilt
For each gene, reads are kept in the order CamoTSS produced them, so get(gene) returns exactly the gene's list from
fetch_reads.pkl.
"""
import os
import pickle
import shutil
import numpy as np

STORE_DIRNAME = 'fetch_reads_store'


class ReadStoreWriter:
    """Append genes one at a time; nothing but the lookup tables is kept in memory."""

    def __init__(self, store_dir):
        self.final_dir = store_dir
        self.tmp_dir = store_dir + '.partial'
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)
        os.makedirs(self.tmp_dir)
        self.f_pos = open(os.path.join(self.tmp_dir, 'pos.int64'), 'wb')
        self.f_cb = open(os.path.join(self.tmp_dir, 'cb.int32'), 'wb')
        self.f_cig = open(os.path.join(self.tmp_dir, 'cigar.int32'), 'wb')
        self.cb_code, self.cig_code = {}, {}
        self.cb_list, self.cig_list = [], []
        self.genes = []                                   # (gene_id, offset, n)
        self.n = 0

    def _code(self, table, lst, s):
        c = table.get(s)
        if c is None:
            c = len(lst)
            table[s] = c
            lst.append(s)
        return c

    def add_gene(self, gene_id, reads):
        """reads: list of (position, barcode, CIGAR) tuples, in CamoTSS's order."""
        k = len(reads)
        pos = np.fromiter((r[0] for r in reads), dtype=np.int64, count=k)
        cb = np.fromiter((self._code(self.cb_code, self.cb_list, r[1]) for r in reads), dtype=np.int32, count=k)
        cig = np.fromiter((self._code(self.cig_code, self.cig_list, r[2]) for r in reads), dtype=np.int32, count=k)
        self.add_gene_arrays(gene_id, pos, cb, cig)

    def add_gene_arrays(self, gene_id, pos, cb, cig):
        self.f_pos.write(np.ascontiguousarray(pos, dtype=np.int64).tobytes())
        self.f_cb.write(np.ascontiguousarray(cb, dtype=np.int32).tobytes())
        self.f_cig.write(np.ascontiguousarray(cig, dtype=np.int32).tobytes())
        self.genes.append((str(gene_id), self.n, len(pos)))
        self.n += len(pos)

    def close(self):
        for f in (self.f_pos, self.f_cb, self.f_cig):
            f.close()
        with open(os.path.join(self.tmp_dir, 'genes.tsv'), 'w') as f:
            for g, o, k in self.genes:
                f.write(f"{g}\t{o}\t{k}\n")
        for name, lst in (('barcodes.txt', self.cb_list), ('cigars.txt', self.cig_list)):
            with open(os.path.join(self.tmp_dir, name), 'w') as f:
                f.write(''.join(s + '\n' for s in lst))
        open(os.path.join(self.tmp_dir, 'COMPLETE'), 'w').close()
        if os.path.exists(self.final_dir):
            shutil.rmtree(self.final_dir)
        os.rename(self.tmp_dir, self.final_dir)


class ReadStore:
    """Read-only access; cheap to open in every worker process (arrays are memory-mapped, not loaded)."""

    def __init__(self, store_dir):
        if not os.path.exists(os.path.join(store_dir, 'COMPLETE')):
            raise IOError(f"read store {store_dir} is missing or unfinished")
        self.dir = store_dir
        self.genes, self.index = [], {}
        with open(os.path.join(store_dir, 'genes.tsv')) as f:
            for line in f:
                g, o, k = line.rstrip('\n').split('\t')
                self.genes.append(g)
                self.index[g] = (int(o), int(k))
        n = sum(k for _, k in self.index.values())
        def mm(name, dt):
            return np.memmap(os.path.join(store_dir, name), dtype=dt, mode='r', shape=(n,)) if n else np.zeros(0, dt)
        self.pos, self.cb, self.cig = mm('pos.int64', np.int64), mm('cb.int32', np.int32), mm('cigar.int32', np.int32)
        self.barcodes = np.array(open(os.path.join(store_dir, 'barcodes.txt')).read().split('\n')[:-1], dtype=object)
        self.cigars = np.array(open(os.path.join(store_dir, 'cigars.txt')).read().split('\n')[:-1], dtype=object)
        # CamoTSS's unencoded-G test, once per distinct CIGAR: soft clip of 14, 15 or 16 bases
        self.cigar_g = np.array([('14S' in c) or ('15S' in c) or ('16S' in c) for c in self.cigars], dtype=bool)

    def n_reads(self, gene_id):
        return self.index[gene_id][1]

    def get(self, gene_id):
        """(positions int64, barcode codes int32, CIGAR codes int32) of one gene, in CamoTSS's read order."""
        o, k = self.index[gene_id]
        return np.array(self.pos[o:o + k]), np.array(self.cb[o:o + k]), np.array(self.cig[o:o + k])

    def get_tuples(self, gene_id):
        """The gene's reads exactly as fetch_reads.pkl held them: list of (position, barcode, CIGAR)."""
        pos, cb, cig = self.get(gene_id)
        return list(zip(pos.tolist(), self.barcodes[cb].tolist(), self.cigars[cig].tolist()))


def store_dir_of(count_out_dir):
    return os.path.join(count_out_dir, STORE_DIRNAME)


def has_store(count_out_dir):
    return os.path.exists(os.path.join(store_dir_of(count_out_dir), 'COMPLETE'))


class _StreamingFetchReads(pickle._Unpickler):
    """Unpickle fetch_reads.pkl ONE GENE AT A TIME (the file is one dict {gene: list of read tuples}).

    The pure-Python unpickler is used so the dict-building opcodes can be intercepted: every batch of (gene, reads)
    items is handed to on_gene() instead of being stored, and the memo is cleared after each batch (otherwise it would
    keep a reference to every object and memory would grow to the size of the whole file). Order is preserved.
    """
    dispatch = dict(pickle._Unpickler.dispatch)

    def __init__(self, f, on_gene):
        super().__init__(f)
        self.on_gene = on_gene
        self.top = None

    def load_setitems(self):
        items = self.pop_mark()
        if self.stack[-1] is not self.top:                 # some other dict: build it normally
            for i in range(0, len(items), 2):
                self.stack[-1][items[i]] = items[i + 1]
            return
        for i in range(0, len(items), 2):
            self.on_gene(items[i], items[i + 1])
        keep = {k: v for k, v in self.memo.items() if v is self.top}
        self.memo.clear()
        self.memo.update(keep)
    dispatch[pickle.SETITEMS[0]] = load_setitems

    def load_setitem(self):
        stack = self.stack
        value = stack.pop()
        key = stack.pop()
        if stack[-1] is self.top:
            self.on_gene(key, value)
        else:
            stack[-1][key] = value
    dispatch[pickle.SETITEM[0]] = load_setitem

    def load_empty_dictionary(self):
        d = {}
        if self.top is None:
            self.top = d
        self.append(d)
    dispatch[pickle.EMPTY_DICT[0]] = load_empty_dictionary


def convert_fetch_reads_pickle(pkl_path, store_dir):
    """One-time conversion of an existing fetch_reads.pkl to the store, streaming (memory ~ one gene)."""
    w = ReadStoreWriter(store_dir)
    with open(pkl_path, 'rb', buffering=64 * 1024 * 1024) as f:
        _StreamingFetchReads(f, w.add_gene).load()
    w.close()
    return store_dir
