"""
Reviewer 3, comment 2: build an INDEPENDENT external validation set of
compounds with experimentally measured DPPH IC50 values reported in the
primary literature, excluding every compound used for training or testing.

Source: ChEMBL REST API (the same primary-literature-curated resource from
which the AODB records used by Ghironi et al. derive). Every retained record
keeps its ChEMBL activity/assay identifiers and the DOI of the primary
publication, so each experimental value is traceable to its source paper.

Selection rules
  1. Assay description mentions DPPH and a 30 min incubation, matching the
     assay window used for the modelling dataset.
  2. standard_type IC50, standard_relation '=', positive value, unit
     convertible to molar.
  3. Structure curated exactly as in the modelling dataset: largest organic
     fragment, charges neutralised, stereochemistry stripped, RDKit canonical
     SMILES.
  4. Any compound whose InChIKey connectivity block matches a training or test
     compound is removed.
  5. Replicate measurements of the same curated structure are averaged; the
     spread across sources is recorded.

Output: revision_work/results/external_set_raw.json
        revision_work/results/external_set.csv
"""

import os, sys, json, time, urllib.request, urllib.parse, pickle, collections, csv
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import warnings; warnings.filterwarnings('ignore')
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import inchi
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog('rdApp.*')

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE); PROJ = os.path.dirname(REV)
OUT = os.path.join(REV, "results"); os.makedirs(OUT, exist_ok=True)
BASE = "https://www.ebi.ac.uk/chembl/api/data/"

PATTERNS = [
    "DPPH radical scavenging activity after 30 min",
    "DPPH radical scavenging activity incubated for 30 min",
    "DPPH free radical scavenging activity after 30 min",
    "DPPH free radical scavenging activity incubated for 30 min",
    "DPPH radical scavenging activity at 30 min",
    "DPPH free radical scavenging activity at 30 min",
    "DPPH radical scavenging activity measured after 30 min",
    "scavenging activity against DPPH after 30 min",
    "scavenging activity against DPPH radical after 30 min",
    "DPPH radical after 30 min",
    "DPPH free radical after 30 min",
    "DPPH assay after 30 min",
    "antioxidant activity by DPPH method after 30 min",
    "DPPH radical scavenging assay after 30 min",
]


def get(path, **kw):
    url = BASE + path + "?" + urllib.parse.urlencode(kw)
    last = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return json.load(r)
        except Exception as e:
            last = e
            time.sleep(2 + 2 * attempt)
    raise RuntimeError(str(last) + " :: " + url)


def paged(path, key, page=1000, **kw):
    off, seen = 0, []
    while True:
        d = get(path, limit=page, offset=off, **kw)
        rows = d[key]
        seen += rows
        tot = d['page_meta']['total_count']
        off += page
        if off >= min(tot, 20000) or not rows:
            return seen


# ---------------------------------------------------------------- curation
_un = rdMolStandardize.Uncharger()


def curate(smiles):
    """Largest organic fragment, neutralised, stereochemistry stripped."""
    if not smiles:
        return None, None
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None, None
    try:
        m = rdMolStandardize.FragmentParent(m)
        m = _un.uncharge(m)
    except Exception:
        return None, None
    if m is None or m.GetNumHeavyAtoms() < 3:
        return None, None
    syms = set(a.GetSymbol() for a in m.GetAtoms())
    if 'C' not in syms:
        return None, None
    allowed = set(['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'H'])
    if syms - allowed:
        return None, None
    Chem.RemoveStereochemistry(m)
    try:
        can = Chem.MolToSmiles(m)
        key = inchi.MolToInchiKey(m)
    except Exception:
        return None, None
    if not can or not key:
        return None, None
    return can, key


TO_MOLAR = {'nM': 1e-9, 'uM': 1e-6, 'mM': 1e-3, 'M': 1.0, 'pM': 1e-12,
            'nmol/L': 1e-9, 'umol/L': 1e-6}


def main():
    # ---- 1. existing compounds (training + test) -------------------------
    with open(os.path.join(PROJ, "results", "processed_data.pkl"), 'rb') as f:
        data = pickle.load(f)
    df = data['df']
    smi_col = 'SMILES' if 'SMILES' in df.columns else df.columns[0]
    used_keys, used_can = set(), set()
    for s in df[smi_col].values:
        can, key = curate(s)
        if key:
            used_keys.add(key[:14])
            used_can.add(can)
    print("modelling dataset: %d rows -> %d connectivity keys" % (len(df), len(used_keys)), flush=True)

    # ---- 2. candidate assays --------------------------------------------
    assays = {}
    for pat in PATTERNS:
        rows = paged("assay.json", "assays", page=1000, description__icontains=pat)
        for a in rows:
            assays[a['assay_chembl_id']] = a['description']
        print("  [%5d] %s" % (len(rows), pat), flush=True)
    print("unique candidate assays: %d" % len(assays), flush=True)

    # ---- 3. activities for those assays ---------------------------------
    ids = sorted(assays)
    acts = []
    for i in range(0, len(ids), 20):
        chunk = ids[i:i + 20]
        rows = paged("activity.json", "activities", page=1000,
                     assay_chembl_id__in=",".join(chunk), standard_type="IC50")
        acts += rows
        if (i // 20) % 5 == 0:
            print("  activities %d/%d -> %d" % (i, len(ids), len(acts)), flush=True)
    print("raw IC50 activities: %d" % len(acts), flush=True)

    # ---- 4. filter + curate ---------------------------------------------
    recs, drop = [], collections.Counter()
    for a in acts:
        if a.get('standard_relation') not in ('=', None):
            drop['relation'] += 1
            continue
        v, u = a.get('standard_value'), a.get('standard_units')
        if v is None or u not in TO_MOLAR:
            drop['units'] += 1
            continue
        try:
            molar = float(v) * TO_MOLAR[u]
        except Exception:
            drop['value'] += 1
            continue
        if not (molar > 0):
            drop['value'] += 1
            continue
        can, key = curate(a.get('canonical_smiles'))
        if key is None:
            drop['structure'] += 1
            continue
        if key[:14] in used_keys or can in used_can:
            drop['in_modelling_set'] += 1
            continue
        recs.append(dict(
            chembl_id=a['molecule_chembl_id'], activity_id=a['activity_id'],
            assay_chembl_id=a['assay_chembl_id'],
            assay_description=assays.get(a['assay_chembl_id'], ''),
            document_chembl_id=a.get('document_chembl_id'),
            smiles=can, inchikey=key,
            ic50_molar=molar, pIC50=float(-np.log10(molar))))
    print("dropped: %s" % dict(drop), flush=True)
    print("candidate external records: %d" % len(recs), flush=True)

    # ---- 5. documents -> DOI / year -------------------------------------
    docids = sorted(set(r['document_chembl_id'] for r in recs if r['document_chembl_id']))
    docs = {}
    for i in range(0, len(docids), 20):
        chunk = docids[i:i + 20]
        rows = paged("document.json", "documents", page=200,
                     document_chembl_id__in=",".join(chunk))
        for d in rows:
            docs[d['document_chembl_id']] = dict(
                doi=d.get('doi'), year=d.get('year'), journal=d.get('journal'),
                title=(d.get('title') or '')[:300])
    print("documents resolved: %d" % len(docs), flush=True)

    # ---- 6. aggregate replicates by curated structure -------------------
    by_key = collections.defaultdict(list)
    for r in recs:
        by_key[r['inchikey']].append(r)
    compounds = []
    for key, rs in by_key.items():
        vals = np.array([r['pIC50'] for r in rs])
        srcs = []
        for r in rs:
            d = docs.get(r['document_chembl_id'], {})
            srcs.append(dict(chembl_id=r['chembl_id'], activity_id=r['activity_id'],
                             assay_chembl_id=r['assay_chembl_id'],
                             assay_description=r['assay_description'],
                             document_chembl_id=r['document_chembl_id'],
                             doi=d.get('doi'), year=d.get('year'),
                             journal=d.get('journal'), pIC50=r['pIC50']))
        compounds.append(dict(
            inchikey=key, smiles=rs[0]['smiles'], chembl_id=rs[0]['chembl_id'],
            pIC50_exp=float(vals.mean()), n_measurements=len(vals),
            pIC50_spread=float(vals.max() - vals.min()),
            dois=sorted(set(s['doi'] for s in srcs if s['doi'])),
            years=sorted(set(s['year'] for s in srcs if s['year'])),
            sources=srcs))
    compounds.sort(key=lambda c: c['inchikey'])
    print("unique external compounds: %d" % len(compounds), flush=True)
    n_doi = sum(1 for c in compounds if c['dois'])
    print("  with a resolvable DOI: %d" % n_doi, flush=True)
    sp = [c['pIC50_spread'] for c in compounds if c['n_measurements'] > 1]
    if sp:
        print("  replicated compounds: %d, median spread %.2f log units"
              % (len(sp), float(np.median(sp))), flush=True)

    with open(os.path.join(OUT, "external_set_raw.json"), 'w') as f:
        json.dump(dict(compounds=compounds, n_assays=len(assays),
                       n_raw_activities=len(acts), drop_counts=dict(drop),
                       patterns=PATTERNS), f, indent=1)
    with open(os.path.join(OUT, "external_set.csv"), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['inchikey', 'smiles', 'chembl_id', 'pIC50_exp', 'n_measurements',
                    'pIC50_spread', 'dois', 'years'])
        for c in compounds:
            w.writerow([c['inchikey'], c['smiles'], c['chembl_id'],
                        "%.4f" % c['pIC50_exp'], c['n_measurements'],
                        "%.4f" % c['pIC50_spread'], '; '.join(c['dois']),
                        '; '.join(str(y) for y in c['years'])])
    print("wrote external_set_raw.json / external_set.csv", flush=True)


if __name__ == '__main__':
    main()
