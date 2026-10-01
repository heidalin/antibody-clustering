import streamlit as st
import pandas as pd
import numpy as np
import torch
from rapidfuzz.distance import Levenshtein
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import normalize
from scipy.spatial.distance import pdist, squareform

# ==========================================
# 1. UI SETUP & CACHING
# ==========================================
st.set_page_config(page_title="Antibody Clustering Suite", layout="wide")
st.title("🧬 Antibody Clustering & Benchmarking Suite")
st.markdown("Upload your single-cell dataset and execute advanced clustering topologies.")

@st.cache_resource
def load_ai_model():
    tokenizer = AutoTokenizer.from_pretrained("brineylab/BALM-paired")
    model = AutoModel.from_pretrained("brineylab/BALM-paired")
    model.eval()
    return tokenizer, model

# ==========================================
# 2. THE MATHEMATICAL TOURNAMENT ENGINE
# ==========================================
def evaluate_clusters(labels, seq_data):
    tp, fp, fn, tn = 0, 0, 0, 0
    clusters = {}
    for idx, c_id in enumerate(labels):
        clusters.setdefault(c_id, []).append(idx)

    for i in range(len(seq_data)):
        my_cluster = clusters[labels[i]]
        has_other_binder = any(m != i and seq_data[m]['label'] == 'BINDER' for m in my_cluster)
        is_binder = seq_data[i]['label'] == 'BINDER'

        if has_other_binder:
            if is_binder: tp += 1
            else: fp += 1
        else:
            if is_binder: fn += 1
            else: tn += 1

    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return f1, tp, fp, fn, tn, clusters

def run_tournament(dist_matrix, thresholds, seq_data, idx_length_desc):
    n_seqs = len(seq_data)
    best_overall = {'f1': -1, 'method': None, 't': None}
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    for curr, t in enumerate(thresholds):
        # 1. Agglomerative
        agg = AgglomerativeClustering(n_clusters=None, metric='precomputed', linkage='complete', distance_threshold=max(t, 1e-6))
        labels_agg = agg.fit_predict(dist_matrix)
        
        # 2. DBSCAN
        db = DBSCAN(eps=max(t, 1e-6), min_samples=1, metric='precomputed')
        labels_db = db.fit_predict(dist_matrix)
        
        # 3. Greedy First
        c_reps_f, c_assign_f = [], {i: [] for i in range(n_seqs)}
        for i in idx_length_desc:
            matched = False
            for rep in c_reps_f:
                if dist_matrix[i, rep] <= t:
                    c_assign_f[rep].append(i); matched = True; break
            if not matched: c_reps_f.append(i); c_assign_f[i].append(i)
        labels_f = np.zeros(n_seqs, dtype=int)
        for c_id, (rep, mem) in enumerate(c_assign_f.items()):
            for m in mem: labels_f[m] = c_id

        # 4. Greedy Best
        c_reps_b, c_assign_b = [], {i: [] for i in range(n_seqs)}
        for i in idx_length_desc:
            best_rep, best_d = None, 999.0
            for rep in c_reps_b:
                d = dist_matrix[i, rep]
                if d <= t and d < best_d: best_d = d; best_rep = rep
            if best_rep is not None: c_assign_b[best_rep].append(i)
            else: c_reps_b.append(i); c_assign_b[i].append(i)
        labels_b = np.zeros(n_seqs, dtype=int)
        for c_id, (rep, mem) in enumerate(c_assign_b.items()):
            for m in mem: labels_b[m] = c_id

        methods = {
            'Agglomerative (Hierarchy)': labels_agg,
            'DBSCAN (Density)': labels_db,
            'Greedy First-Match': labels_f,
            'Greedy Best-Match': labels_b
        }

        for m_name, labels in methods.items():
            f1, tp, fp, fn, tn, clusters = evaluate_clusters(labels, seq_data)
            if f1 > best_overall['f1']:
                best_overall = {'f1': f1, 'method': m_name, 't': t, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'clusters': clusters}

        progress_bar.progress((curr + 1) / len(thresholds))
        status_text.text(f"Testing Thresholds: {curr+1}/{len(thresholds)} | Current Best F1: {best_overall['f1']:.4f}")
        
    return best_overall

# ==========================================
# 3. INTERFACE & LOGIC
# ==========================================
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])
methodology = st.sidebar.selectbox("Select Clustering Methodology", [
    "1. Pure Sequence",
    "2. Clonotype",
    "3. Paratope (6-CDR)",
    "4. Embedding (AI)",
    "5. Structural (3D Proxy)"
])

if st.sidebar.button("▶ Run Clustering Tournament"):
    if uploaded_file is None:
        st.error("Please upload a dataset first!")
    else:
        with st.spinner("Cleaning and preparing data..."):
            df = pd.read_csv(uploaded_file)
            df = df[df["PT_BINDING"] != "UNKNOWN"]
            
            if methodology in ["1. Pure Sequence", "2. Clonotype", "4. Embedding (AI)"]:
                df = df.dropna(subset=["VL_CDR3_AA", "VH_CDR3_AA", "VH_GERMLINE", "VL_GERMLINE"])
            else:
                df = df.dropna(subset=["VL_CDR1_AA", "VL_CDR2_AA", "VL_CDR3_AA", "VH_CDR1_AA", "VH_CDR2_AA", "VH_CDR3_AA"])
                
            seq_data = []
            for index, row in df.iterrows():
                vl_cdr3 = str(row.get("VL_CDR3_AA", "")).upper().strip()
                vh_cdr3 = str(row.get("VH_CDR3_AA", "")).upper().strip()
                vh_gene = str(row.get("VH_GERMLINE", "")).split('*')[0].strip()
                vl_gene = str(row.get("VL_GERMLINE", "")).split('*')[0].strip()
                
                if methodology in ["1. Pure Sequence", "2. Clonotype"]:
                    seq = vl_cdr3 + vh_cdr3
                elif methodology == "4. Embedding (AI)":
                    seq = vh_cdr3 + "</s>" + vl_cdr3
                else:
                    vl1, vl2 = str(row.get("VL_CDR1_AA", "")).upper().strip(), str(row.get("VL_CDR2_AA", "")).upper().strip()
                    vh1, vh2 = str(row.get("VH_CDR1_AA", "")).upper().strip(), str(row.get("VH_CDR2_AA", "")).upper().strip()
                    seq = vl1 + vl2 + vl_cdr3 + vh1 + vh2 + vh_cdr3
                    
                seq_data.append({
                    'id': row.get("CLONE_ID", ""), 'seq': seq, 'label': row["PT_BINDING"], 
                    'vh_gene': vh_gene, 'vl_gene': vl_gene, 'h3_length': len(vh_cdr3), 'total_length': len(seq)
                })

            n_seqs = len(seq_data)
            idx_length_desc = sorted(range(n_seqs), key=lambda i: seq_data[i]['total_length'], reverse=True)
            dist_matrix = np.zeros((n_seqs, n_seqs), dtype=np.float32)

        st.success(f"Loaded {n_seqs} sequences successfully!")

        with st.spinner(f"Calculating Distance Matrix for {methodology}..."):
            if "Embedding" in methodology or "Structural" in methodology:
                tokenizer, model = load_ai_model()
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)
                
                embeddings = []
                with torch.no_grad():
                    for s in seq_data:
                        spaced = " ".join(list(s['seq'])) if "Structural" not in methodology else s['seq']
                        inputs = tokenizer(spaced, return_tensors="pt").to(device)
                        emb = model(**inputs).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                        embeddings.append(emb)
                embeddings = np.array(embeddings)
                
                if "Structural" in methodology:
                    embeddings = normalize(embeddings, norm='l2')
                    base_dist = squareform(pdist(embeddings, metric='euclidean'))
                else:
                    base_dist = cosine_distances(embeddings)
                    
                for i in range(n_seqs):
                    for j in range(n_seqs):
                        if methodology == "4. Embedding (AI)":
                            if (seq_data[i]['vh_gene'] == seq_data[j]['vh_gene']) and (seq_data[i]['vl_gene'] == seq_data[j]['vl_gene']) and (seq_data[i]['h3_length'] == seq_data[j]['h3_length']):
                                dist_matrix[i, j] = base_dist[i, j]
                            else: dist_matrix[i, j] = 999.0
                        else:
                            if seq_data[i]['h3_length'] == seq_data[j]['h3_length']:
                                dist_matrix[i, j] = base_dist[i, j]
                            else: dist_matrix[i, j] = 999.0

            else:
                for i in range(n_seqs):
                    for j in range(n_seqs):
                        allowed = True
                        if methodology == "2. Clonotype":
                            if seq_data[i]['vh_gene'] != seq_data[j]['vh_gene'] or seq_data[i]['h3_length'] != seq_data[j]['h3_length']:
                                allowed = False
                        if methodology == "3. Paratope (6-CDR)":
                            if seq_data[i]['h3_length'] != seq_data[j]['h3_length']:
                                allowed = False
                                
                        if allowed:
                            d = Levenshtein.distance(seq_data[i]['seq'], seq_data[j]['seq'])
                            den = (seq_data[i]['total_length'] + seq_data[j]['total_length']) / 2.0
                            dist_matrix[i, j] = d / den
                        else:
                            dist_matrix[i, j] = 999.0

        unique_distances = set()
        for i in range(n_seqs):
            for j in range(i+1, n_seqs):
                if dist_matrix[i, j] < 900: unique_distances.add(round(dist_matrix[i, j], 3))
        
        thresholds = sorted(list(unique_distances))
        if len(thresholds) == 0: thresholds = np.linspace(0.01, 1.0, 50)
        if "Sequence" in methodology: thresholds = np.linspace(0.05, 0.40, 50)

        st.info("Running 4-Algorithm Tournament Optimization...")
        winner = run_tournament(dist_matrix, thresholds, seq_data, idx_length_desc)

        # ---------------------------------------------------------
        # DISPLAY RESULTS
        # ---------------------------------------------------------
        st.markdown("---")
        st.subheader("🏆 Tournament Results")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Winning Algorithm", winner['method'])
        col2.metric("Optimal F1 Score", f"{winner['f1']:.4f}")
        
        # If it's a Levenshtein method, display Sequence Identity (1.0 - dist)
        display_t = (1.0 - winner['t']) if "Embedding" not in methodology and "Structural" not in methodology else winner['t']
        col3.metric("Optimal Threshold", f"{display_t:.4f}")
        
        st.markdown("### Error Metrics")
        ecol1, ecol2, ecol3, ecol4 = st.columns(4)
        ecol1.metric("True Positives (TP)", winner['tp'])
        ecol2.metric("False Positives (FP)", winner['fp'])
        ecol3.metric("False Negatives (FN)", winner['fn'])
        ecol4.metric("True Negatives (TN)", winner['tn'])
        
        # ---------------------------------------------------------
        # DISPLAY ERROR PATTERNS
        # ---------------------------------------------------------
        st.markdown("---")
        st.subheader("📊 Detailed Error Pattern (Winning Method)")
        
        cluster_stats = []
        for c_id, members in winner['clusters'].items():
            c_tp, c_fp, c_fn, c_tn = 0, 0, 0, 0
            binders_in_cluster = sum(1 for m in members if seq_data[m]['label'] == 'BINDER')
            for m in members:
                if seq_data[m]['label'] == 'BINDER':
                    if binders_in_cluster > 1: c_tp += 1
                    else: c_fn += 1
                else:
                    if binders_in_cluster > 0: c_fp += 1
                    else: c_tn += 1
            cluster_stats.append({'Size': len(members), 'TP': c_tp, 'FP': c_fp, 'FN': c_fn, 'TN': c_tn})
            
        cluster_stats.sort(key=lambda x: x['Size'], reverse=True)
        multi_clusters = [c for c in cluster_stats if c['Size'] > 1]
        singletons = [c for c in cluster_stats if c['Size'] == 1]
        
        st.markdown("**Multi-Element Clusters (Size > 1)**")
        st.dataframe(pd.DataFrame(multi_clusters), use_container_width=True)
        
        st.markdown("**Singleton Clusters (Size = 1)**")
        singleton_fns = sum(c['FN'] for c in singletons)
        singleton_tns = sum(c['TN'] for c in singletons)
        
        st.info(f"**Total Singletons:** {len(singletons)} \n\n "
                f"• **False Negatives** (Binders that were isolated): {singleton_fns} \n\n "
                f"• **True Negatives** (Non-Binders successfully isolated): {singleton_tns}")

        st.success("Tournament Execution Complete!")