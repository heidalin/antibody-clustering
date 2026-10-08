"""
Antibody Clustering & Benchmarking Suite (Ultimate Edition)
Executes 17 orthogonal, hybridized, and ensemble clustering topologies.
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
from rapidfuzz.distance import Levenshtein
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
import fastcluster
from scipy.cluster.hierarchy import fcluster

# ==========================================
# 1. UI SETUP & CACHING
# ==========================================
st.set_page_config(page_title="Antibody Clustering Suite", layout="wide")
st.title("🧬 Antibody Clustering & Benchmarking Suite")
st.markdown("Execute 17 advanced clustering topologies, from pure Levenshtein text-math to 2D AI-Fusion Grid Searches.")

@st.cache_resource
def load_language_model():
    tokenizer = AutoTokenizer.from_pretrained("brineylab/BALM-paired")
    model = AutoModel.from_pretrained("brineylab/BALM-paired")
    model.eval()
    return tokenizer, model

# ==========================================
# 2. TOURNAMENT ENGINE
# ==========================================
def evaluate_clusters(labels, is_binder_array):
    cluster_counts = {}
    for i, c_id in enumerate(labels):
        if is_binder_array[i]: cluster_counts[c_id] = cluster_counts.get(c_id, 0) + 1
            
    tp, fp, fn, tn = 0, 0, 0, 0
    for i, c_id in enumerate(labels):
        is_binder = is_binder_array[i]
        binders_in_cluster = cluster_counts.get(c_id, 0)
        
        if is_binder:
            if binders_in_cluster > 1: tp += 1
            else: fn += 1
        else:
            if binders_in_cluster > 0: fp += 1
            else: tn += 1
            
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return f1, tp, fp, fn, tn

def run_tournament(dist_matrix, thresholds, n_seqs, idx_length_desc, is_binder_array):
    best_overall = {'f1': -1, 'method': None, 't': None}
    
    condensed = squareform(dist_matrix)
    Z = fastcluster.linkage(condensed, method='complete')
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    for curr, t in enumerate(thresholds):
        labels_agg = fcluster(Z, t=max(t, 1e-6), criterion='distance')
        labels_db = DBSCAN(eps=max(t, 1e-6), min_samples=1, metric='precomputed').fit_predict(dist_matrix)
        
        # Greedy First
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

        # Greedy Best
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

        methods = {'Agglomerative (Hierarchy)': labels_agg, 'DBSCAN (Density)': labels_db, 'Greedy First-Match': labels_f, 'Greedy Best-Match': labels_b}

        for m_name, labels in methods.items():
            f1, tp, fp, fn, tn = evaluate_clusters(labels, is_binder_array)
            if f1 > best_overall['f1']:
                
                # Reconstruct cluster dictionary for UI display
                clusters = {}
                for idx, c_id in enumerate(labels):
                    clusters.setdefault(c_id, []).append(idx)
                    
                best_overall = {'f1': f1, 'method': m_name, 't': t, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'clusters': clusters}

        progress_bar.progress((curr + 1) / len(thresholds))
        status_text.text(f"Evaluating Thresholds: {curr+1}/{len(thresholds)} | Peak F1: {best_overall['f1']:.4f}")
        
    return best_overall

# ==========================================
# 3. INTERFACE & LOGIC
# ==========================================
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload PTx CSV Dataset", type=["csv"])

methodologies = [
    "1. NON-AI: Pure Sequence (L3+H3)", 
    "2. NON-AI: Clonotype (L3+H3 + V-Gene Ban)", 
    "3. NON-AI: Paratope (6-CDR Proxy)", 
    "4. NON-AI: Fusion (Sequence + Paratope)",
    "5. NON-AI: Fusion (Sequence + Clonotype Soft-Margin)",
    "6. NON-AI: Fusion (Clonotype + Paratope Strict Ban)",
    "7. NON-AI: Grand Ensemble (Seq + Clono + Para)",
    "8. AI: Original Embedding (L3+H3, No V-Gene)",
    "9. AI: Original Structural (Euclidean, No V-Gene)",
    "10. AI: Embedding (L3+H3) + V-Gene Ban",
    "11. AI: Structural (Euclidean) + V-Gene Ban",
    "12. AI: Embedding (Paratope 6-CDR, No V-Gene)",
    "13. AI: Embedding (Paratope 6-CDR) + V-Gene Ban",
    "14. AI: Embedding (Full Sequence, No V-Gene)",
    "15. AI: Embedding (Full Sequence) + V-Gene Ban",
    "16. AI: PCA-Reduced (50-D) + V-Gene Ban",
    "17. 2D ADAPTIVE GRID SEARCH (AI + Biology Fusion)"
]

methodology = st.sidebar.selectbox("Select Clustering Methodology", methodologies)

if st.sidebar.button("▶ Execute Mathematical Topology"):
    if uploaded_file is None:
        st.error("Please upload a dataset first!")
    else:
        with st.spinner("Extracting sequence features and genetic annotations..."):
            df = pd.read_csv(uploaded_file)
            df = df[df["PT_BINDING"] != "UNKNOWN"]
            df = df.dropna(subset=[
                "VL_CDR1_AA", "VL_CDR2_AA", "VL_CDR3_AA", 
                "VH_CDR1_AA", "VH_CDR2_AA", "VH_CDR3_AA", "VH_GERMLINE", "VL_GERMLINE"
            ])
                
            seq_data = []
            for index, row in df.iterrows():
                vl1, vl2, vl3 = str(row["VL_CDR1_AA"]).upper().strip(), str(row["VL_CDR2_AA"]).upper().strip(), str(row["VL_CDR3_AA"]).upper().strip()
                vh1, vh2, vh3 = str(row["VH_CDR1_AA"]).upper().strip(), str(row["VH_CDR2_AA"]).upper().strip(), str(row["VH_CDR3_AA"]).upper().strip()
                vh_gene = str(row["VH_GERMLINE"]).split('*')[0].strip()
                vl_gene = str(row["VL_GERMLINE"]).split('*')[0].strip()
                
                str_l3h3 = vl3 + vh3
                str_para = vl1 + vl2 + vl3 + vh1 + vh2 + vh3
                lm_l3h3 = vh3 + "</s>" + vl3
                lm_para = vh1 + vh2 + vh3 + "</s>" + vl1 + vl2 + vl3
                
                if "VH_AA" in df.columns and "VL_AA" in df.columns:
                    lm_full = str(row["VH_AA"]).upper().strip() + "</s>" + str(row["VL_AA"]).upper().strip()
                else:
                    lm_full = lm_para
                    
                seq_data.append({
                    'id': row["CLONE_ID"], 'label': row["PT_BINDING"], 
                    'vh_gene': vh_gene, 'vl_gene': vl_gene, 'h3_length': len(vh3),
                    'str_l3h3': str_l3h3, 'len_l3h3': len(str_l3h3),
                    'str_para': str_para, 'len_para': len(str_para),
                    'lm_l3h3': lm_l3h3, 'lm_para': lm_para, 'lm_full': lm_full
                })

            n_seqs = len(seq_data)
            idx_length_desc = sorted(range(n_seqs), key=lambda i: seq_data[i]['len_l3h3'], reverse=True)
            is_binder_array = np.array([s['label'] == 'BINDER' for s in seq_data])
            dist_matrix = np.zeros((n_seqs, n_seqs), dtype=np.float32)

        st.success(f"Isolated {n_seqs} valid sequences.")

        with st.spinner(f"Computing Hyper-Dimensional Distance Matrix for {methodology[:10]}..."):
            
            # --- METHOD 17: 2D ADAPTIVE GRID SEARCH ---
            if "17" in methodology:
                tokenizer, model = load_language_model()
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)
                
                emb_full = []
                with torch.no_grad():
                    for s in seq_data:
                        inputs = tokenizer(" ".join(list(s['lm_full'])), return_tensors="pt").to(device)
                        emb_full.append(model(**inputs).last_hidden_state.mean(dim=1).squeeze().cpu().numpy())
                D_AI_FULL = cosine_distances(np.array(emb_full))
                
                D_text = np.zeros((n_seqs, n_seqs), dtype=np.float32)
                for i in range(n_seqs):
                    for j in range(n_seqs):
                        d_seq = Levenshtein.distance(seq_data[i]['str_l3h3'], seq_data[j]['str_l3h3']) / ((seq_data[i]['len_l3h3'] + seq_data[j]['len_l3h3']) / 2.0)
                        d_par = Levenshtein.distance(seq_data[i]['str_para'], seq_data[j]['str_para']) / ((seq_data[i]['len_para'] + seq_data[j]['len_para']) / 2.0)
                        D_text[i, j] = (d_seq + d_par) / 2.0
                
                best_2d = {'f1': -1, 'w': -1, 't': -1, 'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
                weights = np.linspace(0.0, 1.0, 11)
                thresholds = np.arange(0.01, 0.40, 0.01)
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx_w, w in enumerate(weights):
                    D_fused = (w * D_text) + ((1.0 - w) * D_AI_FULL)
                    for i in range(n_seqs):
                        for j in range(n_seqs):
                            same_genes = (seq_data[i]['vh_gene'] == seq_data[j]['vh_gene']) and (seq_data[i]['vl_gene'] == seq_data[j]['vl_gene'])
                            same_h3 = (seq_data[i]['h3_length'] == seq_data[j]['h3_length'])
                            if not (same_genes and same_h3):
                                D_fused[i, j] = 999.0
                    np.fill_diagonal(D_fused, 0.0)
                    Z = fastcluster.linkage(squareform(D_fused), method='complete')
                    
                    for t in thresholds:
                        labels = fcluster(Z, t=t, criterion='distance')
                        f1, tp, fp, fn, tn = evaluate_clusters(labels, is_binder_array)
                        if f1 > best_2d['f1']:
                            clusters = {}
                            for idx_c, c_id in enumerate(labels):
                                clusters.setdefault(c_id, []).append(idx_c)
                            best_2d = {'f1': f1, 'w': w, 't': t, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'clusters': clusters, 'method': 'Agglomerative (Fused Matrix)'}
                    
                    progress_bar.progress((idx_w + 1) / len(weights))
                    status_text.text(f"Evaluating Weight Fusions: {idx_w+1}/{len(weights)} | Peak F1: {best_2d['f1']:.4f}")
                
                winner = best_2d
                
            # --- METHODS 1 to 16 ---
            else:
                if "AI:" in methodology:
                    tokenizer, model = load_language_model()
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    model.to(device)
                    
                    embeddings = []
                    with torch.no_grad():
                        for s in seq_data:
                            if "L3+H3" in methodology: target = s['lm_l3h3']
                            elif "6-CDR" in methodology: target = s['lm_para']
                            else: target = s['lm_full']
                            
                            inputs = tokenizer(" ".join(list(target)), return_tensors="pt").to(device)
                            embeddings.append(model(**inputs).last_hidden_state.mean(dim=1).squeeze().cpu().numpy())
                    embeddings = np.array(embeddings)
                    
                    if "Structural" in methodology:
                        dist_ai = squareform(pdist(normalize(embeddings, norm='l2'), metric='euclidean'))
                    elif "PCA" in methodology:
                        dist_ai = cosine_distances(PCA(n_components=50).fit_transform(embeddings))
                    else:
                        dist_ai = cosine_distances(embeddings)
                        
                for i in range(n_seqs):
                    for j in range(i+1, n_seqs):
                        d_seq = Levenshtein.distance(seq_data[i]['str_l3h3'], seq_data[j]['str_l3h3']) / ((seq_data[i]['len_l3h3'] + seq_data[j]['len_l3h3']) / 2.0)
                        d_par = Levenshtein.distance(seq_data[i]['str_para'], seq_data[j]['str_para']) / ((seq_data[i]['len_para'] + seq_data[j]['len_para']) / 2.0)
                        
                        same_genes = (seq_data[i]['vh_gene'] == seq_data[j]['vh_gene']) and (seq_data[i]['vl_gene'] == seq_data[j]['vl_gene'])
                        same_h3 = (seq_data[i]['h3_length'] == seq_data[j]['h3_length'])
                        
                        if methodology == "1. NON-AI: Pure Sequence (L3+H3)":
                            dist_matrix[i, j] = dist_matrix[j, i] = d_seq
                        elif methodology == "2. NON-AI: Clonotype (L3+H3 + V-Gene Ban)":
                            dist_matrix[i, j] = dist_matrix[j, i] = d_seq if (same_genes and same_h3) else 999.0
                        elif methodology == "3. NON-AI: Paratope (6-CDR Proxy)":
                            dist_matrix[i, j] = dist_matrix[j, i] = d_par if same_h3 else 999.0
                        elif methodology == "4. NON-AI: Fusion (Sequence + Paratope)":
                            dist_matrix[i, j] = dist_matrix[j, i] = ((d_seq + d_par) / 2.0) if same_h3 else 999.0
                        elif methodology == "5. NON-AI: Fusion (Sequence + Clonotype Soft-Margin)":
                            if not same_h3: dist_matrix[i, j] = dist_matrix[j, i] = 999.0
                            else: dist_matrix[i, j] = dist_matrix[j, i] = d_seq if same_genes else d_seq + 0.30
                        elif methodology == "6. NON-AI: Fusion (Clonotype + Paratope Strict Ban)":
                            dist_matrix[i, j] = dist_matrix[j, i] = ((d_seq + d_par) / 2.0) if (same_genes and same_h3) else 999.0
                        elif methodology == "7. NON-AI: Grand Ensemble (Seq + Clono + Para)":
                            if not same_h3: dist_matrix[i, j] = dist_matrix[j, i] = 999.0
                            else: dist_matrix[i, j] = dist_matrix[j, i] = ((d_seq + d_par) / 2.0) if same_genes else ((d_seq + d_par) / 2.0) + 0.30
                        elif "AI:" in methodology:
                            dist_matrix[i, j] = dist_matrix[j, i] = dist_ai[i, j]
                            if "V-Gene Ban" in methodology and not (same_genes and same_h3): dist_matrix[i, j] = dist_matrix[j, i] = 999.0
                            if "No V-Gene" in methodology and not same_h3: dist_matrix[i, j] = dist_matrix[j, i] = 999.0

                unique_distances = set()
                for i in range(n_seqs):
                    for j in range(i+1, n_seqs):
                        if dist_matrix[i, j] < 900: unique_distances.add(round(dist_matrix[i, j], 3))
                
                thresholds = sorted(list(unique_distances))
                if not thresholds: thresholds = np.linspace(0.01, 1.0, 50)
                
                st.info("Executing topological sweep via 4-Algorithm Tournament...")
                winner = run_tournament(dist_matrix, thresholds, seq_data, idx_length_desc)

        # ---------------------------------------------------------
        # DISPLAY RESULTS
        # ---------------------------------------------------------
        st.markdown("---")
        st.subheader("🏆 Tournament Results")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Winning Topology", winner['method'])
        col2.metric("Maximum F1 Score", f"{winner['f1']:.4f}")
        
        if "17" in methodology:
            col3.metric("Optimal Blend", f"{winner['w']*100:.0f}% Text | {(1-winner['w'])*100:.0f}% AI (Thresh: {winner['t']:.2f})")
        else:
            display_t = winner['t'] if "AI:" in methodology else (1.0 - winner['t'])
            metric_name = "Distance Threshold" if "AI:" in methodology else "Identity Threshold"
            col3.metric(f"Optimal {metric_name}", f"{display_t:.4f}")
        
        st.markdown("### Error Vector Metrics")
        ecol1, ecol2, ecol3, ecol4 = st.columns(4)
        ecol1.metric("True Positives (TP)", winner['tp'])
        ecol2.metric("False Positives (FP)", winner['fp'])
        ecol3.metric("False Negatives (FN)", winner['fn'])
        ecol4.metric("True Negatives (TN)", winner['tn'])
        
        # ---------------------------------------------------------
        # DISPLAY ERROR PATTERNS
        # ---------------------------------------------------------
        st.markdown("---")
        st.subheader("📊 Cluster State Breakdown")
        
        cluster_stats = []
        for c_id, members in winner['clusters'].items():
            c_tp, c_fp, c_fn, c_tn = 0, 0, 0, 0
            binders_in_cluster = sum(1 for m in members if is_binder_array[m])
            for m in members:
                if is_binder_array[m]:
                    if binders_in_cluster > 1: c_tp += 1
                    else: c_fn += 1
                else:
                    if binders_in_cluster > 0: c_fp += 1
                    else: c_tn += 1
            cluster_stats.append({'Size': len(members), 'TP': c_tp, 'FP': c_fp, 'FN': c_fn, 'TN': c_tn})
            
        cluster_stats.sort(key=lambda x: x['Size'], reverse=True)
        multi_clusters = [c for c in cluster_stats if c['Size'] > 1]
        singletons = [c for c in cluster_stats if c['Size'] == 1]
        
        st.markdown("**Multi-Element Manifolds ($|C| > 1$)**")
        st.dataframe(pd.DataFrame(multi_clusters), use_container_width=True)
        
        st.markdown("**Isolated States (Singletons)**")
        singleton_fns = sum(c['FN'] for c in singletons)
        singleton_tns = sum(c['TN'] for c in singletons)
        
        st.info(f"**Total Isolated Sequences:** {len(singletons)} \n\n "
                f"• **False Negatives:** {singleton_fns} \n\n "
                f"• **True Negatives:** {singleton_tns}")

        st.success("Matrix computation and validation mapping complete.")
