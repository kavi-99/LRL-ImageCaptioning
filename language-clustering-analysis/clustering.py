# clustering.py

import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

xm3600 = {
    "ar":"Arabic","bn":"Bengali","zh":"Chinese",
    "cs":"Czech","da":"Danish","nl":"Dutch","en":"English","fa":"Farsi",
    "fi":"Finnish","fr":"French","de":"German",
    "el":"Greek","he":"Hebrew","hi":"Hindi","hu":"Hungarian","id":"Indonesian",
    "it":"Italian","ja":"Japanese","ko":"Korean","mi":"Maori","no":"Norwegian",
    "pl":"Polish","pt":"Portuguese","ro":"Romanian","ru":"Russian","es":"Spanish",
    "sv":"Swedish","sw":"Swahili","th":"Thai","tr":"Turkish","te":"Telugu",
    "uk":"Ukrainian","vi":"Vietnamese",
}
families = {
    "ar":"Semitic","he":"Semitic","fa":"Indo-Iranian","hi":"Indo-Iranian",
    "bn":"Indo-Iranian","te":"Dravidian","es":"Romance","fr":"Romance",
    "it":"Romance","pt":"Romance","ro":"Romance","de":"Germanic","nl":"Germanic",
    "sv":"Germanic","da":"Germanic","no":"Germanic","en":"Germanic","ru":"Slavic",
    "pl":"Slavic","cs":"Slavic","uk":"Slavic","fi":"Uralic",
    "hu":"Uralic","ja":"Japonic","ko":"Koreanic","zh":"Sino-Tibetan",
    "th":"Tai-Kadai","vi":"Austroasiatic","id":"Austronesian",
    "mi":"Austronesian","tr":"Turkic","sw":"Bantu","el":"Hellenic",
}

codes = list(xm3600.keys())
m_syn  = np.load("m_knn_fallback.npy")

# Run k-means for k=6 and k=10 on both matrices
# k=6: coarse grouping (prior: ~6 macro-areas)
# k=10: finer grouping (prior: between elbow=3 and n_families=16)
cluster_colors = [
    "#e41a1c","#377eb8","#4daf4a","#ff7f00","#984ea3",
    "#a65628","#f781bf","#999999","#66c2a5","#d4ac0d",
]

def run_kmeans(mat, k, seed=42):
    km = KMeans(n_clusters=k, n_init=30, random_state=seed)
    labels = km.fit_predict(mat)
    # centroid language = closest to cluster mean
    centroids = []
    for c in range(k):
        members = [i for i, l in enumerate(labels) if l == c]
        cluster_mean = mat[members].mean(axis=0)
        dists = [np.linalg.norm(mat[i] - cluster_mean) for i in members]
        centroids.append(codes[members[np.argmin(dists)]])
    return labels, centroids

def stability_check(mat, k, n_runs=10):
    """Check how often each pair of languages ends up in the same cluster."""
    n = len(codes)
    co_occur = np.zeros((n, n))
    for seed in range(n_runs):
        labels, _ = run_kmeans(mat, k, seed=seed)
        for i in range(n):
            for j in range(n):
                if labels[i] == labels[j]:
                    co_occur[i, j] += 1
    return co_occur / n_runs

def make_cluster_plot(mat, k, title, fname):
    labels, centroids = run_kmeans(mat, k)

    # t-SNE for visualisation
    tsne = TSNE(n_components=2, perplexity=8, random_state=42, n_iter=2000)
    coords = tsne.fit_transform(mat)

    fig, ax = plt.subplots(figsize=(13, 9))
    for i, code in enumerate(codes):
        color = cluster_colors[labels[i] % len(cluster_colors)]
        is_centroid = (code in centroids)
        marker = '*' if is_centroid else 'o'
        size   = 220 if is_centroid else 80
        ax.scatter(coords[i,0], coords[i,1], c=color, s=size,
                   marker=marker, zorder=3, edgecolors='black' if is_centroid else 'white',
                   linewidths=1.2 if is_centroid else 0.4)
        ax.annotate(
            f"{xm3600[code]}\n({code})",
            (coords[i,0], coords[i,1]),
            fontsize=7.5, ha='center', va='bottom',
            xytext=(0, 7), textcoords='offset points',
            color=color,
            fontweight='bold' if is_centroid else 'normal',
        )

    ax.set_title(f"{title}\nk={k}  ★ = centroid language per cluster",
                 fontsize=12, fontweight='bold')
    ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    return labels, centroids

def print_clusters(labels, centroids, k, title):
    print(f"\n{'='*60}")
    print(f"{title}  (k={k})")
    print(f"{'='*60}")
    for c in range(k):
        members = [codes[i] for i, l in enumerate(labels) if l == c]
        member_names = [f"{xm3600[m]}({m})" for m in members]
        centroid = centroids[c]
        fams = list(set(families.get(m,"?") for m in members))
        print(f"\n  Cluster {c+1}  [centroid: {xm3600[centroid]}({centroid})]")
        print(f"  Families: {', '.join(sorted(fams))}")
        print(f"  Members:  {', '.join(member_names)}")

# ── Run for k=6 and k=10 on syntax-only ──────────────────────────────────────
for k in [6, 8, 10]:
    labels, centroids = make_cluster_plot(
        m_syn, k,
        f"K-Means on syntax_average",
        f"kmeans_syn_k{k}.png"
    )
    print_clusters(labels, centroids, k, "syntax_average")

# ── Stability check for k=6 syntax-only ─────────────────────────────────────
print("\n\nStability check (k=6, syntax_only, 10 random seeds):")
print("Checking your key pairs — score = fraction of runs they co-cluster:")
co = stability_check(m_syn, 6)
key_pairs = [
    ("es","ro"), ("hi","bn"), ("hi","te"),
    ("te","tr"), ("id","vi"), ("en","es"),
]
for c1, c2 in key_pairs:
    i1, i2 = codes.index(c1), codes.index(c2)
    score = co[i1,i2]
    bar = "█" * int(score * 20)
    print(f"  {c1}–{c2}  {score:.2f}  {bar}")