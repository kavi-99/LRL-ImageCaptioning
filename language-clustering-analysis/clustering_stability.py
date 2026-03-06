import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

xm3600 = {
    "ar":"Arabic","bn":"Bengali","zh":"Chinese","cs":"Czech","da":"Danish",
    "nl":"Dutch","en":"English","fa":"Farsi","fi":"Finnish","fr":"French",
    "de":"German","el":"Greek","he":"Hebrew","hi":"Hindi","hu":"Hungarian",
    "id":"Indonesian","it":"Italian","ja":"Japanese","ko":"Korean","mi":"Maori",
    "no":"Norwegian","pl":"Polish","pt":"Portuguese","ro":"Romanian","ru":"Russian",
    "es":"Spanish","sv":"Swedish","sw":"Swahili","th":"Thai","tr":"Turkish",
    "te":"Telugu","uk":"Ukrainian","vi":"Vietnamese"
}
families = {
    "ar":"Semitic","he":"Semitic","fa":"Indo-Iranian","hi":"Indo-Iranian",
    "bn":"Indo-Iranian","te":"Dravidian","es":"Romance","fr":"Romance",
    "it":"Romance","pt":"Romance","ro":"Romance","de":"Germanic","nl":"Germanic",
    "sv":"Germanic","da":"Germanic","no":"Germanic","en":"Germanic","ru":"Slavic",
    "pl":"Slavic","cs":"Slavic","uk":"Slavic","fi":"Uralic","hu":"Uralic",
    "ja":"Japonic","ko":"Koreanic","zh":"Sino-Tibetan","th":"Tai-Kadai",
    "vi":"Austroasiatic","id":"Austronesian","mi":"Austronesian",
    "tr":"Turkic","sw":"Bantu","el":"Hellenic",
}

codes = list(xm3600.keys())
mat = np.load("m_knn_fallback.npy")
norms = np.linalg.norm(mat, axis=1, keepdims=True)
mat = mat / norms

cluster_colors = [
    "#e41a1c","#377eb8","#4daf4a","#ff7f00","#984ea3",
    "#a65628","#f781bf","#999999","#66c2a5","#d4ac0d",
]

def cdist(i1, i2):
    a, b = mat[i1], mat[i2]
    return round(1 - float(np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b))), 4)

def make_cluster_plot(labels, centroids, mat, k, title, fname):
    # labels, centroids = run_kmeans(mat, k)

    # t-SNE for visualisation
    tsne = TSNE(n_components=2, perplexity=8, random_state=42, max_iter=2000)
    coords = tsne.fit_transform(mat)

    fig, ax = plt.subplots(figsize=(13, 9))
    for i, code in enumerate(codes):
        color = cluster_colors[labels[i] % len(cluster_colors)]
        is_centroid = (i in centroids)
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

# ── k=8 clusters ─────────────────────────────────────────────────────────────
K = 8
km = KMeans(n_clusters=K, n_init=30, random_state=42)
labels = km.fit_predict(mat)

# Identify centroid language per cluster
cluster_info = {}
centroids = []
for c in range(K):
    members = [i for i, l in enumerate(labels) if l == c]
    mean = mat[members].mean(axis=0)
    dists_to_mean = [np.linalg.norm(mat[i] - mean) for i in members]
    centroid_idx = members[np.argmin(dists_to_mean)]
    centroids.append(centroid_idx)
    cluster_info[c] = {
        "members": members,
        "centroid": codes[centroid_idx],
        "mean": mean,
    }

make_cluster_plot(
    labels, centroids, mat, K,
    f"K-Means on syntax_average",
    f"kmeans_syn_k{K}.png"
)

# ── Stability: 15 seeds ───────────────────────────────────────────────────────
N_RUNS = 15
n = len(codes)
co = np.zeros((n, n))
for seed in range(N_RUNS):
    lbl = KMeans(n_clusters=K, n_init=30, random_state=seed).fit_predict(mat)
    for i in range(n):
        for j in range(n):
            if lbl[i] == lbl[j]:
                co[i, j] += 1
co /= N_RUNS

# ── Print: per-cluster within-cluster distances + stability ──────────────────
print(f"K-MEANS k={K} | hybrid (syntax_average + knn fallback) | L2-normalised")
print(f"Stability computed over {N_RUNS} random seeds\n")

for c in range(K):
    info = cluster_info[c]
    members = info["members"]
    ctr = info["centroid"]
    member_codes = [codes[i] for i in members]
    fams = sorted(set(families.get(m, "?") for m in member_codes))

    print(f"{'─'*70}")
    print(f"CLUSTER {c+1}  centroid: {xm3600[ctr]} ({ctr})")
    print(f"Families: {', '.join(fams)}")
    print(f"Members:  {', '.join(f'{xm3600[m]}({m})' for m in member_codes)}")

    # Within-cluster pairwise cosine distances
    if len(members) > 1:
        print(f"\n  Within-cluster pairwise cosine distances:")
        pairs_done = set()
        all_dists = []
        rows = []
        for i in members:
            for j in members:
                if i < j:
                    d = cdist(i, j)
                    all_dists.append(d)
                    rows.append((codes[i], codes[j], d))
        # Sort by distance
        rows.sort(key=lambda x: x[2])
        for c1, c2, d in rows:
            bar = "█" * int((1 - d) * 20)  # longer bar = more similar
            print(f"    {c1}–{c2:<6}  {d:.4f}  {bar}")
        print(f"  Mean intra-cluster distance: {np.mean(all_dists):.4f}  "
              f"(min: {min(all_dists):.4f}, max: {max(all_dists):.4f})")

    # Stability scores for all within-cluster pairs
    if len(members) > 1:
        print(f"\n  Stability scores (fraction of {N_RUNS} runs they co-cluster):")
        rows_stab = []
        for i in members:
            for j in members:
                if i < j:
                    rows_stab.append((codes[i], codes[j], co[i,j]))
        rows_stab.sort(key=lambda x: -x[2])
        for c1, c2, s in rows_stab:
            bar = "█" * int(s * 20)
            flag = "  ← unstable" if s < 0.7 else ""
            print(f"    {c1}–{c2:<6}  {s:.2f}  {bar}{flag}")
    print()

# ── Cross-cluster distances for pairs of interest ────────────────────────────
print(f"{'─'*70}")
print("CROSS-CLUSTER DISTANCES (pairs relevant to transfer hypotheses):")
print(f"  {'pair':<10}  {'cosine dist':>12}  {'same cluster?':>14}  note")
print(f"  {'─'*60}")
cross_pairs = [
    ("hi","te","Indo-Aryan→Dravidian"),
    ("bn","te","Bengali→Telugu"),
    ("tr","fa","Turkish→Farsi"),
    ("tr","hu","Turkish→Hungarian"),
    ("en","de","English→German"),
    ("es","ro","Spanish→Romanian"),
    ("en","te","English→Telugu (max gap)"),
    ("en","tr","English→Turkish"),
]
for c1, c2, note in cross_pairs:
    i1, i2 = codes.index(c1), codes.index(c2)
    d = cdist(i1, i2)
    same = "YES" if labels[i1] == labels[i2] else "no"
    stab = co[i1, i2]
    bar = "█" * int(stab * 10)
    print(f"  {c1}–{c2:<6}  {d:>12.4f}  {same:>14}  stab={stab:.2f} {bar}  {note}")