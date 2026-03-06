# build_matrix.py

import lang2vec.lang2vec as l2v
import numpy as np

xm3600_codes = [
    "ar","bn","zh","cs","da","nl","en","fa",
    "fi","fr","de","el","he","hi","hu","id",
    "it","ja","ko","mi","no","pl","pt","ro",
    "ru","es","sv","sw","th","tr","te","uk","vi"
]
# Note: Filipino (fil) and Croatian (hr) removed — 0 observed features in all sources

def get_raw(lang, fs):
    return l2v.get_features(lang, fs)[lang]

def build_matrix_knn_fallback(codes):
    """
    For each language and each dimension:
      - Use syntax_average value if it has a real observation (not --)
      - Fall back to syntax_knn value if syntax_average is --
    No column mean imputation at all.
    """
    mat = []
    fallback_counts = {}
    for lang in codes:
        avg = get_raw(lang, "syntax_average")
        knn = get_raw(lang, "syntax_knn")
        vec = []
        n_fallback = 0
        for a, k in zip(avg, knn):
            if a == '--':
                vec.append(float(k))   # knn fallback
                n_fallback += 1
            else:
                vec.append(float(a))   # real observed value
        mat.append(vec)
        fallback_counts[lang] = n_fallback

    mat = np.array(mat)
    assert not np.isnan(mat).any(), "NaNs remain after knn fallback — unexpected"
    return mat, fallback_counts

codes = xm3600_codes
mat, fallback_counts = build_matrix_knn_fallback(codes)

print(f"Matrix shape: {mat.shape}  (no NaNs: {not np.isnan(mat).any()})")
print()
print("Fallback dimensions used per language (syntax_average -- → syntax_knn):")
print(f"  {'Language':12s}  {'observed':>10}  {'knn_fallback':>12}")
print("  " + "-"*38)
for lang in codes:
    fb = fallback_counts[lang]
    obs = 103 - fb
    bar = "█" * (fb // 3)
    print(f"  {lang:12s}  {obs:>10}/103  {fb:>8}/103  {bar}")

np.save("m_knn_fallback.npy", mat)
# np.save("codes.npy", np.array(codes))
print("\nSaved.")