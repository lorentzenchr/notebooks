"""
Benchmark script for HistGradientBoostingClassifier.

Output written as "bench_loss_module_hgbt.parquet"
"""
from collections import OrderedDict
from neurtu import delayed, Benchmark
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils._openmp_helpers import _openmp_parallelism_enabled, _openmp_effective_n_threads


print("openmp enabled: ", _openmp_parallelism_enabled())
print("openmp threads: ", _openmp_effective_n_threads())
n_threads = _openmp_effective_n_threads()


n_samples, n_features = 100_000, 20
n_informative = int(n_features * 0.9)
bench_options = {
    "wall_time": True,
    "cpu_time": True,
    "peak_memory": True,
    "repeat": 20,
}
early_stopping = [True, False]
options = {}


def benchmark_cases(X, y):
    for N in np.logspace(
            np.log10(n_samples/1e3), np.log10(n_samples), 4
    ).astype('int'):
        for es in early_stopping:
            tags = OrderedDict(N=N, early_stopping=es)
            clf = HistGradientBoostingClassifier(**options)
            yield delayed(clf.fit, tags=tags)(X[:N, :], y[:N])

            
# 1. Binary Histogram Gradient Booster
n_classes = 2
X, y = make_classification(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=n_informative,
    n_classes=n_classes,
    random_state=42,
)

bench = Benchmark(**bench_options)
print("Run binary histogram gradient booster.")
df_hgbt_binary = bench(benchmark_cases(X, y))
df_hgbt_binary["estimator"] = "HistGradientBoostingClassifier"
df_hgbt_binary["n_classes"] = n_classes


# 2. Multiclass Histogram Gradient Booster
n_classes = 10
X, y = make_classification(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=n_informative,
    n_classes=n_classes,
    random_state=42,
)
            
bench = Benchmark(**bench_options)
print("Run multiclass histogram gradient booster.")
df_hgbt_multi = bench(benchmark_cases(X, y))
df_hgbt_multi["estimator"] = "HistGradientBoostingClassifier"
df_hgbt_multi["n_classes"] = n_classes


# 3. Save Results
print("Save benchmark results from HistGradientBoostingClassifier.")
df = pd.concat([df_hgbt_binary, df_hgbt_multi])
df["n_threads"] = n_threads
df.to_parquet("bench_loss_module_hgbt.parquet")
