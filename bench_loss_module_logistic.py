"""
Benchmark script for Logistic Regression.

Output written as "bench_loss_module_logistic.parquet"
"""
from collections import OrderedDict
import time
from neurtu import delayed, Benchmark
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.utils._openmp_helpers import _openmp_parallelism_enabled, _openmp_effective_n_threads


if __name__ == "__main__":
    print("openmp enabled: ", _openmp_parallelism_enabled())
    print("openmp threads: ", _openmp_effective_n_threads())
    n_threads = _openmp_effective_n_threads()


    n_samples, n_features = 100_000, 50
    n_informative = int(n_features * 0.9)
    bench_options = {
        "wall_time": True,
        "cpu_time": True,
        "peak_memory": True,
        "repeat": 10,
    }
    options = {"max_iter": 1000}
    solvers = ["lbfgs", "newton-cg"]


    def benchmark_cases(X, y):
        for N in np.logspace(
                np.log10(n_samples/1e3), np.log10(n_samples), 4
        ).astype('int'):
            for solver in solvers:
                tags = OrderedDict(N=N, solver=solver)
                clf = LogisticRegression(solver=solver, **options)
                yield delayed(clf.fit, tags=tags)(X[:N, :], y[:N])

                
    # 1. Binary Logistic Regression
    n_classes = 2
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_classes=n_classes,
        random_state=42,
    )

    bench = Benchmark(**bench_options)
    print("Run binary logistic regression.")
    start = time.time()
    df_logistic_binary = bench(benchmark_cases(X, y))
    end =  time.time()
    print(f"\tDone in {end - start} seconds.")
    df_logistic_binary["estimator"] = "LogisticRegression"
    df_logistic_binary["n_classes"] = n_classes


    # 2. Multiclass Logistic Regression
    n_classes = 10
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_classes=n_classes,
        random_state=42,
    )
                
    bench = Benchmark(**bench_options)
    print("Run multiclass logistic regression.")
    start = time.time()
    df_logistic_multi = bench(benchmark_cases(X, y))
    end =  time.time()
    print(f"\tDone in {end - start} seconds.")
    df_logistic_multi["estimator"] = "LogisticRegression"
    df_logistic_multi["n_classes"] = n_classes


    # 3. Save Results
    print("Save benchmark results from LogisticRegression.")
    df = pd.concat([df_logistic_binary, df_logistic_multi])
    df["n_threads"] = n_threads
    df.to_parquet("bench_loss_module_logistic.parquet")
