"""
plot_vmmul.py -- CSC 746 CP3 results: CSVs and charts from benchmark logs.
Adapted from plot_mmul.py (E. Wes Bethel, 2022).

Usage (in the build directory):
    ./benchmark-basic      > basic.log
    ./benchmark-vectorized > vectorized.log
    ./benchmark-blas       > blas.log
    bash ./job-openmp      > openmp.log
    python ../plot_vmmul.py            # or: python ../plot_vmmul.py a.log b.log

Outputs:
    runtimes.csv, mflops.csv, speedup.csv, bandwidth_pct.csv
    chart1_mflops_basic_vect_blas.png    Chart #1
    chart2_speedup_openmp.png            Chart #2
    chart3_mflops_bestomp_vs_blas.png    Chart #3
"""

import csv
import glob
import math
import re
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PLATFORM = ("Perlmutter CPU node: AMD EPYC 7763 (Milan), 2.45 GHz, "
            "GCC, OMP_PLACES=threads, OMP_PROC_BIND=spread, static schedule")

PEAK_MFLOPS = 39200.0          # per core: 2.45 GHz x 16 DP FLOPs/cycle
PEAK_BW_GBS = 204.8            # per socket: 8 x DDR4-3200
SPEEDUP_BASE = "omp-1"         # or "basic"


def flops(n):
    return 2.0 * n * n


def bytes_moved(n):
    return 8.0 * (n * n + 2.0 * n)


OMP_NAMES = ["omp-1", "omp-4", "omp-16", "omp-64"]
ALL_NAMES = ["cblas", "basic", "vectorized"] + OMP_NAMES

LABELS = {"cblas": "CBLAS", "basic": "Basic", "vectorized": "Vectorized",
          "omp-1": "OpenMP, 1 thread", "omp-4": "OpenMP, 4 threads",
          "omp-16": "OpenMP, 16 threads", "omp-64": "OpenMP, 64 threads"}

STYLES = {"cblas": "k-o", "basic": "r-x", "vectorized": "g-^",
          "omp-1": "r-o", "omp-4": "b-x", "omp-16": "c-s", "omp-64": "g-^"}

RE_DESC = re.compile(r"^\s*Description:\s*(.*)$")
RE_THREADS = re.compile(r"OMP_NUM_THREA\w*\s*=\s*(\d+)")
RE_TIME = re.compile(r"N\s*=\s*(\d+).*?elapsed time[^=]*=\s*([0-9.eE+-]+)")


def config_name(desc, threads):
    d = desc.lower()
    if "openmp" in d:
        return "omp-%d" % threads if threads else None
    if "vectorized" in d:
        return "vectorized"
    if "basic" in d:
        return "basic"
    if "reference" in d:
        return "cblas"
    return None


def parse_logs(paths):
    """Return {config: {N: [seconds, ...]}}, skipping the warm-up N=1024 run."""
    data = {}
    for path in paths:
        threads, config, warmup_done = None, None, False
        with open(path) as f:
            for line in f:
                m = RE_THREADS.search(line)
                if m:
                    threads = int(m.group(1))
                    continue
                m = RE_DESC.match(line)
                if m:
                    config = config_name(m.group(1), threads)
                    warmup_done = False
                    continue
                if "Error:" in line:
                    print("WARNING: correctness check failed in %s: %s"
                          % (path, line.strip()))
                    continue
                m = RE_TIME.search(line)
                if m and config:
                    n, t = int(m.group(1)), float(m.group(2))
                    if n == 1024 and not warmup_done:
                        warmup_done = True
                        continue
                    data.setdefault(config, {}).setdefault(n, []).append(t)
    return data


def median_times(data):
    sizes = sorted({n for d in data.values() for n in d})
    times = {k: [statistics.median(data[k][n]) for n in sizes]
             for k in ALL_NAMES if k in data}
    return sizes, times


def geomean(values):
    return math.exp(sum(math.log(v) for v in values) / len(values))


def write_csv(fname, sizes, series, fmt):
    names = [k for k in ALL_NAMES if k in series]
    with open(fname, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N"] + names)
        for i, n in enumerate(sizes):
            w.writerow([n] + [fmt % series[k][i] for k in names])
    print("wrote " + fname)


def make_chart(fname, title, ylabel, sizes, series, names, peak=False):
    plt.figure(figsize=(11.5, 6.5))
    plt.title(title, fontsize=13)
    xlocs = list(range(len(sizes)))
    plt.xticks(xlocs, ["%d\n(%.0f MiB)" % (n, bytes_moved(n) / 2**20)
                       for n in sizes], fontsize=9)

    legend = []
    for k in names:
        plt.plot(xlocs, series[k], STYLES[k], linewidth=1.8, markersize=7)
        legend.append(LABELS[k])
    if peak:
        plt.axhline(y=PEAK_MFLOPS, color="k", linestyle="--", linewidth=1.8)
        legend.append("AMD EPYC 7763 per-core peak = %.1f GFLOP/s"
                      % (PEAK_MFLOPS / 1000))
    plt.ylim(bottom=0)

    plt.xlabel("Problem size N (A + x + y memory footprint)")
    plt.ylabel(ylabel)
    plt.legend(legend, loc="upper left", bbox_to_anchor=(1.02, 1.0),
               borderaxespad=0.0, fontsize=10)
    plt.grid(axis="both")
    plt.figtext(0.42, 0.03, PLATFORM, ha="center", fontsize=8, style="italic")
    plt.subplots_adjust(left=0.08, right=0.74, top=0.90, bottom=0.17)
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print("wrote " + fname)


def print_table(title, sizes, series, fmt):
    print("\n" + title)
    print("  %-12s" % "" + "".join("%10d" % n for n in sizes))
    for k in [k for k in ALL_NAMES if k in series]:
        print("  %-12s" % k + "".join(fmt % v for v in series[k]))


def main():
    paths = sys.argv[1:] or sorted(glob.glob("*.log"))
    data = parse_logs(paths)
    if not data:
        sys.exit("no timing data found in: " + ", ".join(paths))

    sizes, times = median_times(data)
    missing = [k for k in ALL_NAMES if k not in times]
    if missing:
        print("warning: missing configurations: " + ", ".join(missing))

    mflops = {k: [flops(n) / t / 1e6 for n, t in zip(sizes, v)]
              for k, v in times.items()}
    bw = {k: [100.0 * bytes_moved(n) / t / 1e9 / PEAK_BW_GBS
              for n, t in zip(sizes, v)]
          for k, v in times.items()}
    speedup = {k: [b / t for b, t in zip(times[SPEEDUP_BASE], times[k])]
               for k in OMP_NAMES if k in times} if SPEEDUP_BASE in times else {}
    omp = [k for k in OMP_NAMES if k in mflops]
    best = max(omp, key=lambda k: geomean(mflops[k])) if omp else None

    write_csv("runtimes.csv", sizes, times, "%.6f")
    write_csv("mflops.csv", sizes, mflops, "%.1f")
    write_csv("speedup.csv", sizes, speedup, "%.3f")
    write_csv("bandwidth_pct.csv", sizes, bw, "%.2f")

    serial = [k for k in ["cblas", "basic", "vectorized"] if k in mflops]
    if serial:
        make_chart("chart1_mflops_basic_vect_blas.png",
                   "VMM MFLOP/s: CBLAS vs Basic vs Vectorized",
                   "MFLOP/s", sizes, mflops, serial, peak=True)
    if speedup:
        make_chart("chart2_speedup_openmp.png",
                   "VMM OpenMP Speedup (static schedule)",
                   "Speedup = T(%s) / T(P threads)" % SPEEDUP_BASE,
                   sizes, speedup, list(speedup))
    if best and "cblas" in mflops:
        make_chart("chart3_mflops_bestomp_vs_blas.png",
                   "VMM MFLOP/s: Best OpenMP (%s threads) vs Serial CBLAS"
                   % best.split("-")[1],
                   "MFLOP/s", sizes, mflops, ["cblas", best])

    print_table("MFLOP/s", sizes, mflops, "%10.1f")
    if speedup:
        print_table("Speedup vs %s" % SPEEDUP_BASE, sizes, speedup, "%10.2f")
        eff = {k: [s / int(k.split("-")[1]) for s in v]
               for k, v in speedup.items()}
        print_table("Parallel efficiency (1.00 = linear)", sizes, eff, "%10.2f")
    print_table("%% of peak memory bandwidth (%.1f GB/s)" % PEAK_BW_GBS,
                sizes, bw, "%9.2f%%")
    if best:
        print("\nBest OpenMP configuration (Chart #3): " + best)


if __name__ == "__main__":
    main()