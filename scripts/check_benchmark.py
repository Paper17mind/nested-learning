"""Benchmark HOPE-lite pada laptop kamu: berapa d_model yang realistis, berapa thread BLAS
yang paling cepat, dan di mana waktu/RAM terbuang.

Taruh di folder `scripts/` repo (atau jalankan dengan --repo /path/ke/nested-learning), lalu:

    python scripts/benchmark_hardware.py                      # sweep standar
    python scripts/benchmark_hardware.py --data-mb 20 --target-minutes 30
    python scripts/benchmark_hardware.py --dims 64 128 256 --seq-len 128 --batch 8
    python scripts/benchmark_hardware.py --ram-gb 8           # kalau RAM tidak terdeteksi

Hanya memakai stdlib + NumPy. Tiap konfigurasi dijalankan di subprocess terpisah karena jumlah
thread BLAS harus diset SEBELUM numpy di-import, dan supaya peak RAM terukur bersih.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

# --------------------------------------------------------------------------- util sistem


def detect_ram_gb() -> float | None:
    """Total RAM fisik (GB) tanpa dependensi eksternal."""
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        return int(line.split()[1]) / 1024 / 1024
        elif sys.platform == "darwin":
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            return int(out.strip()) / 1024**3
        elif sys.platform == "win32":
            import ctypes

            class MEMSTAT(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            st = MEMSTAT()
            st.dwLength = ctypes.sizeof(MEMSTAT)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return st.ullTotalPhys / 1024**3
    except Exception:
        pass
    return None


def detect_available_ram_gb() -> float | None:
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable"):
                        return int(line.split()[1]) / 1024 / 1024
    except Exception:
        pass
    return None


def cpu_name() -> str:
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        elif sys.platform == "darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def est_cache_mb(batch: int, seq_len: int, d: int) -> float:
    """Perkiraan RAM utama: memory_history = (T+1) snapshot matriks B x D x D float32,
    disimpan sepanjang forward->backward di SelfModifyingLayer."""
    return batch * (seq_len + 1) * d * d * 4 / 1024**2


# --------------------------------------------------------------------------- worker


def worker(cfg: dict) -> None:
    """Dijalankan di subprocess: ukur waktu per step dan peak RSS."""
    if cfg.get("repo"):
        sys.path.insert(0, cfg["repo"])
    import numpy as np
    import nested_learning as nl

    np.random.seed(0)
    d, T, B = cfg["d_model"], cfg["seq_len"], cfg["batch"]
    n_layers = cfg["n_layers"]
    # tier: separuh lapisan cepat (period 1), separuh lambat (period 4) seperti contoh README
    fast = n_layers // 2
    tiers = [[fast, 1], [n_layers - fast, 4]] if n_layers > 1 else [[n_layers, 1]]

    tok = nl.ByteTokenizer()
    model = nl.HOPE(vocab_size=tok.vocab_size, d_model=d, n_layers=n_layers, cms_tiers=tiers)
    loss_fn = nl.CrossEntropyLoss(ignore_index=tok.pad_token_id)
    opt = nl.NestedOptimizer(model.tier_param_groups(), lr=5e-3, optimizer_cls=nl.AdamW)

    if cfg.get("data"):
        ids = np.array(tok.encode(Path(cfg["data"]).read_text(encoding="utf-8", errors="ignore")[:2_000_000]))
        ds = nl.TextDataset(ids, seq_len=T, pad_token_id=tok.pad_token_id)
        batches = [b for _, b in zip(range(cfg["steps"] + 1), ds.get_batches(B))]
        batches = [b for b in batches if b[0].shape[0] == B] or None
    else:
        batches = None
    if batches is None:
        x = np.random.randint(1, tok.vocab_size, size=(B, T)).astype(np.int64)
        y = np.roll(x, -1, axis=1)
        m = np.ones((B, T), dtype=np.float32)
        batches = [(x, y, m)] * (cfg["steps"] + 1)

    # pembungkus waktu untuk memisahkan biaya fast-memory (loop per-token) vs sisanya
    acc = {"fast_fwd": 0.0, "fast_bwd": 0.0}
    fm = model.fast_memory
    of, ob = fm.forward, fm.backward

    def tf(*a, **k):
        t0 = time.perf_counter()
        r = of(*a, **k)
        acc["fast_fwd"] += time.perf_counter() - t0
        return r

    def tb(*a, **k):
        t0 = time.perf_counter()
        r = ob(*a, **k)
        acc["fast_bwd"] += time.perf_counter() - t0
        return r

    fm.forward, fm.backward = tf, tb

    def one_step(batch):
        x, y, m = batch
        opt.zero_grad()
        t0 = time.perf_counter()
        logits, _ = model.forward(x, mask=m)
        t1 = time.perf_counter()
        loss = loss_fn.forward(logits, y)
        dl = loss_fn.backward()
        model.backward(dl)
        t2 = time.perf_counter()
        opt.step()
        t3 = time.perf_counter()
        return float(loss), t1 - t0, t2 - t1, t3 - t2

    one_step(batches[0])  # warmup
    acc["fast_fwd"] = acc["fast_bwd"] = 0.0
    times, fwd, bwd, ost, losses = [], [], [], [], []
    for i in range(cfg["steps"]):
        t0 = time.perf_counter()
        loss, f, b, o = one_step(batches[1 + i])
        times.append(time.perf_counter() - t0)
        fwd.append(f); bwd.append(b); ost.append(o); losses.append(loss)

    peak_mb = None
    try:
        import resource

        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_mb = r / 1024 / 1024 if sys.platform == "darwin" else r / 1024
    except Exception:
        try:
            import psutil  # opsional

            peak_mb = psutil.Process().memory_info().rss / 1024**2
        except Exception:
            pass

    n = cfg["steps"]
    step_s = statistics.median(times)
    out = {
        "params": model.count_parameters(),
        "step_s": step_s,
        "tok_s": B * T / step_s,
        "fwd_s": statistics.median(fwd),
        "bwd_s": statistics.median(bwd),
        "opt_s": statistics.median(ost),
        "fast_share": (acc["fast_fwd"] + acc["fast_bwd"]) / n / step_s,
        "peak_mb": peak_mb,
        "loss": losses[-1],
    }
    print("RESULT " + json.dumps(out))


# --------------------------------------------------------------------------- orchestrator


def run_cfg(cfg: dict, threads: int, timeout: int) -> dict | None:
    env = dict(os.environ)
    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[k] = str(threads)
    cmd = [sys.executable, os.path.abspath(__file__), "--worker", json.dumps(cfg)]
    try:
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"error": f"timeout >{timeout}s"}
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    return {"error": (p.stderr.strip().splitlines() or ["unknown error"])[-1][:120]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    ap.add_argument("--repo", default=None, help="path repo nested-learning (default: induk folder script)")
    ap.add_argument("--dims", type=int, nargs="+", default=[32, 64, 128, 256])
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=3, help="step terukur per konfigurasi (median)")
    ap.add_argument("--threads", type=int, nargs="+", default=None, help="default: 1 dan jumlah core")
    ap.add_argument("--data", default=None, help="file teks asli (opsional; default data acak)")
    ap.add_argument("--data-mb", type=float, default=20.0, help="ukuran korpus latih untuk estimasi waktu")
    ap.add_argument("--target-minutes", type=float, default=30.0, help="batas waktu 1 epoch yang masih nyaman")
    ap.add_argument("--ram-gb", type=float, default=None, help="override RAM total jika tidak terdeteksi")
    ap.add_argument("--ram-budget", type=float, default=0.6, help="porsi RAM yang boleh dipakai (default 60%%)")
    ap.add_argument("--timeout", type=int, default=300)
    a = ap.parse_args()

    if a.worker:
        worker(json.loads(a.worker))
        return

    repo = a.repo or str(Path(__file__).resolve().parents[1])
    if not (Path(repo) / "nested_learning").is_dir():
        sys.exit(f"Folder nested_learning tidak ada di {repo}. Pakai --repo /path/ke/nested-learning")

    ncpu = os.cpu_count() or 1
    ram = a.ram_gb or detect_ram_gb()
    avail = detect_available_ram_gb()
    threads_list = a.threads or sorted({1, ncpu})

    print("=" * 78)
    print(f"CPU     : {cpu_name()}  ({ncpu} logical core)")
    print(f"RAM     : {f'{ram:.1f} GB total' if ram else 'tidak terdeteksi (pakai --ram-gb)'}"
          f"{f', {avail:.1f} GB tersedia' if avail else ''}")
    print(f"Python  : {platform.python_version()}  | OS: {platform.system()} {platform.release()}")
    try:
        import numpy as np

        print(f"NumPy   : {np.__version__}")
        cfgd = np.show_config(mode="dicts") if hasattr(np, "show_config") else {}
        blas = (cfgd or {}).get("Build Dependencies", {}).get("blas", {})
        if blas:
            print(f"BLAS    : {blas.get('name', '?')} {blas.get('version', '')}")
    except Exception:
        pass
    print(f"Setelan : layers={a.layers} seq_len={a.seq_len} batch={a.batch} (1 byte = 1 token)")
    print("=" * 78)

    budget_mb = (ram * 1024 * a.ram_budget) if ram else None
    rows = []
    hdr = f"{'d_model':>7} {'param':>9} {'thr':>3} {'ms/step':>8} {'tok/s':>7} {'fast%':>6} {'RAM est':>8} {'RAM peak':>9}"
    print(hdr)
    print("-" * len(hdr))
    for d in a.dims:
        est = est_cache_mb(a.batch, a.seq_len, d)  # cache hanya ada di 1 layer fast-memory
        if budget_mb and est * 2.5 > budget_mb:  # x2.5 untuk grad, aktivasi, optimizer
            print(f"{d:>7} {'-':>9} {'-':>3}  dilewati: estimasi RAM ~{est * 2.5:.0f} MB > anggaran {budget_mb:.0f} MB")
            continue
        for thr in threads_list:
            cfg = dict(repo=repo, d_model=d, seq_len=a.seq_len, batch=a.batch, n_layers=a.layers,
                       steps=a.steps, data=a.data)
            r = run_cfg(cfg, thr, a.timeout)
            if r is None or "error" in r:
                print(f"{d:>7} {'-':>9} {thr:>3}  GAGAL: {(r or {}).get('error')}")
                continue
            r.update(d=d, thr=thr, est=est)
            rows.append(r)
            peak = f"{r['peak_mb']:.0f}MB" if r["peak_mb"] else "n/a"
            print(f"{d:>7} {r['params']:>9,} {thr:>3} {r['step_s'] * 1000:>8.0f} {r['tok_s']:>7.0f} "
                  f"{r['fast_share'] * 100:>5.0f}% {est:>6.0f}MB {peak:>9}")

    if not rows:
        return

    # ------------------------------------------------------------------ analisis
    print("\n" + "=" * 78)
    print("ANALISIS")
    print("=" * 78)

    # 1. thread terbaik per d_model
    best = {}
    for r in rows:
        if r["d"] not in best or r["tok_s"] > best[r["d"]]["tok_s"]:
            best[r["d"]] = r
    if len(threads_list) > 1:
        for d, r in sorted(best.items()):
            one = next((x for x in rows if x["d"] == d and x["thr"] == 1), None)
            if one and r["thr"] != 1:
                print(f"d={d}: {r['thr']} thread = {r['tok_s'] / one['tok_s']:.2f}x dibanding 1 thread")
            elif one:
                print(f"d={d}: 1 thread sudah paling cepat (multi-thread tidak membantu / malah overhead)")

    # 2. di mana waktu habis
    r0 = best[max(best)]
    print(f"\nPembagian waktu di d={r0['d']}: forward {r0['fwd_s'] * 1000:.0f} ms | backward {r0['bwd_s'] * 1000:.0f} ms"
          f" | optimizer {r0['opt_s'] * 1000:.0f} ms")
    print(f"Layer fast-memory (loop per-token) memakan ~{r0['fast_share'] * 100:.0f}% dari total waktu step.")
    if r0["fast_share"] > 0.5:
        print("  -> Bottleneck = loop Python per token (T iterasi x banyak matmul kecil), bukan ukuran matriks.")
        print("     Mengurangi --seq-len atau menaikkan --batch biasanya lebih berpengaruh daripada tuning BLAS.")

    # 3. rekomendasi ukuran
    tokens = a.data_mb * 1024 * 1024
    print(f"\nEstimasi waktu 1 epoch untuk korpus {a.data_mb:g} MB (~{tokens / 1e6:.0f}M token), thread terbaik:")
    ok = []
    for d, r in sorted(best.items()):
        mins = tokens / r["tok_s"] / 60
        flag = "OK " if mins <= a.target_minutes else "   "
        print(f"  {flag}d_model={d:<4} {mins:>8.1f} menit/epoch  ({r['tok_s']:.0f} tok/s, {r['thr']} thread)")
        if mins <= a.target_minutes:
            ok.append(d)
    if ok:
        print(f"\nRekomendasi: d_model={max(ok)} masih muat dalam {a.target_minutes:g} menit/epoch.")
    else:
        smallest = min(best)
        need = tokens / best[smallest]["tok_s"] / 60
        print(f"\nTidak ada ukuran yang muat dalam {a.target_minutes:g} menit/epoch untuk {a.data_mb:g} MB. "
              f"Kecilkan korpus (d={smallest} butuh ~{need:.0f} menit) atau longgarkan --target-minutes.")

    print("\nCatatan: RAM naik ~ batch x seq_len x d_model^2 (riwayat memori disimpan untuk backprop).")
    print("Menggandakan d_model => RAM utama 4x. Untuk d_model besar, turunkan --batch atau --seq-len dulu.")


if __name__ == "__main__":
    main()