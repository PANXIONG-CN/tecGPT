import os
import json
import shutil
import time
from pathlib import Path


def register_iri(src_dir: str, dst_root: str = "outputs/GIMtec/IRI_2020/seed_external") -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = Path(dst_root) / ts
    out.mkdir(parents=True, exist_ok=True)
    for name in ["predictions.npy", "metrics.json", "manifest.json", "stepwise_rmse.csv"]:
        src = Path(src_dir) / name
        if src.exists():
            shutil.copy2(src, out / src.name)
    # ensure manifest
    mpath = out / "manifest.json"
    if mpath.exists():
        try:
            m = json.load(open(mpath))
        except Exception:
            m = {}
    else:
        m = {}
    m.update({
        "dataset": "GIMtec",
        "dataset_slug": "gim",
        "model": "IRI_2020",
        "units": "TECU",
        "normalized": False,
    })
    json.dump(m, open(mpath, "w"), indent=2)
    print(f"IRI-2020 registered to: {out}")
    return str(out)


def register_c1pg(src_path: str = "data/GIMtec/C1PG.npy",
                  dst_root: str = "outputs/gim/C1PG/seed_external") -> str:
    """Register external C1PG result into unified outputs tree.

    src_path: path to numpy array (predictions) in TECU with shape [B,H,N] or [B,H,N,1].
    """
    ts = time.strftime("%Y%m%dT%H%M%SZ")
    out = Path(dst_root) / (ts + "-external")
    out.mkdir(parents=True, exist_ok=True)
    import numpy as np
    ypred = np.load(src_path)
    if ypred.ndim == 4 and ypred.shape[-1] == 1:
        ypred = ypred[..., 0]
    np.save(out / "predictions.npy", ypred)
    # minimal manifest
    m = {
        "dataset": "GIMtec",
        "dataset_slug": "gim",
        "model": "C1PG",
        "units": "TECU",
        "normalized": False,
    }
    json.dump(m, open(out / "manifest.json", "w"), indent=2)
    print(f"C1PG registered to: {out}")
    return str(out)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=False, help="folder containing external IRI outputs or C1PG file")
    p.add_argument("--dst", default="outputs/GIMtec/IRI_2020/seed_external")
    p.add_argument("--mode", choices=["iri","c1pg"], default="iri")
    args = p.parse_args()
    if args.mode == 'iri':
        if not args.src:
            raise SystemExit("--src required for IRI mode")
        register_iri(args.src, args.dst)
    else:
        register_c1pg(args.src or "data/GIMtec/C1PG.npy", "outputs/gim/C1PG/seed_external")
