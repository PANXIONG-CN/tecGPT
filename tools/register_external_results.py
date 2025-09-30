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


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="folder containing external IRI outputs")
    p.add_argument("--dst", default="outputs/GIMtec/IRI_2020/seed_external")
    args = p.parse_args()
    register_iri(args.src, args.dst)

