#!/usr/bin/env bash
set -euo pipefail

# --------------------------- 0) Secrets & Proxy ---------------------------
# Load secrets if provided (do NOT commit secrets.env)
SECRETS_FILE="${SECRETS_FILE:-/root/tecGPT/.secrets/wandb_oss.env}"
if [[ -f "$SECRETS_FILE" ]]; then
  # shellcheck source=/dev/null
  source "$SECRETS_FILE"
fi

# Define socks5h proxy for W&B; allow full training lifecycle to inherit
PROXY_SOCKS5="${PROXY_SOCKS5:-socks5h://127.0.0.1:1080}"
export PROXY_SOCKS5
# OSS should bypass proxy regardless
export NO_PROXY="oss-accelerate.aliyuncs.com,oss-cn-beijing.aliyuncs.com,localhost,127.0.0.1"

# ====== 训练全程代理（保障 wandb.log 走 socks5h）======
if [[ -n "${PROXY_SOCKS5:-}" ]]; then
  export HTTP_PROXY="$PROXY_SOCKS5"
  export HTTPS_PROXY="$PROXY_SOCKS5"
  export ALL_PROXY="$PROXY_SOCKS5"
  export WANDB_HTTP_PROXY="$PROXY_SOCKS5"
  export WANDB_HTTPS_PROXY="$PROXY_SOCKS5"
  echo "[proxy] Enabled for training: $PROXY_SOCKS5"
fi
# 避免 wandb 单独起进程拿不到你后续对 env 的改动
export WANDB_START_METHOD=${WANDB_START_METHOD:-thread}

# Default W&B context (can be overridden by secrets)
export WANDB_ENTITY="${WANDB_ENTITY:-xiongpan-tsinghua-university}"
export WANDB_PROJECT="${WANDB_PROJECT:-Ion-Phys-Toolkit}"

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  HTTPS_PROXY="$PROXY_SOCKS5" HTTP_PROXY="$PROXY_SOCKS5" ALL_PROXY="$PROXY_SOCKS5" \
    wandb login --relogin "${WANDB_API_KEY}" >/dev/null 2>&1 || true
fi

check_wandb() {
  curl -sS --max-time 5 https://api.wandb.ai/status >/dev/null || return 1
}
if ! HTTPS_PROXY="$PROXY_SOCKS5" HTTP_PROXY="$PROXY_SOCKS5" ALL_PROXY="$PROXY_SOCKS5" check_wandb; then
  echo "[WARN] W&B 不可达，启用离线模式（WANDB_MODE=offline）。"
  export WANDB_MODE=offline
  export WANDB_DIR="${WANDB_DIR:-/tmp/wandb-offline}"
fi

# --------------------------- 1) Parse WANDB_CONFIG -------------------------
# Strictly accept only 'seeds' for multi-seed metadata; training still runs once.
DATASET=""; MODEL=""; SEED=""; SEEDS_COUNT=1
USE_PINN=""; USE_DRIVERS=""; USE_ADV=""; USE_DIFFUSION=""; NOCHEM=""

if [[ -n "${WANDB_CONFIG:-}" ]]; then
  eval "$({
    python - "$WANDB_CONFIG" <<'PY'
import os, sys, json
cfg = json.loads(sys.argv[1]) if len(sys.argv)>1 else {}
def exp(k):
    v = cfg.get(k, None)
    if v is None:
        return
    if isinstance(v, bool):
        print(f'export {k}={"True" if v else "False"}')
    else:
        print(f'export {k}={json.dumps(v)}')

# Basic keys commonly injected
for k in ['dataset','model','seed','use_pinn','use_drivers','use_adv','use_diffusion','nochem',
          'batch_size','epochs','accumulate_steps','optimizer','scheduler','lr_init','weight_decay',
          'year_split','amp','node_chunk','use_lora','init_from','loss_func','config']:
    exp(k)

# Strict multi-seed rule: only accept key 'seeds'
if 'seeds' in cfg and isinstance(cfg['seeds'], (list, tuple)):
    print(f'export SEEDS_COUNT={len(cfg["seeds"]) }')
else:
    print('export SEEDS_COUNT=1')

# warn on legacy keys
for bad in ('seed_list','seed_array'):
    if bad in cfg:
        print(f'echo "[WARN] 忽略非标准键 {bad}，请使用 seeds" 1>&2')
PY
  })"
  # shellcheck disable=SC2153
  DATASET="${dataset:-${DATASET:-}}"
  MODEL="${model:-${MODEL:-}}"
  SEED="${seed:-${SEED:-}}"
  USE_PINN="${use_pinn:-${USE_PINN:-}}"
  USE_DRIVERS="${use_drivers:-${USE_DRIVERS:-}}"
  USE_ADV="${use_adv:-${USE_ADV:-}}"
  USE_DIFFUSION="${use_diffusion:-${USE_DIFFUSION:-}}"
  NOCHEM="${nochem:-${NOCHEM:-}}"
fi

if [[ -z "$DATASET" || -z "$MODEL" ]]; then
  echo "[ERROR] 需要在 WANDB_CONFIG 或环境变量中提供 dataset 与 model" >&2
  exit 2
fi
if [[ -z "$SEED" ]]; then
  SEED=0
fi
echo "[INFO] dataset=$DATASET model=$MODEL seed=$SEED (seeds_count=${SEEDS_COUNT})"

# --------------------------- 2) Launch single run --------------------------
set +e
python -u model/Run.py \
  -dataset "$DATASET" -model "$MODEL" -mode ori -seed "$SEED" \
  ${USE_PINN:+-use_pinn ${USE_PINN}} \
  ${USE_DRIVERS:+-use_drivers ${USE_DRIVERS}} \
  ${USE_ADV:+-use_adv ${USE_ADV}} \
  ${USE_DIFFUSION:+-use_diffusion ${USE_DIFFUSION}} \
  ${NOCHEM:+-nochem ${NOCHEM}} \
  ${batch_size:+-batch_size ${batch_size}} \
  ${epochs:+-epochs ${epochs}} \
  ${accumulate_steps:+-accumulate_steps ${accumulate_steps}} \
  ${optimizer:+-optimizer ${optimizer}} \
  ${scheduler:+-scheduler ${scheduler}} \
  ${lr_init:+-lr_init ${lr_init}} \
  ${weight_decay:+-weight_decay ${weight_decay}} \
  ${year_split:+-year_split ${year_split}} \
  ${amp:+-amp ${amp}} \
  ${node_chunk:+--node_chunk ${node_chunk}} \
  ${use_lora:+--use_lora ${use_lora}} \
  ${init_from:+-init_from ${init_from}} \
  ${loss_func:+-loss_func ${loss_func}}
code=$?
set -e

if [[ $code -ne 0 ]]; then
  echo "[ERROR] 训练进程返回非零状态: $code" >&2
  exit $code
fi

# --------------------------- 3) Locate last outputs ------------------------
MODEL_SLUG=$(echo "$MODEL" | tr 'A-Z' 'a-z')
# Map dataset to slug (lowercase)
DATASET_SLUG=$(python - <<'PY' "$DATASET"
import sys
name=sys.argv[1].strip().lower()
slug={'gimtec':'gim','tec':'gim','china_hires':'china_hires','metr_la':'metrla','nyc_taxi':'nyc_taxi','nyc_bike':'nyc_bike','pems08':'pems08'}.get(name,name)
print(slug)
PY
)
ROOT="outputs/${DATASET_SLUG}/${MODEL_SLUG}/seed_${SEED}"
if [[ ! -d "$ROOT" ]]; then
  echo "[WARN] 未找到输出目录: $ROOT" >&2
  exit 0
fi
LATEST_DIR=$(ls -td "$ROOT"/* 2>/dev/null | head -n 1 || true)
if [[ -z "$LATEST_DIR" ]]; then
  echo "[WARN] 未发现最新输出目录" >&2
  exit 0
fi
echo "[INFO] 最新输出目录: $LATEST_DIR"

# --------------------------- 4) Upload small files (W&B) -------------------
if [[ -z "${WANDB_MODE:-}" || "${WANDB_MODE}" != "offline" ]]; then
  for f in metrics.json compute_cost.json stepwise_rmse.csv manifest.json; do
    p="$LATEST_DIR/$f"
    if [[ -f "$p" ]]; then
      echo "[INFO] W&B 兜底上传: $f"
      HTTPS_PROXY="$PROXY_SOCKS5" HTTP_PROXY="$PROXY_SOCKS5" ALL_PROXY="$PROXY_SOCKS5" \
      python - <<'PY' "$p"
import os, sys
try:
    import wandb
except Exception:
    sys.exit(0)
p = sys.argv[1]
run = wandb.run or wandb.init(project=os.getenv('WANDB_PROJECT','Ion-Phys-Toolkit'),
                               entity=os.getenv('WANDB_ENTITY','xiongpan-tsinghua-university'))
art = wandb.Artifact(name=os.path.basename(p).split('.')[0]+'-fallback', type='results')
art.add_file(p)
run.log_artifact(art, aliases=['latest'])
PY
    fi
  done
fi

# --------------------------- 5) Upload large files (OSS) -------------------
if [[ -n "${OSS_ACCESS_KEY_ID:-}" && -n "${OSS_ACCESS_KEY_SECRET:-}" && -n "${OSS_BUCKET:-}" ]]; then
  echo "[INFO] 开始同步到 OSS: bucket=${OSS_BUCKET}"
  python - <<'PY' "$LATEST_DIR" "$DATASET" "$MODEL_SLUG"
import os, sys, io, tarfile
from pathlib import Path

import requests
import oss2

root = Path(sys.argv[1])
dataset = sys.argv[2]
model_slug = sys.argv[3]
seed = os.environ.get('SEED','0')
ts = os.path.basename(str(root))
endpoint = os.environ.get('OSS_ENDPOINT', 'oss-accelerate.aliyuncs.com')
bucket_name = os.environ['OSS_BUCKET']
ak = os.environ['OSS_ACCESS_KEY_ID']
sk = os.environ['OSS_ACCESS_KEY_SECRET']

# direct session w/o proxies
sess = requests.Session(); sess.trust_env = False
auth = oss2.Auth(ak, sk)
bucket = oss2.Bucket(auth, 'https://'+endpoint, bucket_name, session=oss2.Session(sess))

prefix = f"Ion-Phys-Toolkit/{dataset}/{model_slug}/seed_{seed}/{ts}/"

def upload_small(p: Path):
    key = prefix + p.relative_to(root).as_posix()
    with open(p, 'rb') as f:
        bucket.put_object(key, f)

def resumable_upload(local_path: Path, key: str):
    # 6 threads, 10MB part size
    oss2.resumable_upload(bucket, key, str(local_path), multipart_threshold=10*1024*1024,
                          num_threads=6, part_size=10*1024*1024)

for p in root.rglob('*'):
    if not p.is_file():
        continue
    size = p.stat().st_size
    key = prefix + p.relative_to(root).as_posix()
    try:
        if size <= 50*1024*1024:
            # try in-memory put
            with open(p, 'rb') as f:
                data = f.read()
            bucket.put_object(key, data)
        else:
            # compress then upload with resume + threads
            tmp = p.with_suffix(p.suffix + '.tar.gz')
            with tarfile.open(tmp, 'w:gz') as tar:
                tar.add(str(p), arcname=p.name)
            resumable_upload(tmp, key + '.tar.gz')
            try:
                tmp.unlink()
            except Exception:
                pass
    except Exception as e:
        # fallback: try resumable upload of the raw file
        try:
            resumable_upload(p, key)
        except Exception:
            sys.stderr.write(f"[ERR] 上传失败: {p}\n")

print('[INFO] OSS 同步完成')
PY
  # After upload, log W&B artifacts referencing s3:// URIs
  # Derive run_tag from dirname and compose s3 prefix
  if [[ -z "${WANDB_MODE:-}" || "${WANDB_MODE}" != "offline" ]]; then
    HTTPS_PROXY="$PROXY_SOCKS5" HTTP_PROXY="$PROXY_SOCKS5" ALL_PROXY="$PROXY_SOCKS5" \
    python - <<'PY' "$LATEST_DIR" "$DATASET_SLUG" "$MODEL_SLUG" "$SEED" "$OSS_BUCKET"
import os, sys
out_dir, ds, md, seed, bucket = sys.argv[1:6]
run_tag = os.path.basename(out_dir)
prefix = f"Ion-Phys-Toolkit/{ds}/{md}/seed_{seed}/{run_tag}/"
try:
    import wandb
except Exception:
    sys.exit(0)
run = wandb.run or wandb.init(project=os.getenv('WANDB_PROJECT','Ion-Phys-Toolkit'),
                               entity=os.getenv('WANDB_ENTITY','xiongpan-tsinghua-university'))
# references for predictions and best model if exist
refs=[]
p_pred = os.path.join(out_dir, 'predictions.npy')
if os.path.exists(p_pred):
    refs.append(f"s3://{bucket}/{prefix}predictions.npy")
bm = os.path.join(out_dir, 'best_model.pth')
if os.path.exists(bm):
    refs.append(f"s3://{bucket}/{prefix}best_model.pth")
if refs:
    art = wandb.Artifact(name=f"oss-refs-{run.id}", type='external')
    for r in refs:
        art.add_reference(r)
    run.log_artifact(art, aliases=['latest'])
PY
  fi
else
  echo "[INFO] 未配置 OSS 访问凭据，跳过 OSS 上传"
fi

echo "[INFO] 运行结束"
