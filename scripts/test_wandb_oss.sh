#!/usr/bin/env bash
set -e
echo "== W&B via socks5h:1080 =="
curl -sS --socks5-hostname 127.0.0.1:1080 -o /dev/null -w "status=%{http_code} total=%{time_total}s\n" https://api.wandb.ai/status || true

echo "== OSS direct =="
for host in oss-cn-beijing.aliyuncs.com autodl-models.oss-cn-beijing.aliyuncs.com; do
  HTTPS_PROXY= HTTP_PROXY= curl -sS -o /dev/null -w "$host total=%{time_total}s\n" "https://$host" || true
done
