import os, time, tempfile, oss2, wandb

# 1) 让 W&B 仅走 socks5h:1080
os.environ["HTTPS_PROXY"] = "socks5h://127.0.0.1:1080"
os.environ["HTTP_PROXY"]  = "socks5h://127.0.0.1:1080"
run = wandb.init(project=os.getenv("WANDB_PROJECT","tecgpt"),
                 entity=os.getenv("WANDB_ENTITY","xiongpan-tsinghua-university"),
                 config={"probe":"ok"})
for i in range(3): wandb.log({"latency_ms": i*10})
run.finish()
print("✅ W&B done")

# 2) OSS 直连（清除代理环境变量）
del os.environ["HTTPS_PROXY"]
del os.environ["HTTP_PROXY"]

ak=os.environ["OSS_ACCESS_KEY_ID"]
sk=os.environ["OSS_ACCESS_KEY_SECRET"]
endpoint=os.environ.get("OSS_ENDPOINT","oss-cn-beijing.aliyuncs.com")
bucket_name=os.environ["OSS_BUCKET"]

auth=oss2.Auth(ak, sk)
bucket = oss2.Bucket(auth, 'https://'+endpoint, bucket_name)

data=os.urandom(4*1024*1024)  # 4MB
with tempfile.NamedTemporaryFile(delete=False) as f: 
    f.write(data)
    tmp=f.name

try:
    t0=time.time()
    bucket.put_object_from_file("speedtest/4M.bin", tmp)
    t1=time.time()
    print("✅ OSS 4MB upload: %.3fs (%.2f MB/s)" % (t1-t0, 4.0/(t1-t0)))
finally:
    os.unlink(tmp)
