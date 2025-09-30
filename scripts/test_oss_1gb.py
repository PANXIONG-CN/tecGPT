import os, time, io, oss2

# 确保不使用代理
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("https_proxy", None)
os.environ.pop("http_proxy", None)

# OSS 配置
ak = os.environ["OSS_ACCESS_KEY_ID"]
sk = os.environ["OSS_ACCESS_KEY_SECRET"]
endpoint = os.environ.get("OSS_ENDPOINT", "oss-cn-beijing.aliyuncs.com")
bucket_name = os.environ["OSS_BUCKET"]

auth = oss2.Auth(ak, sk)
bucket = oss2.Bucket(auth, 'https://' + endpoint, bucket_name)

# 生成 1GB 数据（在内存中）
print("🔄 正在生成 1GB 随机数据...")
data_size_mb = 1024  # 1GB = 1024 MB
chunk_size_mb = 64   # 每次生成 64MB
data = bytearray()

for i in range(data_size_mb // chunk_size_mb):
    data.extend(os.urandom(chunk_size_mb * 1024 * 1024))
    print(f"   已生成: {(i+1) * chunk_size_mb} MB / {data_size_mb} MB")

print(f"✅ 数据生成完成，大小: {len(data) / (1024**3):.2f} GB")

# 使用 BytesIO 从内存直接上传
print("\n🚀 开始上传到 OSS...")
data_stream = io.BytesIO(bytes(data))
del data  # 释放原始数据内存

object_key = f"speedtest/test_1GB_{int(time.time())}.bin"

t0 = time.time()
result = bucket.put_object(object_key, data_stream)
t1 = time.time()

elapsed = t1 - t0
speed_mbps = (data_size_mb / elapsed)

print(f"\n✅ 上传完成!")
print(f"📊 统计信息:")
print(f"   - 文件大小: {data_size_mb} MB (1 GB)")
print(f"   - 上传耗时: {elapsed:.2f} 秒")
print(f"   - 上传速度: {speed_mbps:.2f} MB/s ({speed_mbps * 8:.2f} Mbps)")
print(f"   - OSS 对象: {object_key}")
print(f"   - ETag: {result.etag}")
print(f"   - Request ID: {result.request_id}")
