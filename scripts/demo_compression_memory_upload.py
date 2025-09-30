#!/usr/bin/env python3
"""演示：内存中压缩并直接上传到 OSS"""
import os, time, io, gzip, oss2

# OSS 配置
ak = os.environ["OSS_ACCESS_KEY_ID"]
sk = os.environ["OSS_ACCESS_KEY_SECRET"]
endpoint = "oss-accelerate.aliyuncs.com"
bucket_name = os.environ["OSS_BUCKET"]

auth = oss2.Auth(ak, sk)
bucket = oss2.Bucket(auth, 'https://' + endpoint, bucket_name)

print("=" * 70)
print("演示：内存中压缩并直接上传（无磁盘 I/O）")
print("=" * 70)

# 1. 生成测试数据 (100MB，模拟模型权重)
print("\n🔄 步骤 1: 生成 100MB 测试数据...")
data_size_mb = 100
original_data = os.urandom(data_size_mb * 1024 * 1024)
print(f"   原始大小: {len(original_data) / (1024**2):.1f} MB")

# 2. 在内存中压缩
print("\n🗜️  步骤 2: 在内存中压缩...")
t0 = time.time()

# 创建内存缓冲区
compressed_buffer = io.BytesIO()

# 使用 gzip 压缩到内存
with gzip.GzipFile(fileobj=compressed_buffer, mode='wb', compresslevel=6) as gz:
    gz.write(original_data)

compressed_data = compressed_buffer.getvalue()
t1 = time.time()

compression_ratio = (1 - len(compressed_data) / len(original_data)) * 100
print(f"   压缩后大小: {len(compressed_data) / (1024**2):.1f} MB")
print(f"   压缩率: {compression_ratio:.1f}%")
print(f"   压缩耗时: {t1-t0:.2f} 秒")

# 3. 直接从内存上传
print("\n📤 步骤 3: 从内存直接上传到 OSS...")
t2 = time.time()

# 使用 BytesIO 包装压缩数据
upload_stream = io.BytesIO(compressed_data)
object_key = f"demo/compressed_in_memory_{int(time.time())}.gz"

result = bucket.put_object(object_key, upload_stream)
t3 = time.time()

upload_speed = (len(compressed_data) / (1024**2)) / (t3 - t2)
print(f"   上传耗时: {t3-t2:.2f} 秒")
print(f"   上传速度: {upload_speed:.2f} MB/s")
print(f"   OSS 对象: {object_key}")

# 4. 总结
print("\n" + "=" * 70)
print("✅ 全流程完成（完全在内存中，无磁盘 I/O）")
print("=" * 70)
total_time = t3 - t0
saved_time = (len(original_data) / (1024**2) / 11.10) - total_time
print(f"📊 性能对比:")
print(f"   原始数据大小: {len(original_data) / (1024**2):.1f} MB")
print(f"   实际上传大小: {len(compressed_data) / (1024**2):.1f} MB")
print(f"   压缩 + 上传总耗时: {total_time:.2f} 秒")
print(f"   直接上传需要: {len(original_data) / (1024**2) / 11.10:.2f} 秒")
print(f"   节省时间: {saved_time:.2f} 秒 ({saved_time / (len(original_data) / (1024**2) / 11.10) * 100:.1f}%)")
print("=" * 70)

print("\n💡 关键优势:")
print("   ✅ 全程在内存，零磁盘 I/O")
print("   ✅ 减少网络传输量")
print("   ✅ 降低 OSS 存储成本")
print("   ✅ 下载时自动解压（如果客户端支持）")
