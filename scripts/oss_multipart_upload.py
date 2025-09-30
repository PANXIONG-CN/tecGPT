#!/usr/bin/env python3
"""
OSS 分片并行上传工具
适用于大文件 (>100MB) 的快速上传
"""
import os, oss2, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from oss2.models import PartInfo

class MultipartUploader:
    def __init__(self, bucket, part_size_mb=100, max_workers=4):
        self.bucket = bucket
        self.part_size = part_size_mb * 1024 * 1024  # 转换为字节
        self.max_workers = max_workers
        
    def upload_part_worker(self, args):
        """单个分片上传任务"""
        upload_id, key, part_num, data = args
        start = time.time()
        result = self.bucket.upload_part(key, upload_id, part_num, data)
        elapsed = time.time() - start
        size_mb = len(data) / (1024**2)
        speed = size_mb / elapsed if elapsed > 0 else 0
        print(f"  ✓ Part {part_num}: {size_mb:.1f} MB uploaded in {elapsed:.1f}s ({speed:.1f} MB/s)")
        return PartInfo(part_num, result.etag)
    
    def upload_file(self, local_file, oss_key):
        """分片并行上传文件"""
        file_size = os.path.getsize(local_file)
        file_size_mb = file_size / (1024**2)
        
        print(f"📤 开始分片上传:")
        print(f"   本地文件: {local_file}")
        print(f"   OSS 路径: {oss_key}")
        print(f"   文件大小: {file_size_mb:.1f} MB")
        print(f"   分片大小: {self.part_size/(1024**2):.0f} MB")
        print(f"   并发数: {self.max_workers}")
        
        # 初始化分片上传
        upload_id = self.bucket.init_multipart_upload(oss_key).upload_id
        
        # 读取文件并分片
        parts_args = []
        part_num = 1
        
        with open(local_file, 'rb') as f:
            while True:
                data = f.read(self.part_size)
                if not data:
                    break
                parts_args.append((upload_id, oss_key, part_num, data))
                part_num += 1
        
        total_parts = len(parts_args)
        print(f"   总分片数: {total_parts}\n")
        
        # 并行上传
        start_time = time.time()
        parts = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.upload_part_worker, args) 
                      for args in parts_args]
            
            for future in as_completed(futures):
                parts.append(future.result())
        
        # 完成分片上传
        parts.sort(key=lambda x: x.part_number)
        self.bucket.complete_multipart_upload(oss_key, upload_id, parts)
        
        elapsed = time.time() - start_time
        speed = file_size_mb / elapsed
        
        print(f"\n✅ 上传完成!")
        print(f"   总耗时: {elapsed:.1f} 秒")
        print(f"   平均速度: {speed:.2f} MB/s ({speed*8:.1f} Mbps)")
        print(f"   加速比: {self.max_workers * speed / 11.10:.1f}x")

# 使用示例
if __name__ == '__main__':
    # OSS 配置
    ak = os.environ["OSS_ACCESS_KEY_ID"]
    sk = os.environ["OSS_ACCESS_KEY_SECRET"]
    endpoint = os.environ.get("OSS_ENDPOINT", "oss-accelerate.aliyuncs.com")
    bucket_name = os.environ["OSS_BUCKET"]
    
    auth = oss2.Auth(ak, sk)
    bucket = oss2.Bucket(auth, 'https://' + endpoint, bucket_name)
    
    # 创建上传器
    uploader = MultipartUploader(bucket, part_size_mb=200, max_workers=6)
    
    # 上传文件
    # uploader.upload_file('/path/to/large_file.bin', 'oss_key')
    print("使用方法:")
    print("  uploader.upload_file('/path/to/file', 'oss/path/file')")
