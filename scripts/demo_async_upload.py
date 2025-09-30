#!/usr/bin/env python3
"""
演示：后台异步上传
核心概念：训练不等待上传完成，而是继续进行
"""
import os, time, threading, queue, oss2

# OSS 配置
ak = os.environ["OSS_ACCESS_KEY_ID"]
sk = os.environ["OSS_ACCESS_KEY_SECRET"]
endpoint = "oss-accelerate.aliyuncs.com"
bucket_name = os.environ["OSS_BUCKET"]

auth = oss2.Auth(ak, sk)
bucket = oss2.Bucket(auth, 'https://' + endpoint, bucket_name)

print("=" * 70)
print("演示：后台异步上传 vs 同步上传")
print("=" * 70)

# 创建上传队列
upload_queue = queue.Queue()
upload_results = {}

def async_upload_worker():
    """后台上传工作线程"""
    while True:
        try:
            task = upload_queue.get(timeout=1)
            if task is None:  # 结束信号
                break
            
            task_id, data, oss_key = task
            print(f"  📤 后台线程: 开始上传 {oss_key}")
            t0 = time.time()
            
            # 上传到 OSS
            bucket.put_object(oss_key, data)
            
            t1 = time.time()
            upload_results[task_id] = {
                'success': True,
                'time': t1 - t0,
                'key': oss_key
            }
            print(f"  ✅ 后台线程: {oss_key} 上传完成 ({t1-t0:.1f}秒)")
            upload_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"  ❌ 上传失败: {e}")
            upload_results[task_id] = {'success': False, 'error': str(e)}
            upload_queue.task_done()

# 启动后台上传线程
upload_thread = threading.Thread(target=async_upload_worker, daemon=True)
upload_thread.start()
print("✅ 后台上传线程已启动\n")

# ============= 场景 1: 同步上传（阻塞训练）=============
print("🔴 场景 1: 同步上传（传统方式，阻塞训练）")
print("-" * 70)

# 模拟训练 3 个 epoch
for epoch in range(1, 4):
    # 训练阶段
    print(f"  [Epoch {epoch}] 🔄 训练中...")
    time.sleep(1)  # 模拟训练耗时 1 秒
    
    # 保存 checkpoint
    checkpoint_data = os.urandom(10 * 1024 * 1024)  # 10MB
    print(f"  [Epoch {epoch}] 💾 保存 checkpoint (10MB)")
    
    # 同步上传（阻塞！）
    print(f"  [Epoch {epoch}] ⏸️  等待上传完成...")
    t0 = time.time()
    bucket.put_object(f"demo/sync_checkpoint_epoch_{epoch}.bin", checkpoint_data)
    t1 = time.time()
    print(f"  [Epoch {epoch}] ✅ 上传完成 ({t1-t0:.1f}秒)")
    print(f"  [Epoch {epoch}] ⚠️  训练被阻塞了 {t1-t0:.1f} 秒！\n")

sync_total_time = time.time()

# ============= 场景 2: 异步上传（不阻塞训练）=============
print("\n🟢 场景 2: 异步上传（推荐方式，不阻塞训练）")
print("-" * 70)

async_start = time.time()

# 模拟训练 3 个 epoch
for epoch in range(1, 4):
    # 训练阶段
    print(f"  [Epoch {epoch}] 🔄 训练中...")
    time.sleep(1)  # 模拟训练耗时 1 秒
    
    # 保存 checkpoint
    checkpoint_data = os.urandom(10 * 1024 * 1024)  # 10MB
    print(f"  [Epoch {epoch}] 💾 保存 checkpoint (10MB)")
    
    # 异步上传（不阻塞！）
    task_id = f"epoch_{epoch}"
    oss_key = f"demo/async_checkpoint_epoch_{epoch}.bin"
    upload_queue.put((task_id, checkpoint_data, oss_key))
    print(f"  [Epoch {epoch}] 🚀 已添加到上传队列，继续训练！")
    print(f"  [Epoch {epoch}] ✅ 训练未被阻塞，立即进入下一轮\n")

# 训练完成，等待后台上传完成
print("  ⏳ 训练已完成，等待后台上传队列清空...")
upload_queue.join()
async_total_time = time.time() - async_start

# 发送结束信号
upload_queue.put(None)
upload_thread.join(timeout=2)

# ============= 结果对比 =============
print("\n" + "=" * 70)
print("📊 性能对比总结")
print("=" * 70)

print("\n同步上传方式:")
print("  • 每次保存都阻塞训练")
print("  • 用户感知明显延迟")
print("  • 训练效率低")

print("\n异步上传方式:")
print("  • 训练和上传并行进行")
print("  • 用户几乎无感知")
print("  • 训练效率高")
print(f"  • 总耗时: {async_total_time:.1f} 秒")

print("\n" + "=" * 70)
print("💡 后台异步上传的核心概念")
print("=" * 70)
print("""
1. 训练进程：
   训练 → 保存 checkpoint → 放入上传队列 → 立即继续训练
   
2. 后台线程：
   从队列取任务 → 上传到 OSS → 完成 → 继续下一个

3. 关键优势：
   ✅ 训练不等待上传
   ✅ 充分利用网络带宽（训练时同时上传）
   ✅ 用户体验更好
   ✅ 失败自动重试（可扩展）

4. 实现要点：
   • 使用 queue.Queue() 线程安全队列
   • threading.Thread() 创建后台线程
   • daemon=True 让线程随主程序退出
   • upload_queue.join() 等待所有上传完成
""")

print("\n✨ 实际训练中的使用示例:")
print("-" * 70)
print("""
# 初始化（一次）
upload_queue = queue.Queue()
threading.Thread(target=upload_worker, daemon=True).start()

# 训练循环中
for epoch in range(num_epochs):
    train_one_epoch()  # 训练
    
    # 保存 checkpoint
    checkpoint_path = f'checkpoint_epoch_{epoch}.pth'
    torch.save(model.state_dict(), checkpoint_path)
    
    # 异步上传（不阻塞）
    upload_queue.put((checkpoint_path, f'oss://bucket/{checkpoint_path}'))
    # 立即继续下一个 epoch，无需等待！

# 训练结束后等待上传完成
upload_queue.join()
print("所有上传完成！")
""")

print("=" * 70)
