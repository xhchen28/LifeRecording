import torch
import time
import numpy as np
import argparse
from torch.profiler import profile, record_function, ProfilerActivity


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--miss_ratio', type=float)
    return parser.parse_args()

args = parse_args()


def flush_cache(size=100 * 1024 * 1024):
    """
    尝试刷新CPU缓存（创建并访问大量数据）
    """
    # 创建一个大数组来刷新缓存（约100MB）
    dummy = torch.empty(size, dtype=torch.uint8)
    _ = dummy.sum()
    del dummy


def benchmark_gather(kv_len=32*1024, hidden_dim=656, topk=2048, num_runs=100, warmup=10, 
                     pin_memory=False, flush_cache_between_runs=True, randomize_data=True):
    """
    测试torch.gather从大tensor中提取topk个索引数据的时间
    
    Args:
        kv_len: tensor的第一维度大小
        hidden_dim: tensor的第二维度大小
        topk: 要选取的索引数量
        num_runs: 测试运行次数
        warmup: 预热运行次数
        pin_memory: 是否使用pinned memory
        flush_cache_between_runs: 是否在每次运行间刷新缓存
        randomize_data: 是否每次使用不同的数据和索引
    """
    print(f"配置信息:")
    print(f"  Tensor shape: [{kv_len}, {hidden_dim}]")
    print(f"  Dtype: uint8")
    print(f"  Device: CPU")
    print(f"  Pin Memory: {pin_memory}")
    print(f"  TopK: {topk}")
    print(f"  预热次数: {warmup}")
    print(f"  测试次数: {num_runs}")
    print(f"  刷新缓存: {flush_cache_between_runs}")
    print(f"  随机化数据: {randomize_data}")
    print("-" * 60)
    
    # 预先创建所有测试数据（如果使用随机化）
    tensors = []
    indices_list = []
    
    total_iterations = warmup + num_runs
    
    if randomize_data:
        print("预生成测试数据...")
        for i in range(total_iterations):
            if pin_memory:
                tensor = torch.randint(0, 256, (kv_len, hidden_dim), dtype=torch.uint8, device='cpu').pin_memory()
            else:
                tensor = torch.randint(0, 256, (kv_len, hidden_dim), dtype=torch.uint8, device='cpu')
            
            # 每次使用不同的随机索引
            indices = torch.randperm(kv_len)[:topk]
            indices_expanded = indices.unsqueeze(1).expand(topk, hidden_dim)
            
            tensors.append(tensor)
            indices_list.append(indices_expanded)
            
            if (i + 1) % 20 == 0:
                print(f"  生成 {i+1}/{total_iterations} 数据")
    else:
        # 使用固定数据
        if pin_memory:
            tensor = torch.randint(0, 256, (kv_len, hidden_dim), dtype=torch.uint8, device='cpu').pin_memory()
        else:
            tensor = torch.randint(0, 256, (kv_len, hidden_dim), dtype=torch.uint8, device='cpu')
        
        indices = torch.randperm(kv_len)[:topk]
        indices_expanded = indices.unsqueeze(1).expand(topk, hidden_dim)
        
        for _ in range(total_iterations):
            tensors.append(tensor)
            indices_list.append(indices_expanded)
    
    print(f"单个 tensor 大小: {tensors[0].element_size() * tensors[0].nelement() / 1024 / 1024:.2f} MB")
    print(f"Tensor is pinned: {tensors[0].is_pinned()}")
    print(f"索引数量: {topk}")
    print(f"输出 tensor shape: [{topk}, {hidden_dim}]")
    print("-" * 60)
    
    # 预热
    print("预热中...")
    for i in range(warmup):
        if flush_cache_between_runs:
            flush_cache()
        result = torch.gather(tensors[i], 0, indices_list[i])
    
    # 正式测试
    print("开始测试...")
    times = []
    for i in range(num_runs):
        if flush_cache_between_runs:
            flush_cache()
        
        start = time.perf_counter()
        result = torch.gather(tensors[warmup + i], 0, indices_list[warmup + i])
        end = time.perf_counter()
        times.append((end - start) * 1000)  # 转换为毫秒
        
        if (i + 1) % 20 == 0:
            print(f"  完成 {i+1}/{num_runs} 次测试")
    
    # 计算数据传输量
    # gather操作需要读取的数据量：topk * hidden_dim * sizeof(uint8)
    bytes_read = topk * hidden_dim * 1  # uint8 = 1 byte
    # gather操作需要读取的索引量：topk * hidden_dim * sizeof(int64)
    bytes_index = topk * hidden_dim * 8  # int64 = 8 bytes (indices_expanded)
    # 写入的数据量：topk * hidden_dim * sizeof(uint8)
    bytes_write = topk * hidden_dim * 1
    # 总数据传输量
    total_bytes = bytes_read + bytes_index + bytes_write
    
    # 统计结果
    times = np.array(times)
    print("-" * 60)
    print("测试结果:")
    print(f"  平均时间: {np.mean(times):.4f} ms")
    print(f"  中位数时间: {np.median(times):.4f} ms")
    print(f"  标准差: {np.std(times):.4f} ms")
    print(f"  最小时间: {np.min(times):.4f} ms")
    print(f"  最大时间: {np.max(times):.4f} ms")
    print(f"  95分位数: {np.percentile(times, 95):.4f} ms")
    print(f"  99分位数: {np.percentile(times, 99):.4f} ms")
    
    # 带宽计算
    print("-" * 60)
    print("带宽分析:")
    print(f"  读取数据: {bytes_read / 1024 / 1024:.2f} MB")
    print(f"  读取索引: {bytes_index / 1024 / 1024:.2f} MB")
    print(f"  写入数据: {bytes_write / 1024 / 1024:.2f} MB")
    print(f"  总传输量: {total_bytes / 1024 / 1024:.2f} MB")
    
    # 计算带宽 (GB/s)
    avg_time_seconds = np.mean(times) / 1000  # 转换为秒
    bandwidth_avg = (total_bytes / 1024**3) / avg_time_seconds  # GB/s
    
    min_time_seconds = np.min(times) / 1000
    bandwidth_peak = (total_bytes / 1024**3) / min_time_seconds  # 峰值带宽
    
    print(f"  平均带宽: {bandwidth_avg:.2f} GB/s")
    print(f"  峰值带宽: {bandwidth_peak:.2f} GB/s")
    
    return result, times


def compare_cache_strategies(kv_len=32*1024, hidden_dim=656, topk=2048, num_runs=100):
    """
    对比不同缓存控制策略的影响
    """
    print("\n" + "=" * 60)
    print("缓存策略对比测试")
    print("=" * 60)
    
    strategies = [
        ("固定数据 + 不刷新缓存", False, False),
        ("固定数据 + 刷新缓存", True, False),
        ("随机数据 + 不刷新缓存", False, True),
        ("随机数据 + 刷新缓存", True, True),
    ]
    
    results = {}
    
    # 计算理论数据传输量
    bytes_read = topk * hidden_dim * 1
    bytes_index = topk * hidden_dim * 8
    bytes_write = topk * hidden_dim * 1
    total_bytes = bytes_read + bytes_index + bytes_write
    
    for name, flush, randomize in strategies:
        print(f"\n策略: {name}")
        print("-" * 60)
        _, times = benchmark_gather(
            kv_len=kv_len,
            hidden_dim=hidden_dim,
            topk=topk,
            num_runs=num_runs,
            warmup=5,
            pin_memory=True,
            flush_cache_between_runs=flush,
            randomize_data=randomize
        )
        results[name] = times

    # 对比结果
    print("\n" + "=" * 60)
    print("策略对比总结:")
    print("=" * 60)
    for name, times in results.items():
        avg_time_s = np.mean(times) / 1000
        bandwidth = (total_bytes / 1024**3) / avg_time_s
        print(f"{name}:")
        print(f"  平均: {np.mean(times):.4f} ms, 标准差: {np.std(times):.4f} ms, 带宽: {bandwidth:.2f} GB/s")
    print("-" * 60)


def compare_methods(kv_len=32*1024, hidden_dim=656, topk=2048, num_runs=50, 
                   pin_memory=False, flush_cache_between_runs=True):
    """
    比较不同方法的性能（排除缓存影响）
    """
    print("\n" + "=" * 60)
    print(f"方法对比测试 (Pin Memory: {pin_memory}, 刷新缓存: {flush_cache_between_runs})")
    print("=" * 60)
    
    # 预生成所有测试数据
    tensors = []
    indices_list = []
    
    print("预生成测试数据...")
    for i in range(num_runs):
        if pin_memory:
            tensor = torch.randint(0, 256, (kv_len, hidden_dim), dtype=torch.uint8, device='cpu').pin_memory()
        else:
            tensor = torch.randint(0, 256, (kv_len, hidden_dim), dtype=torch.uint8, device='cpu')

        indices = torch.randperm(kv_len)[:topk]
        tensors.append(tensor)
        indices_list.append(indices)
    
    # 计算数据传输量
    bytes_read = topk * hidden_dim * 1
    bytes_index = topk * 4
    bytes_write = topk * hidden_dim * 1
    total_bytes = bytes_read + bytes_index + bytes_write
    
    # 方法1: torch.gather
    print("\n方法1: torch.gather")
    times1 = []
    for i in range(num_runs):
        if flush_cache_between_runs:
            flush_cache()
        indices_expanded = indices_list[i].unsqueeze(1).expand(topk, hidden_dim)
        start = time.perf_counter()
        result1 = torch.gather(tensors[i], 0, indices_expanded)
        times1.append((time.perf_counter() - start) * 1000)
    bandwidth1 = (total_bytes / 1024**3) / (np.mean(times1) / 1000)
    print(f"  平均时间: {np.mean(times1):.4f} ms, 带宽: {bandwidth1:.2f} GB/s")
    
    # 方法2: 直接索引
    print("\n方法2: 直接索引 (tensor[indices])")
    times2 = []
    for i in range(num_runs):
        if flush_cache_between_runs:
            flush_cache()
        start = time.perf_counter()
        result2 = tensors[i][indices_list[i]]
        times2.append((time.perf_counter() - start) * 1000)
    bandwidth2 = (total_bytes / 1024**3) / (np.mean(times2) / 1000)
    print(f"  平均时间: {np.mean(times2):.4f} ms, 带宽: {bandwidth2:.2f} GB/s")
    
    # 方法3: index_select
    print("\n方法3: torch.index_select")
    times3 = []
    for i in range(num_runs):
        if flush_cache_between_runs:
            flush_cache()
        start = time.perf_counter()
        result3 = torch.index_select(tensors[i], 0, indices_list[i])
        times3.append((time.perf_counter() - start) * 1000)
    bandwidth3 = (total_bytes / 1024**3) / (np.mean(times3) / 1000)
    print(f"  平均时间: {np.mean(times3):.4f} ms, 带宽: {bandwidth3:.2f} GB/s")

    print("\n性能对比:")
    baseline = np.mean(times1)
    print(f"  gather:        1.00x (基准) - {bandwidth1:.2f} GB/s")
    print(f"  直接索引:      {baseline/np.mean(times2):.2f}x - {bandwidth2:.2f} GB/s")
    print(f"  index_select:  {baseline/np.mean(times3):.2f}x - {bandwidth3:.2f} GB/s")
    
    return {
        'gather': times1,
        'direct': times2,
        'index_select': times3
    }


# 数据传输使用pin_memory 数据保存使用正常的 memry 
def measure_bandwidth_for_DV32(kv_len=32*1024, hidden_dim=656, topk=2048, num_runs=50, 
                                miss_ratio = 0.5, batch_size = 32,
                                pin_memory=False, flush_cache_between_runs=True):
    """
    比较不同方法的性能（排除缓存影响）
    """
    # print("\n" + "=" * 60)
    # print(f"方法对比测试 (Pin Memory: {pin_memory}, 刷新缓存: {flush_cache_between_runs})")
    # print("=" * 60)
    
    # 预生成所有测试数据
    tensors = []
    indices_list = []

    # 计算数据传输量
    trans_k = int(topk * miss_ratio * batch_size)

    # print("预生成测试数据...")
    if pin_memory:
        tensor = torch.randint(0, 256, (batch_size * kv_len, hidden_dim), dtype=torch.uint8, device='cpu').pin_memory()
    else:
        tensor = torch.randint(0, 256, (batch_size * kv_len, hidden_dim), dtype=torch.uint8, device='cpu')

    for _ in range(num_runs):
        indices = torch.randperm(batch_size * kv_len)[:trans_k]
        indices_list.append(indices)

    tensors = tensor

    # # 计算有效数据传输量，只考虑读的部分，其他的都是overhead 带宽占用
    bytes_read = trans_k * hidden_dim * 1
    total_bytes = bytes_read

    # print("\n方法1: gather")
    with profile(
            activities=[ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True
    ) as prof:
        for i in range(num_runs):
            indices_expanded = indices_list[i].unsqueeze(1).expand(trans_k, hidden_dim)
            result = torch.gather(tensors, 0, indices_expanded)
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    # print("\n方法2: 直接索引")
    with profile(
            activities=[ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True,
            # with_stack=True,
            with_flops=True
    ) as prof:
        for i in range(num_runs):
            x = tensors[indices_list[i]]
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    # 方法3: index_select
    # print("\n方法3: torch.index_select")
    with profile(
            activities=[ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True,
            # with_stack=True,
            with_flops=True
    ) as prof:
        start = time.perf_counter()
        for i in range(num_runs):
            result3 = torch.index_select(tensors, 0, indices_list[i])
        dur_time = (time.perf_counter() - start)
    bankwidth = trans_k * 656 * num_runs / (2**30) / dur_time
    print(f"  平均带宽: {bankwidth:.4f} GB/s")
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    return bankwidth


if __name__ == "__main__":
    # 测试1: 对比缓存策略的影响
    # compare_cache_strategies(
    #     kv_len=32*1024,
    #     hidden_dim=656,
    #     topk=2048,
    #     num_runs=50
    # )

    # 测试2: 使用最严格的设置（随机数据+刷新缓存）进行方法对比
    # print("\n" + "=" * 60)
    # print("最严格设置下的方法对比")
    # print("=" * 60)
    # compare_methods(
    #     kv_len=32*1024,
    #     hidden_dim=656,
    #     topk=2048,
    #     num_runs=50,
    #     pin_memory=True,
    #     flush_cache_between_runs=True
    # )

    # 测试3: 在 V32 的 offload 场景中进行方法对比
    print("=============================== 开始测量, 使用select_index ===============================")
    for bs in [32, 64, 96, 128, 160, 192, 224, 256]:
        for miss_ratio in [0.1 * i for i in range(1, 10)]:
            max_bandwidth = measure_bandwidth_for_DV32(
                kv_len=32*1024,
                hidden_dim=656,
                topk=2048,
                num_runs=1024,
                pin_memory=True,
                flush_cache_between_runs=False,
                miss_ratio = miss_ratio,
                batch_size = bs
            )

            print(f"batch_size: {bs}, miss_ratio: {miss_ratio}, max_bandwidth: {max_bandwidth}")