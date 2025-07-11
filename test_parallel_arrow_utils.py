#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'debug'))

import pyarrow as pa
import numpy as np
import time
from arrow_utils import extract_depth_feature_from_arrow_table

def create_test_table(num_batches=4, rows_per_batch=10000, num_cols=10):
    """创建一个测试用的 Arrow Table"""
    batches = []
    
    for batch_idx in range(num_batches):
        # 创建测试数据 - 每个batch的数据都不同
        arrays = []
        fields = []
        
        for col_idx in range(num_cols):
            # 生成随机数据，每个batch和每列都不同
            np.random.seed(batch_idx * num_cols + col_idx)
            data = np.random.random(rows_per_batch) * (batch_idx + 1) * (col_idx + 1)
            arrays.append(pa.array(data))
            fields.append(pa.field(f'col_{col_idx}', pa.float64()))
        
        schema = pa.schema(fields)
        batch = pa.RecordBatch.from_arrays(arrays, schema=schema)
        batches.append(batch)
    
    return pa.Table.from_batches(batches)

def test_parallel_processing():
    """测试并行处理功能"""
    print("Creating test table...")
    table = create_test_table(num_batches=8, rows_per_batch=5000, num_cols=6)
    print(f"Created table with {table.num_rows} rows, {table.num_columns} columns")
    
    # 定义深度值
    depth_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    # 测试不同的线程数
    test_cases = [
        ("默认线程数", None),
        ("单线程", 1),
        ("2线程", 2),
        ("4线程", 4),
        ("8线程", 8),
    ]
    
    results = {}
    
    for test_name, max_threads in test_cases:
        print(f"\n=== {test_name} ===")
        
        start_time = time.time()
        
        if max_threads is None:
            result_table = extract_depth_feature_from_arrow_table(table, depth_values)
        else:
            result_table = extract_depth_feature_from_arrow_table(table, depth_values, max_threads)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        print(f"处理时间: {elapsed:.4f} 秒")
        print(f"结果表: {result_table.num_rows} 行, {result_table.num_columns} 列")
        
        results[test_name] = {
            'time': elapsed,
            'table': result_table
        }
        
        # 验证结果的一致性（所有结果应该相同）
        if test_name != "默认线程数":
            # 比较与默认结果的一致性
            default_table = results["默认线程数"]['table']
            if result_table.num_rows == default_table.num_rows and result_table.num_columns == default_table.num_columns:
                print("✓ 结果与默认线程数结果一致")
            else:
                print("✗ 结果与默认线程数结果不一致！")
    
    # 显示性能对比
    print("\n=== 性能对比 ===")
    baseline_time = results["单线程"]['time']
    for test_name, result in results.items():
        if test_name != "单线程":
            speedup = baseline_time / result['time']
            print(f"{test_name}: {result['time']:.4f}s (加速比: {speedup:.2f}x)")

def test_error_handling():
    """测试错误处理"""
    print("\n=== 错误处理测试 ===")
    
    table = create_test_table(num_batches=2, rows_per_batch=100, num_cols=4)
    depth_values = [0.1, 0.5, 1.0]
    
    # 测试无效的 max_threads 参数
    try:
        extract_depth_feature_from_arrow_table(table, depth_values, 0)
        print("✗ 应该抛出错误但没有")
    except RuntimeError as e:
        print(f"✓ 正确捕获错误: {e}")
    
    try:
        extract_depth_feature_from_arrow_table(table, depth_values, -1)
        print("✗ 应该抛出错误但没有")
    except RuntimeError as e:
        print(f"✓ 正确捕获错误: {e}")

if __name__ == "__main__":
    print("Arrow Utils 并行处理测试")
    print("=" * 50)
    
    test_parallel_processing()
    test_error_handling()
    
    print("\n测试完成！")
