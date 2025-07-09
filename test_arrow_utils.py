import pyarrow as pa
import numpy as np
import sys
import os

# 假设编译后的模块在 build 或 debug 目录中
# 将构建目录添加到 Python 路径中
build_dir = os.path.join(os.path.dirname(__file__), 'build')
debug_dir = os.path.join(os.path.dirname(__file__), 'debug')
if os.path.exists(build_dir):
    sys.path.append(build_dir)
if os.path.exists(debug_dir):
    sys.path.append(debug_dir)

try:
    import arrow_utils
except ImportError:
    print("错误：无法导入 arrow_utils 模块。")
    print("请确保项目已经编译，并且 arrow_utils.so (或类似文件) 位于 build/ 或 debug/ 目录中。")
    sys.exit(1)

def run_demo():
    """
    创建测试数据并调用 C++ 模块中的函数。
    """
    # 1. 创建一个包含 4 列的 PyArrow RecordBatch
    data = [
        pa.array([1, 2, 3, 4], type=pa.int64()),    
        pa.array([-1, 5, -7, 8], type=pa.int64()), 
        pa.array([9, 0, 11, -12], type=pa.int64()), 
        pa.array([13, 15, 17, 19], type=pa.int64()),
    ]

    schema = pa.schema([
        pa.field('col1', pa.int64()),
        pa.field('col2', pa.int64()),
        pa.field('col3', pa.int64()),
        pa.field('col4', pa.int64()),
    ])

    batch = pa.RecordBatch.from_arrays(data, schema=schema)

    print("原始 RecordBatch:")
    print(batch)
    print("-" * 40)

    # 2. 调用 C++ 函数
    try:
        print("调用 C++ 函数 arrow_utils.process_record_batch...")
        output_array = arrow_utils.process_record_batch(batch)
        print("调用成功！")
        print("-" * 40)
    except Exception as e:
        print(f"调用 C++ 函数时出错: {e}")
        return

    # 3. 打印并验证结果
    print("返回的 ChunkedArray:")
    print(output_array)
    print("-" * 40)

    print("验证结果:")
    expected_counts = pa.array([3, 2, 3, 1], type=pa.int64())
    
    # output_array is a ChunkedArray, we can compare it directly
    # or convert to a single array if we are sure it has one chunk.
    # For simplicity, we'll compare the Python list representation.
    
    print(f"预期计数值: {expected_counts.to_pylist()}")
    print(f"实际计数值:   {output_array.to_pylist()}")

    if expected_counts.to_pylist() == output_array.to_pylist():
        print("\n验证成功！C++ 函数按预期工作。")
    else:
        print("\n验证失败。输出与预期不符。")

if __name__ == "__main__":
    run_demo()
