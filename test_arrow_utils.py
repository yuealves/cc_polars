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
        pa.array([1.1, 1.2, 1.3, 1.4], type=pa.float64()),
        pa.array([2.1, 2.2, 2.3, 2.4], type=pa.float64()),
        pa.array([10, 50, 20, 40], type=pa.float64()),
        pa.array([15, 30, 40, 50], type=pa.float64()),
    ]

    schema = pa.schema([
        pa.field('a1_p', pa.float64()),
        pa.field('a2_p', pa.float64()),
        pa.field('a1_v', pa.float64()),
        pa.field('a2_v', pa.float64()),
    ])


    batch = pa.RecordBatch.from_arrays(data, schema=schema)

    print("原始 RecordBatch:")
    print(batch)
    print("-" * 40)

    # 2. 为 extract_depth_feature 创建 depth_list
    depth_list = np.array([50.0, 100.0, 200.0])
    print(f"使用的 depth_list: {depth_list}")
    print("-" * 40)

    # 3. 调用新的 C++ 函数
    try:
        print("调用 C++ 函数 arrow_utils.extract_depth_feature...")
        output_array = arrow_utils.extract_depth_feature(batch, depth_list.tolist())
        print("调用成功！")
        print("-" * 40)
    except Exception as e:
        print(f"调用 C++ 函数时出错: {e}")
        return

    # 4. 打印并验证结果
    print("返回的 Array:")
    print(output_array)
    print("-" * 40)

    print("验证结果:")
    expected_values = pa.array([2.1, 1.2, 2.3, 1.4])
    
    print(f"预期值: {expected_values.to_pylist()}")
    print(f"实际值:   {output_array.to_pylist()}")

    if expected_values.equals(output_array):
        print("\n验证成功！C++ 函数 'extract_depth_feature' 按预期工作。")
    else:
        print("\n验证失败。输出与预期不符。")

if __name__ == "__main__":
    run_demo()
