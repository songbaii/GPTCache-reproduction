import os
from datasets import load_dataset

def load_sembenchmark(dataset : str):
    # 将数据集下载到当前文件所在目录的data子目录中
    # 可用的数据集如下：
    # SemBenchmarkSearchQueries
    # SemBenchmarkClassificationSorted
    # SemBenchmarkLmArena
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    ds = load_dataset(f"vCache/{dataset}", cache_dir=data_dir)
    return ds

if __name__ == '__main__':
    ds = load_sembenchmark("SemBenchmarkSearchQueries")
    print(ds.keys())
    print("训练集大小：", len(ds["train"]))
    print("列名：", ds["train"].column_names)
    prompts = ds["train"]["prompt"]
    avg_len = sum(len(p) for p in prompts) / len(prompts)
    print("prompt 平均长度：", avg_len)
    