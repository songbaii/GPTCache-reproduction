import os
from datasets import load_dataset
from datasets import DatasetDict

class pre_processor:
    def __init__(self, dataset : str):
        self.dataset = dataset

    def pre_process_vector_prompt(self, prompt:str) -> str:
        prompt = prompt.lower().strip()
        return prompt

    def load_sembenchmark(self)-> DatasetDict:
        # 将数据集下载到当前文件所在目录的data子目录中
        # 可用的数据集如下：
        # SemBenchmarkSearchQueries
        # SemBenchmarkClassificationSorted
        # SemBenchmarkLmArena
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        os.makedirs(data_dir, exist_ok=True)
        ds = load_dataset(f"vCache/{self.dataset}", cache_dir=data_dir)
        return ds

    def pre_process_vector(self)-> DatasetDict:
        ds = self.load_sembenchmark()
        if self.dataset == "SemBenchmarkClassificationSorted":
            columns_to_remove = ['dataset_name', 'emb_gte', 'emb_gte_lat', 'response_llama_3_8b_lat', 'output_format', 'id']
        elif self.dataset == "SemBenchmarkSearchQueries":
            columns_to_remove = ['word_count', 'emb_gte', 'emb_gte_lat', 'response_llama_3_8b_lat', 'response_llama_3_8b', 'id']
        elif self.dataset == "SemBenchmarkLmArena":
            columns_to_remove = ['dataset_name', 'emb_text-embedding-3-large', 'emb_text-embedding-3-large_lat','emb_text-embedding-3-small', 'emb_text-embedding-3-small_lat', 'response_gpt-4o-mini_lat', 'response_gpt-4.1-nano_lat', 'emb_gte', 'emb_gte_lat', 'emb_gte_ft', 'emb_gte_ft_lat', 'emb_e5_large_v2', 'emb_e5_large_v2_lat', 'emb_e5_large_v2_ft', 'emb_e5_large_v2_ft_lat', 'response_gpt-4o-mini', 'response_gpt-4.1-nano', 'id']
        else:
            columns_to_remove = []
        ds["train"] = ds["train"].remove_columns(columns_to_remove)
        ds["train"] = ds["train"].map(lambda x: {**x, "prompt": self.pre_process_vector_prompt(x["prompt"])})
        return ds

    

if __name__ == '__main__':
    dataset = ["SemBenchmarkClassificationSorted", "SemBenchmarkLmArena", "SemBenchmarkSearchQueries"]
    for ds in dataset:
        print(f"向量化处理测试{ds}数据集")
        processor = pre_processor(ds)
        ds = processor.pre_process_vector()
        print("处理后的列名：", ds["train"].column_names)