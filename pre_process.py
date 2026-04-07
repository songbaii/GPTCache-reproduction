import load_dataset
from datasets import DatasetDict

def pre_process_vector_prompt(prompt:str) -> str:
    prompt = prompt.lower().strip()
    return prompt

def pre_process_vector(dataset : str)-> DatasetDict:
    ds = load_dataset.load_sembenchmark(dataset)
    if dataset == "SemBenchmarkClassificationSorted":
        columns_to_remove = ['dataset_name', 'emb_gte', 'emb_gte_lat', 'response_llama_3_8b_lat', 'output_format']
    elif dataset == "SemBenchmarkSearchQueries":
        columns_to_remove = ['word_count', 'emb_gte', 'emb_gte_lat', 'response_llama_3_8b_lat', 'response_llama_3_8b']
    elif dataset == "SemBenchmarkLmArena":
        columns_to_remove = ['dataset_name', 'emb_text-embedding-3-large', 'emb_text-embedding-3-large_lat','emb_text-embedding-3-small', 'emb_text-embedding-3-small_lat', 'response_gpt-4o-mini_lat', 'response_gpt-4.1-nano_lat', 'emb_gte', 'emb_gte_lat', 'emb_gte_ft', 'emb_gte_ft_lat', 'emb_e5_large_v2', 'emb_e5_large_v2_lat', 'emb_e5_large_v2_ft', 'emb_e5_large_v2_ft_lat']
    else:
        columns_to_remove = []
    ds["train"] = ds["train"].remove_columns(columns_to_remove)
    ds["train"] = ds["train"].map(lambda x: {**x, "prompt": pre_process_vector_prompt(x["prompt"])})
    return ds

if __name__ == '__main__':
    dataset = "SemBenchmarkLmArena"
    print(f"向量化处理测试{dataset}数据集")
    ds = pre_process_vector(dataset)
    print("处理后的列名：", ds["train"].column_names)