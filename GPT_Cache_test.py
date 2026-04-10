import pre_process
import embedding 
import vector_database
import sqlite3
import os
from datasets import load_from_disk
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dataset = "SemBenchmarkClassificationSorted"
    milvus_db_name  = "milvus_gpt_cache.db"
    sqllite_db_name = "sqlite_gpt_cache.db"
    collection_name = "classification_sorted_collection"
    embedding_model = 'sentence-transformers/paraphrase-albert-small-v2'
    dimension = 768
    ds = pre_process.pre_process_vector(dataset)
    print("处理后的列名：", ds["train"].column_names)
    dir_path = os.path.dirname(os.path.abspath(__file__))
    dir_path = dir_path.replace("\\", "/")
    if os.path.isdir(rf"{dir_path}/data/{dataset}_embedding"):
        ds = load_from_disk(rf"{dir_path}/data/{dataset}_embedding")
    else:
        ds = embedding.embed_ds(ds, embedding_model)
        ds["train"].remove_columns(["prompt"])
        ds.save_to_disk(rf"{dir_path}/data/{dataset}_embedding")
    print("处理后的列名：", ds["train"].column_names)
    client = vector_database.create_milvus_db(rf"{dir_path}/{milvus_db_name}")
    vector_database.create_collection(collection_name, client, dimension)
    conn = sqlite3.connect(rf"{dir_path}/{sqllite_db_name}")
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS gpt_cache")
    cursor.execute("CREATE TABLE gpt_cache (id INTEGER PRIMARY KEY, response TEXT)")
    conn.commit()
    cache_hit = 0
    right_hit = 0
    miss = 0
    hit_rate = []
    sample_counts = []
    for i in range(len(ds["train"])):
        # 查找在向量库中有没有相似的向量
        query_embedding = ds["train"][i]["embedding"]
        similar_ids = vector_database.single_search_collection(client, collection_name, query_embedding, threshold=0.86)
        if similar_ids:
            conn.commit()
            cache_hit += 1
            cursor.execute("SELECT response FROM gpt_cache WHERE id=?", (similar_ids[0],))
            cached_response = cursor.fetchone()
            if cached_response[0] == ds["train"][i]["response_llama_3_8b"]:
                right_hit += 1
        else:
            miss += 1
            vector_database.insert_into_collection(client, collection_name, [query_embedding], [i])
            cursor.execute("INSERT INTO gpt_cache (id, response) VALUES (?, ?)", (i, ds["train"][i]["response_llama_3_8b"]))
        hit_rate.append(cache_hit / (cache_hit + miss))
        sample_counts.append(i + 1)
    cursor.close()
    conn.close()
    print(f"缓存命中: {cache_hit}, 正确命中: {right_hit}, 未命中: {miss}")
    print(f"缓存命中率: {cache_hit / len(ds['train']):.2%}, 正确命中率: {right_hit / cache_hit if cache_hit > 0 else 0:.2%}")
    # 绘制对应的缓存命中率图像
    plt.figure(figsize=(10, 6))
    plt.plot(sample_counts, hit_rate, label='Cache Hit Rate', color='blue')
    plt.xlabel('Number of Samples')
    plt.ylabel('Cache Hit Rate')
    plt.ylim([0, 1])
    plt.xlim([0, len(ds['train'])])
    plt.title('Cache Hit Rate Over Time')
    plt.legend()
    plt.grid()
    plt.savefig(rf"{dir_path}/result/cache_hit_rate.png")
   
