import pre_process
import embedding 
import vector_database
import sqlite3
import os
from datasets import load_from_disk
import picture

if __name__ == '__main__':
    dataset = "SemBenchmarkSearchQueries"
    # 可用dataset
    # SemBenchmarkSearchQueries
    # SemBenchmarkClassificationSorted
    # SemBenchmarkLmArena
    milvus_db_name  = "milvus_gpt_cache.db"
    sqllite_db_name = "sqlite_gpt_cache.db"
    collection_name = dataset + "_collection"
    table_name = dataset + "_table"
    embedding_model = 'paraphrase-albert-small-v2'
    if embedding_model == 'paraphrase-albert-small-v2':
        dimension = 768
    elif embedding_model == 'e5-large-v2':
        dimension = 1024
    elif embedding_model == 'gte-large-en-v1.5':
        dimension = 1024
    else:
        raise ValueError("不支持的embedding_model")
    ds = pre_process.pre_process_vector(dataset)
    print("处理后的列名：", ds["train"].column_names)
    dir_path = os.path.dirname(os.path.abspath(__file__))
    dir_path = dir_path.replace("\\", "/")
    if os.path.isdir(rf"{dir_path}/data/{dataset}_{embedding_model}_embedding"):
        ds = load_from_disk(rf"{dir_path}/data/{dataset}_{embedding_model}_embedding")
    else:
        ds = embedding.embed_ds(ds, embedding_model)
        ds["train"].remove_columns(["prompt"])
        ds.save_to_disk(rf"{dir_path}/data/{dataset}_{embedding_model}_embedding")
    print("处理后的列名：", ds["train"].column_names)
    client = vector_database.create_milvus_db(rf"{dir_path}/{milvus_db_name}")
    vector_database.create_collection(collection_name, client, dimension)
    conn = sqlite3.connect(rf"{dir_path}/{sqllite_db_name}")
    cursor = conn.cursor()
    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
    if dataset == "SemBenchmarkLmArena":
        cursor.execute(f"CREATE TABLE {table_name} (id INTEGER PRIMARY KEY, id_set INTEGER)")
        conn.commit()
        cache_hit = 0
        right_hit = 0
        miss = 0
        hit_rate = []
        sample_counts = []
        error_rate = []
        for i in range(len(ds["train"])):
            # 查找在向量库中有没有相似的向量
            query_embedding = ds["train"][i]["embedding"]
            similar_ids = vector_database.single_search_collection(client, collection_name, query_embedding, threshold=0.86)
            if similar_ids:
                conn.commit()
                cache_hit += 1
                cursor.execute(f"SELECT id_set FROM {table_name} WHERE id=?", (similar_ids[0],))
                cached_response = cursor.fetchone()
                if cached_response[0] == ds["train"][i]["ID_Set"]:
                    right_hit += 1
            else:
                miss += 1
                vector_database.insert_into_collection(client, collection_name, [query_embedding], [i])
                cursor.execute(f"INSERT INTO {table_name} (id, id_set) VALUES (?, ?)", (i, ds["train"][i]["ID_Set"]))
            hit_rate.append(cache_hit / (cache_hit + miss))
            error_rate.append((cache_hit - right_hit) / (cache_hit + miss))
            sample_counts.append(i + 1)
        cursor.close()
        conn.close()
        print(f"缓存命中: {cache_hit}, 正确命中: {right_hit}, 未命中: {miss}")
        print(f"缓存命中率: {cache_hit / len(ds['train']):.2%}, 正确命中率: {right_hit / cache_hit if cache_hit > 0 else 0:.2%}")
        # 绘制对应的缓存命中率图像
        path = os.path.join(os.path.dirname(__file__), "pictures")
        os.makedirs(path, exist_ok=True)
        picture.plot_error_rate(sample_counts, error_rate, os.path.join(path, f"{dataset}_cache_error_rate.png"))
    elif dataset == "SemBenchmarkClassificationSorted":
        cursor.execute(f"CREATE TABLE {table_name} (id INTEGER PRIMARY KEY, response TEXT)")
        conn.commit()
        cache_hit = 0
        right_hit = 0
        miss = 0
        hit_rate = []
        sample_counts = []
        error_rate = []
        for i in range(len(ds["train"])):
            # 查找在向量库中有没有相似的向量
            query_embedding = ds["train"][i]["embedding"]
            similar_ids = vector_database.single_search_collection(client, collection_name, query_embedding, threshold=0.86)
            if similar_ids:
                conn.commit()
                cache_hit += 1
                cursor.execute(f"SELECT response FROM {table_name} WHERE id=?", (similar_ids[0],))
                cached_response = cursor.fetchone()
                if cached_response[0] == ds["train"][i]["response_llama_3_8b"]:
                    right_hit += 1
            else:
                miss += 1
                vector_database.insert_into_collection(client, collection_name, [query_embedding], [i])
                cursor.execute(f"INSERT INTO {table_name} (id, response) VALUES (?, ?)", (i, ds["train"][i]["response_llama_3_8b"]))
            hit_rate.append(cache_hit / (cache_hit + miss))
            error_rate.append((cache_hit - right_hit) / (cache_hit + miss))
            sample_counts.append(i + 1)
        cursor.close()
        conn.close()
        print(f"缓存命中: {cache_hit}, 正确命中: {right_hit}, 未命中: {miss}")
        print(f"缓存命中率: {cache_hit / len(ds['train']):.2%}, 正确命中率: {right_hit / cache_hit if cache_hit > 0 else 0:.2%}")
        # 绘制对应的缓存命中率图像
        path = os.path.join(os.path.dirname(__file__), "pictures")
        os.makedirs(path, exist_ok=True)
        picture.plot_error_rate(sample_counts, error_rate, os.path.join(path, f"{dataset}_cache_error_rate.png"))
    elif dataset == "SemBenchmarkSearchQueries":
        cursor.execute(f"CREATE TABLE {table_name} (id INTEGER PRIMARY KEY, id_set INTEGER, cluster_id INTEGER)")
        conn.commit()
        cache_hit = 0
        right_hit = 0
        miss = 0
        hit_rate = []
        sample_counts = []
        error_rate = []
        for i in range(len(ds["train"])):
            # 查找在向量库中有没有相似的向量
            query_embedding = ds["train"][i]["embedding"]
            similar_ids = vector_database.single_search_collection(client, collection_name, query_embedding, threshold=0.86)
            if similar_ids:
                conn.commit()
                cache_hit += 1
                cursor.execute(f"SELECT id_set, cluster_id FROM {table_name} WHERE id=?", (similar_ids[0],))
                cached_response = cursor.fetchone()
                if cached_response[0] == ds["train"][i]["id_set"] and cached_response[1] == ds["train"][i]["cluster_id"]:
                    right_hit += 1
            else:
                miss += 1
                vector_database.insert_into_collection(client, collection_name, [query_embedding], [i])
                cursor.execute(f"INSERT INTO {table_name} (id, id_set, cluster_id) VALUES (?, ?, ?)", (i, ds["train"][i]["id_set"], ds["train"][i]["cluster_id"]))
            hit_rate.append(cache_hit / (cache_hit + miss))
            error_rate.append((cache_hit - right_hit) / (cache_hit + miss))
            sample_counts.append(i + 1)
        cursor.close()
        conn.close()
        print(f"缓存命中: {cache_hit}, 正确命中: {right_hit}, 未命中: {miss}")
        print(f"缓存命中率: {cache_hit / len(ds['train']):.2%}, 正确命中率: {right_hit / cache_hit if cache_hit > 0 else 0:.2%}")
        # 绘制对应的缓存命中率图像
        path = os.path.join(os.path.dirname(__file__), "pictures")
        os.makedirs(path, exist_ok=True)
        picture.plot_error_rate(sample_counts, error_rate, os.path.join(path, f"{dataset}_cache_error_rate.png"))
   
