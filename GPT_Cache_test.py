import pre_process
import embedding 
import vector_database
import sqlite3
import os
from datasets import load_from_disk

if __name__ == '__main__':
    dataset = "SemBenchmarkClassificationSorted"
    milvus_db_name  = "milvus_gpt_cache.db"
    sqllite_db_name = "sqlite_gpt_cache.db"
    collection_name = "classification_sorted_collection"
    dimension = 768
    ds = pre_process.pre_process_vector(dataset)
    print("处理后的列名：", ds["train"].column_names)
    dir_path = os.path.dirname(os.path.abspath(__file__))
    dir_path = dir_path.replace("\\", "/")
    if os.path.isdir(rf"{dir_path}/data/{dataset}_embedding"):
        ds = load_from_disk(rf"{dir_path}/data/{dataset}_embedding")
    else:
        ds = embedding.embed_ds(ds)
        ds.save_to_disk(rf"{dir_path}/data/{dataset}_embedding")
    print("处理后的列名：", ds["train"].column_names)
    client = vector_database.create_milvus_db(rf"{dir_path}/{milvus_db_name}")
    vector_database.create_collection(collection_name, client, dimension)
    conn = sqlite3.connect(rf"{dir_path}/{sqllite_db_name}")
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS gpt_cache")
    cursor.execute("CREATE TABLE gpt_cache (id INTEGER PRIMARY KEY, prompt TEXT)")
    conn.commit()
    
    '''for i in range(0, 1):
        # 查找在向量库中有没有相似的向量
        query_embedding = ds["train"][i]["embedding"]
        similar_ids = vector_database.search_collection(client, collection_name, query_embedding, top_k=1, threshold=0.7)
        if similar_ids:
            print(f"找到相似的向量，ID: {similar_ids[0]}")
        else:
            print("没有找到相似的向量，插入新的向量")
            vector_database.insert_into_collection(client, collection_name, [query_embedding], [i])
            cursor.execute("INSERT INTO gpt_cache (id, prompt) VALUES (?, ?)", (i, ds["train"][i]["prompt"]))
            conn.commit()
    conn.close()
    cursor.close()'''
    
   
