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
    if os.path.isdir(f"D:\\毕设\\lab\\GPTCache-reproduction\\data\\{dataset}_embedding"):
        ds = load_from_disk(f"D:\\毕设\\lab\\GPTCache-reproduction\\data\\{dataset}_embedding")
    else:
        ds = embedding.embed_ds(ds)
        ds.save_to_disk(f"D:\\毕设\\lab\\GPTCache-reproduction\\data\\{dataset}_embedding")
    print("处理后的列名：", ds["train"].column_names)
    client = vector_database.create_milvus_db(f"D:\\毕设\\lab\\GPTCache-reproduction\\{milvus_db_name}")
    vector_database.create_collection(client, collection_name, dimension)
    conn = sqlite3.connect(f"D:\\毕设\\lab\\GPTCache-reproduction\\{sqllite_db_name}")
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS gpt_cache")
    cursor.execute("CREATE TABLE gpt_cache (id INTEGER PRIMARY KEY, prompt TEXT)")
    conn.commit()
    #for i in range(len(ds["train"])):


