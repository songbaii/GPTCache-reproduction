from pre_process import pre_process
import embedding 
from vector_database import milvus_db
import os
from datasets import load_from_disk
from picture import picture_generator
from my_sqllite3 import LMarenaSQLiteManager, ClassificationSortedSQLiteManager, SearchQueriesSQLiteManager, vcache_hit_record_SQLiteManager
from vcache_final import SimpleVCache

if __name__ == '__main__':
    dataset = "SemBenchmarkClassificationSorted"
    # 可用dataset
    # SemBenchmarkSearchQueries
    # SemBenchmarkClassificationSorted
    # SemBenchmarkLmArena
    milvus_db_name  = "milvus_cache.db"
    sqllite_db_name = "sqlite_cache.db"
    collection_name = dataset + "_collection"
    table_name = dataset + "_table"
    embedding_model = 'paraphrase-albert-small-v2'
    test_mode = 'GPTcache' # 'GPTcache' or 'vcache'
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
    now_milvus_db = milvus_db(rf"{dir_path}/{milvus_db_name}")
    now_milvus_db.create_collection(collection_name, dimension)
    cache_hit = 0
    right_hit = 0
    miss = 0
    hit_rate = []
    sample_counts = []
    error_rate = []
    if test_mode == 'GPTcache':
        if dataset == "SemBenchmarkLmArena":
            LMarena_db = LMarenaSQLiteManager(rf"{dir_path}/{sqllite_db_name}")
            for i in range(len(ds["train"])):
                # 查找在向量库中有没有相似的向量
                query_embedding = ds["train"][i]["embedding"]
                similar_ids = now_milvus_db.single_search_collection(collection_name, query_embedding, threshold=0.86)
                if similar_ids:
                    cache_hit += 1
                    if LMarena_db.search_by_id(similar_ids[0][0]) == ds["train"][i]["ID_Set"]:
                        right_hit += 1
                else:
                    miss += 1
                    now_milvus_db.insert_into_collection(collection_name, [query_embedding], [i])
                    LMarena_db.insert(i, ds["train"][i]["ID_Set"])
                hit_rate.append(cache_hit / (cache_hit + miss))
                error_rate.append((cache_hit - right_hit) / (cache_hit + miss))
                sample_counts.append(i + 1)
            LMarena_db.close()
        elif dataset == "SemBenchmarkClassificationSorted":
            ClassificationSorted_db = ClassificationSortedSQLiteManager(rf"{dir_path}/{sqllite_db_name}")
            for i in range(len(ds["train"])):
                # 查找在向量库中有没有相似的向量
                query_embedding = ds["train"][i]["embedding"]
                similar_ids = now_milvus_db.single_search_collection(collection_name, query_embedding, threshold=0.86)
                if similar_ids:
                    cache_hit += 1
                    if ClassificationSorted_db.search_by_id(similar_ids[0][0]) == ds["train"][i]["response_llama_3_8b"]:
                        right_hit += 1
                else:
                    miss += 1
                    now_milvus_db.insert_into_collection(collection_name, [query_embedding], [i])
                    ClassificationSorted_db.insert(i, ds["train"][i]["response_llama_3_8b"])   
                hit_rate.append(cache_hit / (cache_hit + miss))
                error_rate.append((cache_hit - right_hit) / (cache_hit + miss))
                sample_counts.append(i + 1)
            ClassificationSorted_db.close()
        elif dataset == "SemBenchmarkSearchQueries":
            SearchQueries_db = SearchQueriesSQLiteManager(rf"{dir_path}/{sqllite_db_name}")
            for i in range(len(ds["train"])):
                # 查找在向量库中有没有相似的向量
                query_embedding = ds["train"][i]["embedding"]
                similar_ids = now_milvus_db.single_search_collection(collection_name, query_embedding, threshold=0.86)
                if similar_ids:
                    cache_hit += 1
                    cached_response = SearchQueries_db.search_by_id(similar_ids[0][0])
                    if cached_response[0] == ds["train"][i]["id_set"] and cached_response[1] == ds["train"][i]["cluster_id"]:
                        right_hit += 1
                else:
                    miss += 1
                    now_milvus_db.insert_into_collection(collection_name, [query_embedding], [i])
                    SearchQueries_db.insert(i, ds["train"][i]["id_set"], ds["train"][i]["cluster_id"])
                hit_rate.append(cache_hit / (cache_hit + miss))
                error_rate.append((cache_hit - right_hit) / (cache_hit + miss))
                sample_counts.append(i + 1)
            SearchQueries_db.close()
    elif test_mode == 'vcache':
        error_rate_max = 0.1
        vcache_instance = SimpleVCache(delta=error_rate_max)
        vcache_hit = vcache_hit_record_SQLiteManager(rf"{dir_path}/{sqllite_db_name}")
        if dataset == "SemBenchmarkClassificationSorted":
            ClassificationSorted_db = ClassificationSortedSQLiteManager(rf"{dir_path}/{sqllite_db_name}")
            for i in range(len(ds["train"])):
                # 查找在向量库中有没有相似的向量
                query_embedding = ds["train"][i]["embedding"]
                similar_ids = now_milvus_db.single_search_collection(collection_name, query_embedding, threshold=-1)
                if similar_ids: # 如果向量库中有向量
                    s_vals, c_vals = vcache_hit.search_by_id(similar_ids[0][0])
                    if vcache_instance.decide(similar_ids[0][1], s_vals, c_vals) == "exploit":
                        cache_hit += 1
                        similar_id = similar_ids[0][0]
                        if ClassificationSorted_db.search_by_id(similar_ids[0][0]) == ds["train"][i]["response_llama_3_8b"]: # 命中且正确
                            right_hit += 1
                    else: # 调用后端大模型
                        miss += 1
                        if ClassificationSorted_db.search_by_id(similar_ids[0][0]) == ds["train"][i]["response_llama_3_8b"]:
                            s_vals.append(similar_ids[0][1])
                            c_vals.append(1)
                        else:
                            s_vals.append(similar_ids[0][1])
                            c_vals.append(0)
                            vcache_hit.add_or_update(i, [], [])
                            ClassificationSorted_db.insert(i, ds["train"][i]["response_llama_3_8b"])  
                            now_milvus_db.insert_into_collection(collection_name, [query_embedding], [i])
                        vcache_hit.add_or_update(similar_ids[0][0], s_vals, c_vals)
                else: # 如果向量库中没有向量，直接调用后端大模型
                    miss += 1
                    now_milvus_db.insert_into_collection(collection_name, [query_embedding], [i])
                    ClassificationSorted_db.insert(i, ds["train"][i]["response_llama_3_8b"])   
                    vcache_hit.add_or_update(i, [], [])
                hit_rate.append(cache_hit / (cache_hit + miss))
                error_rate.append((cache_hit - right_hit) / (cache_hit + miss))
                sample_counts.append(i + 1)
            ClassificationSorted_db.close()
    print(f"缓存命中: {cache_hit}, 正确命中: {right_hit}, 未命中: {miss}")
    print(f"缓存命中率: {cache_hit / len(ds['train']):.2%}, 正确命中率: {right_hit / cache_hit if cache_hit > 0 else 0:.2%}")
    # 绘制对应的缓存命中率图像
    path = os.path.join(os.path.dirname(__file__), "pictures")
    os.makedirs(path, exist_ok=True)
    pic_gen = picture_generator(sample_counts, hit_rate, error_rate)
    pic_gen.plot_error_rate(os.path.join(path, f"{dataset}_{test_mode}_cache_error_rate.png"))
    pic_gen.plot_hit_rate(os.path.join(path, f"{dataset}_{test_mode}_cache_hit_rate.png"))
    print("Cache plots saved.")
    

