from pymilvus import MilvusClient, DataType

def create_milvus_db(db_name: str)-> MilvusClient: # 进行连接，若不存在则创建
    client = MilvusClient(db_name)
    return client

# 这里应该也没有问题
def create_collection(collection_name: str, client: MilvusClient, dimension: int):
    schema = client.create_schema()
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=dimension)
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="embedding", index_type="FLAT", metric_type="COSINE")
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)
    client.create_collection(collection_name, schema=schema, index_params=index_params)

# 检索向量，返回相似id列表
def single_search_collection(client: MilvusClient, collection_name: str, query_embedding: list, threshold: float = 0.7):
    # Milvus 查询接口
    results = client.search(
        collection_name=collection_name,
        data=[query_embedding],
        limit=1,
        anns_field="embedding",
        search_params={"metric_type": "COSINE"}
    )
    ids = []
    if results and results[0] and results[0][0].score >= threshold:
        ids.append(results[0][0].id)
    return ids

# 插入向量和id到集合, 应该没问题
def insert_into_collection(client: MilvusClient, collection_name: str, embeddings: list, ids: list):
    # embeddings: List[List[float]]
    # ids: List[int] 或 List[str]
    entities = [{"id": id_, "embedding": emb} for id_, emb in zip(ids, embeddings)]
    client.insert(collection_name=collection_name, data=entities)
    client.flush(collection_name=collection_name)  # 确保数据被写入磁盘

if __name__ == '__main__':
    import os
    test_db_name = "test_milvus.db"
    test_collection_name = "test_collection"
    dimension = 768
    dir_path = os.path.dirname(os.path.abspath(__file__))
    dir_path = dir_path.replace("\\", "/")
    client = create_milvus_db(rf"{dir_path}/{test_db_name}")
    create_collection(test_collection_name, client, dimension)
    import numpy as np
    print("开始 Milvus 测试")
    # 构造测试向量和 id
    test_id = 12345
    test_vector = np.random.rand(dimension).tolist()
    # 插入
    insert_into_collection(client, test_collection_name, [test_vector], [test_id])
    print("插入完成")
    # 检索
    result_ids = single_search_collection(client, test_collection_name, test_vector, threshold=0.99)
    print("检索结果：", result_ids)
    # 判断
    if test_id in result_ids:
        print("Milvus 测试通过")
    else:
        print("Milvus 测试失败")
    os.remove(rf"{dir_path}/{test_db_name}")