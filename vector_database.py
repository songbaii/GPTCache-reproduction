from pymilvus import MilvusClient, DataType

class milvus_bd:
    def __init__(self, db_name: str):
        self.client = MilvusClient(db_name)

    def create_collection(self, collection_name: str, dimension: int):
        schema = self.client.create_schema()
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=dimension)
        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="embedding", index_type="FLAT", metric_type="COSINE")
        if self.client.has_collection(collection_name):
            self.client.drop_collection(collection_name)
        self.client.create_collection(collection_name, schema=schema, index_params=index_params)

    def insert_into_collection(self, collection_name: str, embeddings: list, ids: list):
        entities = [{"id": id_, "embedding": emb} for id_, emb in zip(ids, embeddings)]
        self.client.insert(collection_name=collection_name, data=entities)
        self.client.flush(collection_name=collection_name)  # 确保数据被写入磁盘

    def single_search_collection(self, collection_name: str, query_embedding: list, threshold: float = 0.7):
         # Milvus 查询接口
        results = self.client.search(
        collection_name=collection_name,
        data=[query_embedding],
        limit=1,
        anns_field="embedding",
        search_params={"metric_type": "COSINE"}
        )
        ids = []
        if results and results[0] and results[0][0].score >= threshold:
            ids.append([results[0][0].id, results[0][0].score])
        return ids

if __name__ == '__main__':
    import os
    test_db_name = "test_milvus.db"
    test_collection_name = "test_collection"
    dimension = 768
    dir_path = os.path.dirname(os.path.abspath(__file__))
    dir_path = dir_path.replace("\\", "/")
    test_db = milvus_bd(rf"{dir_path}/{test_db_name}")
    test_db.create_collection(test_collection_name, dimension)
    import numpy as np
    print("开始 Milvus 测试")
    # 构造测试向量和 id
    test_id = 12345
    test_vector = np.random.rand(dimension).tolist()
    # 插入
    test_db.insert_into_collection(test_collection_name, [test_vector], [test_id])
    print("插入完成")
    # 检索
    result_ids = test_db.single_search_collection(test_collection_name, test_vector, threshold=0.99)
    print("检索结果：", result_ids)
    # 判断
    if test_id == result_ids[0][0]:
        print("Milvus 测试通过")
    else:
        print("Milvus 测试失败")
    os.remove(rf"{dir_path}/{test_db_name}")