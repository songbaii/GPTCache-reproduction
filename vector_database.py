from pymilvus import MilvusClient

def create_milvus_db(db_name: str)-> MilvusClient:
    client = MilvusClient(db_name)
    return client

def create_collection(collection_name: str, client: MilvusClient, dimension: int):
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)
    client.create_collection(collection_name, dimension=dimension)

if __name__ == '__main__':
    test_db_name = "test_milvus.db"
    test_collection_name = "test_collection"
    dimension = 768
    client = create_milvus_db(f"D:\\毕设\\lab\\GPTCache-reproduction\\{test_db_name}")
    create_collection(test_collection_name, client, dimension)