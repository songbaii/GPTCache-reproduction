from sentence_transformers import SentenceTransformer
from datasets import DatasetDict, load_from_disk
import os

class embedding_generator:
    def __init__(self, embedding_model: str):
        self.embedding_model = embedding_model
        if embedding_model == 'paraphrase-albert-small-v2':
            self.dimension = 768
        elif embedding_model == 'e5-large-v2':
            self.dimension = 1024
        elif embedding_model == 'gte-large-en-v1.5':
            self.dimension = 1024
        else:
            raise ValueError("不支持的embedding_model")

    def embed_sentences(self, sentences: list[str]) -> list[list[float]]: # 做模型确定性嵌入测试
        if self.embedding_model == 'paraphrase-albert-small-v2':
            embedding_model_real = 'sentence-transformers/paraphrase-albert-small-v2'
        elif self.embedding_model == 'e5-large-v2':
            embedding_model_real = 'intfloat/e5-large-v2'
        elif self.embedding_model == 'gte-large-en-v1.5':
            embedding_model_real = 'Alibaba-NLP/gte-large-en-v1.5'
        else:
            raise ValueError("不支持的embedding_model")
        model = SentenceTransformer(embedding_model_real, trust_remote_code=True)
        embeddings = model.encode(sentences, normalize_embeddings=True)
        return embeddings

    def embed_ds(self, ds: DatasetDict, path: str)-> DatasetDict:
        # embedding_model可以选择的有
        # 'sentence-transformers/paraphrase-albert-small-v2' check
        # 'Alibaba-NLP/gte-large-en-v1.5',暂时不知道为什么不能使用
        # 'intfloat/e5-large-v2' check 
        if os.path.isdir(path):
            return load_from_disk(path)
        else:
            if self.embedding_model == 'paraphrase-albert-small-v2':
                embedding_model_real = 'sentence-transformers/paraphrase-albert-small-v2'
            elif self.embedding_model == 'e5-large-v2':
                embedding_model_real = 'intfloat/e5-large-v2'
            elif self.embedding_model == 'gte-large-en-v1.5':
                embedding_model_real = 'Alibaba-NLP/gte-large-en-v1.5'
            else:
                raise ValueError("不支持的embedding_model")
            model = SentenceTransformer(embedding_model_real, trust_remote_code=True)
            ds = ds.map(lambda x: {**x, "embedding": model.encode(x["prompt"], normalize_embeddings=True, batch_size=32)}, batched=True, batch_size=16)
            ds["train"] = ds["train"].remove_columns(["prompt"])
            ds.save_to_disk(path)
            return ds


if __name__ == "__main__":
    import os
    from pre_process import pre_processor
    dataset = ["SemBenchmarkClassificationSorted", "SemBenchmarkLmArena", "SemBenchmarkSearchQueries"]
    # SemBenchmarkClassificationSorted
    # SemBenchmarkLmArena
    # SemBenchmarkSearchQueries
    embedding_model = "e5-large-v2"
    # 'paraphrase-albert-small-v2' check
    # 'gte-large-en-v1.5',暂时不知道为什么不能使用
    # 'e5-large-v2' check 
    embedder = embedding_generator(embedding_model)
    dir_path = os.path.dirname(os.path.abspath(__file__))

    for ds_name in dataset:
        processor = pre_processor(ds_name)
        ds = processor.pre_process_vector()
        print(f"开始嵌入 {ds_name} 数据集...", flush=True)
        ds = embedder.embed_ds(ds, rf"{dir_path}/data/{ds_name}_{embedding_model}_embedding")
        print(f"{ds_name} 处理完毕", flush=True)