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
            ds = ds.map(lambda x: {**x, "embedding": model.encode(x["prompt"], normalize_embeddings=True)}, batched=True)
            ds["train"].remove_columns(["prompt"])
            ds.save_to_disk(path)
            return ds


if __name__ == "__main__":
    import numpy as np
    sentences = [
        "The cat is on the roof.",
        "The cat is on the roof.",
        "A dog is in the garden.",
        "The sun is shining brightly."
    ]
    gte_large_en_v1_5_embedding_model = embedding_generator('gte-large-en-v1.5')
    paraphrase_albert_small_v2_embedding_model = embedding_generator('paraphrase-albert-small-v2')
    embeddings1 = gte_large_en_v1_5_embedding_model.embed_sentences(sentences)
    embeddings2 = paraphrase_albert_small_v2_embedding_model.embed_sentences(sentences)
    if np.array_equal(embeddings1[0], embeddings2[0]): # 是确定嵌入模型的输出是相同的
        print("嵌入模型输出相同的句子得到相同的嵌入")
    else:
        print("嵌入模型输出相同的句子得到不同的嵌入")
    print(embeddings1)