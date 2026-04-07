from sentence_transformers import SentenceTransformer
from datasets import DatasetDict

def embed_sentences(sentences: list[str]) -> list[list[float]]: # 做模型确定性嵌入测试
    model = SentenceTransformer('sentence-transformers/paraphrase-albert-small-v2')
    embeddings = model.encode(sentences, normalize_embeddings=True)
    return embeddings

def embed_ds(ds: DatasetDict)-> DatasetDict:
    model = SentenceTransformer('sentence-transformers/paraphrase-albert-small-v2')
    ds = ds.map(lambda x: {**x, "embedding": model.encode(x["prompt"], normalize_embeddings=True)}, batched=True)
    return ds


if __name__ == "__main__":
    import numpy as np
    sentences = [
        "The cat is on the roof.",
        "The cat is on the roof.",
        "A dog is in the garden.",
        "The sun is shining brightly."
    ]
    embeddings = embed_sentences(sentences)
    if np.array_equal(embeddings[0], embeddings[1]): # 是确定嵌入模型的输出是相同的
        print("嵌入模型输出相同的句子得到相同的嵌入")
    print(embeddings)