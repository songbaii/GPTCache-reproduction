from sentence_transformers import SentenceTransformer
from datasets import DatasetDict

def embed_sentences(sentences: list[str], embedding_model: str) -> list[list[float]]: # 做模型确定性嵌入测试
    if embedding_model == 'paraphrase-albert-small-v2':
        embedding_model_real = 'sentence-transformers/paraphrase-albert-small-v2'
    elif embedding_model == 'e5-large-v2':
        embedding_model_real = 'intfloat/e5-large-v2'
    elif embedding_model == 'gte-large-en-v1.5':
        embedding_model_real = 'Ceceliachenen/gte-large-en-v1.5'
    else:
        raise ValueError("不支持的embedding_model")
    model = SentenceTransformer(embedding_model_real, trust_remote_code=True)
    embeddings = model.encode(sentences, normalize_embeddings=True)
    return embeddings

def embed_ds(ds: DatasetDict, embedding_model: str)-> DatasetDict:
    # embedding_model可以选择的有
    # 'sentence-transformers/paraphrase-albert-small-v2' check
    # 'Alibaba-NLP/gte-large-en-v1.5',暂时不知道为什么不能使用
    # 'intfloat/e5-large-v2' check 
    if embedding_model == 'paraphrase-albert-small-v2':
        embedding_model_real = 'sentence-transformers/paraphrase-albert-small-v2'
    elif embedding_model == 'e5-large-v2':
        embedding_model_real = 'intfloat/e5-large-v2'
    elif embedding_model == 'gte-large-en-v1.5':
        embedding_model_real = 'Alibaba-NLP/gte-large-en-v1.5'
    else:
        raise ValueError("不支持的embedding_model")
    model = SentenceTransformer(embedding_model_real, trust_remote_code=True)
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
    embedding_model = 'gte-large-en-v1.5'
    embeddings = embed_sentences(sentences, embedding_model)
    embedding_model = 'paraphrase-albert-small-v2'
    embeddings2 = embed_sentences(sentences, embedding_model)
    if np.array_equal(embeddings[0], embeddings2[0]): # 是确定嵌入模型的输出是相同的
        print("嵌入模型输出相同的句子得到相同的嵌入")
    else:
        print("嵌入模型输出相同的句子得到不同的嵌入")
    print(embeddings)