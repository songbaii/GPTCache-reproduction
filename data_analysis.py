from list_store import list_store
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

class different_cache_compare:
    def __init__(self, dataset: str, embedding: str):
        dir_path = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
        list_store_GPTcache_cos = list_store(rf"{dir_path}/data/{dataset}_{embedding}_gpt_get_list_cache.json")
        self.db_GPTcache_cos = list_store_GPTcache_cos.load_list()
        list_store_GPTcache_new = list_store(rf"{dir_path}/data/{dataset}_{embedding}_gpt_new_get_list_cache.json")
        self.db_GPTcache_new = list_store_GPTcache_new.load_list()
        list_store_vCache_01 = list_store(rf"{dir_path}/data/{dataset}_{embedding}_SimpleVCache_delta = 0.1_vcache_get_list_cache.json")
        self.db_vCache_01 = list_store_vCache_01.load_list()
        list_store_vCache_0015 = list_store(rf"{dir_path}/data/{dataset}_{embedding}_SimpleVCache_delta = 0.015_vcache_get_list_cache.json")
        self.db_vCache_0015 = list_store_vCache_0015.load_list()
        self.dataset = dataset
        self.embedding = embedding
        self.dir_path = dir_path

    def hit_rate_compare(self):
        sample_counts = self.db_GPTcache_cos[0]
        GPT_cache_cos_hit_rate = self.db_GPTcache_cos[1]
        GPT_cache_new_hit_rate = self.db_GPTcache_new[1]
        vCache_01_hit_rate = self.db_vCache_01[1]
        vCache_0015_hit_rate = self.db_vCache_0015[1]
        plt.figure(figsize=(12, 6))
        plt.plot(sample_counts, GPT_cache_cos_hit_rate, label='GPT Cache Cosine Hit Rate', color='blue')
        plt.plot(sample_counts, GPT_cache_new_hit_rate, label='GPT Cache LLM Hit Rate', color='cyan')
        plt.plot(sample_counts, vCache_01_hit_rate, label='vCache Delta=0.1 Hit Rate', color='orange')
        plt.plot(sample_counts, vCache_0015_hit_rate, label='vCache Delta=0.015 Hit Rate', color='green')
        plt.xlabel('Number of Samples')
        plt.ylabel('Cache Hit Rate')
        plt.grid()
        plt.legend()
        plt.savefig(rf"{self.dir_path}/pictures/final/{self.dataset}_{self.embedding}_hit_rate_comparison.png")

    def error_rate_compare(self):
        sample_counts = self.db_GPTcache_cos[0]
        GPT_cache_cos_error_rate = self.db_GPTcache_cos[2]
        GPT_cache_new_error_rate = self.db_GPTcache_new[2]
        vCache_01_error_rate = self.db_vCache_01[2]
        vCache_0015_error_rate = self.db_vCache_0015[2]
        plt.figure(figsize=(12, 6))
        plt.plot(sample_counts, GPT_cache_cos_error_rate, label='GPT Cache Cosine Error Rate', color='blue')
        plt.plot(sample_counts, GPT_cache_new_error_rate, label='GPT Cache LLM Error Rate', color='cyan')
        plt.plot(sample_counts, vCache_01_error_rate, label='vCache Delta=0.1 Error Rate', color='orange')
        plt.plot(sample_counts, vCache_0015_error_rate, label='vCache Delta=0.015 Error Rate', color='green')
        plt.axhline(0.1, color='red', linestyle='--', label='Error Rate = 0.1')
        plt.axhline(0.015, color='purple', linestyle='--', label='Error Rate = 0.015')
        plt.xlabel('Number of Samples')
        plt.ylabel('Cache Error Rate')
        plt.legend()
        plt.grid()
        plt.savefig(rf"{self.dir_path}/pictures/final/{self.dataset}_{self.embedding}_error_rate_comparison.png")

    def compare_all(self):
        smaple_counts = self.db_GPTcache_cos[0]
        GPT_cache_cos_hit_rate = self.db_GPTcache_cos[1]
        GPT_cache_new_hit_rate = self.db_GPTcache_new[1]
        vCache_01_hit_rate = self.db_vCache_01[1]
        vCache_0015_hit_rate = self.db_vCache_0015[1]
        GPT_cache_cos_error_rate = self.db_GPTcache_cos[2]
        GPT_cache_new_error_rate = self.db_GPTcache_new[2]
        vCache_01_error_rate = self.db_vCache_01[2]
        vCache_0015_error_rate = self.db_vCache_0015[2]
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(smaple_counts, GPT_cache_cos_hit_rate, label='GPT Cache Cosine Hit Rate', color='blue')
        plt.plot(smaple_counts, GPT_cache_new_hit_rate, label='GPT Cache LLM Hit Rate', color='cyan')
        plt.plot(smaple_counts, vCache_01_hit_rate, label='vCache Delta=0.1 Hit Rate', color='orange')
        plt.plot(smaple_counts, vCache_0015_hit_rate, label='vCache Delta=0.015 Hit Rate', color='green')
        plt.xlabel('Number of Samples')
        plt.ylabel('Cache Hit Rate')
        plt.grid()
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(smaple_counts, GPT_cache_cos_error_rate, label='GPT Cache Cosine Error Rate', color='blue')
        plt.plot(smaple_counts, GPT_cache_new_error_rate, label='GPT Cache LLM Error Rate', color='cyan')
        plt.plot(smaple_counts, vCache_01_error_rate, label='vCache Delta=0.1 Error Rate', color='orange')
        plt.plot(smaple_counts, vCache_0015_error_rate, label='vCache Delta=0.015 Error Rate', color='green')
        plt.axhline(0.1, color='red', linestyle='--', label='Error Rate = 0.1')
        plt.axhline(0.015, color='purple', linestyle='--', label='Error Rate = 0.015')
        plt.xlabel('Number of Samples')
        plt.ylabel('Cache Error Rate')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(rf"{self.dir_path}/pictures/final/{self.dataset}_{self.embedding}_hit_error_rate_comparison.png")

class compare_by_embedding_all:
    def __init__(self, embedding: str):
        datasets = ["SemBenchmarkClassificationSorted", "SemBenchmarkLmArena", "SemBenchmarkSearchQueries"]
        test = []
        for dataset in datasets:
            test.append([different_cache_compare(dataset, embedding).db_GPTcache_cos, different_cache_compare(dataset, embedding).db_GPTcache_new, different_cache_compare(dataset, embedding).db_vCache_01, different_cache_compare(dataset, embedding).db_vCache_0015])
        fig = plt.figure(figsize=(16, 8))
        title_size = 20
        left_label_size = title_size - 2
        x_size = title_size - 4
        y_size = title_size - 4
        plt.subplot(2, 3, 1)
        plt.plot(test[0][0][0], test[0][0][2], label='GPT Semantic Cache', color='blue')
        plt.plot(test[0][1][0], test[0][1][2], label='GPT Cache', color='cyan')
        plt.plot(test[0][2][0], test[0][2][2], label='vCache Delta=0.1', color='orange')
        plt.plot(test[0][3][0], test[0][3][2], label='vCache Delta=0.015', color='green')
        plt.axhline(0.1, color='red', linestyle='--', label='Error Rate = 0.1')
        plt.axhline(0.015, color='purple', linestyle='--', label='Error Rate = 0.015')
        plt.ylabel('Cache Error Rate', fontsize=left_label_size, fontweight='bold')
        plt.title(rf'SemBenchmarkClassificationSorted', fontsize=title_size)
        plt.xticks(fontsize=x_size)
        plt.yticks(fontsize=y_size)
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.subplot(2, 3, 2)
        plt.plot(test[1][0][0], test[1][0][2], color='blue')
        plt.plot(test[1][1][0], test[1][1][2], color='cyan')
        plt.plot(test[1][2][0], test[1][2][2], color='orange')
        plt.plot(test[1][3][0], test[1][3][2], color='green')
        plt.axhline(0.1, color='red', linestyle='--')
        plt.axhline(0.015, color='purple', linestyle='--')
        plt.title(rf'SemBenchmarkLmArena', fontsize=title_size)
        plt.xticks(fontsize=x_size)
        plt.yticks(fontsize=y_size)
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(5))
        plt.subplot(2, 3, 3)
        plt.plot(test[2][0][0], test[2][0][2], color='blue')
        plt.plot(test[2][1][0], test[2][1][2], color='cyan')
        plt.plot(test[2][2][0], test[2][2][2], color='orange')
        plt.plot(test[2][3][0], test[2][3][2], color='green')
        plt.axhline(0.1, color='red', linestyle='--')
        plt.axhline(0.015, color='purple', linestyle='--')
        plt.title(rf'SemBenchmarkSearchQueries', fontsize=title_size)
        plt.xticks(fontsize=x_size)
        plt.yticks(fontsize=y_size)
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(5))
        plt.subplot(2, 3, 4)
        plt.plot(test[0][0][0], test[0][0][1], color='blue')
        plt.plot(test[0][1][0], test[0][1][1], color='cyan')
        plt.plot(test[0][2][0], test[0][2][1], color='orange')
        plt.plot(test[0][3][0], test[0][3][1], color='green')
        plt.xlabel('Number of Queries', fontsize=left_label_size)
        plt.ylabel('Cache hit Rate', fontsize=left_label_size, fontweight='bold')
        plt.xticks(fontsize=x_size)
        plt.yticks(fontsize=y_size)
        plt.subplot(2, 3, 5)
        plt.plot(test[1][0][0], test[1][0][1], color='blue')
        plt.plot(test[1][1][0], test[1][1][1], color='cyan')
        plt.plot(test[1][2][0], test[1][2][1], color='orange')
        plt.plot(test[1][3][0], test[1][3][1], color='green')
        plt.xlabel('Number of Queries', fontsize=left_label_size)
        plt.xticks(fontsize=x_size)
        plt.yticks(fontsize=y_size)
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(5))
        plt.subplot(2, 3, 6)
        plt.plot(test[2][0][0], test[2][0][1], color='blue')
        plt.plot(test[2][1][0], test[2][1][1], color='cyan')
        plt.plot(test[2][2][0], test[2][2][1], color='orange')
        plt.plot(test[2][3][0], test[2][3][1], color='green')
        plt.xlabel('Number of Queries', fontsize=left_label_size)
        plt.xticks(fontsize=x_size)
        plt.yticks(fontsize=y_size)
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(5))
        plt.tight_layout(rect=[0, 0, 1, 0.9])
        fig.legend(handles, labels, loc='upper center', ncol=3, fontsize = 18, bbox_to_anchor=(0.5, 1.01))
        dir_path = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
        plt.savefig(rf"{dir_path}/pictures/final/comparison_by_{embedding}.png")

class compare_by_cache_all:
    def __init__(self):
        datasets = ["SemBenchmarkClassificationSorted", "SemBenchmarkLmArena", "SemBenchmarkSearchQueries"]
        embeddings = ["paraphrase-albert-small-v2", "e5-large-v2"]
        test = []
        for embedding in embeddings:
            for dataset in datasets:
                test.append([different_cache_compare(dataset, embedding).db_GPTcache_cos, different_cache_compare(dataset, embedding).db_GPTcache_new, different_cache_compare(dataset, embedding).db_vCache_01, different_cache_compare(dataset, embedding).db_vCache_0015])
        title_size = 20
        left_label_size = title_size - 2
        x_size = title_size - 4
        y_size = title_size - 4
        fig = plt.figure(figsize=(16, 8))
        plt.subplot(2, 3, 1)
        plt.plot(test[0][0][0], test[0][0][2], label='GPT Semantic Cache at paraphrase-albert-small-v2', color='blue')
        plt.plot(test[0][1][0], test[0][1][2], label='GPT Cache at paraphrase-albert-small-v2', color='cyan')
        plt.plot(test[3][0][0], test[3][0][2], label='GPT Semantic Cache at e5-large-v2', color='orange')
        plt.plot(test[3][1][0], test[3][1][2], label='GPT Cache at e5-large-v2', color='green')
        plt.ylabel('Cache Error Rate', fontsize=left_label_size, fontweight='bold')
        plt.title(rf'SemBenchmarkClassificationSorted', fontsize=title_size)
        plt.xticks(fontsize=x_size)
        plt.yticks(fontsize=y_size)
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.subplot(2, 3, 2)
        plt.plot(test[1][0][0], test[1][0][2], color='blue')
        plt.plot(test[1][1][0], test[1][1][2], color='cyan')
        plt.plot(test[4][0][0], test[4][0][2], color='orange')
        plt.plot(test[4][1][0], test[4][1][2], color='green')
        plt.title(rf'SemBenchmarkLmArena', fontsize=title_size)
        plt.xticks(fontsize=x_size)
        plt.yticks(fontsize=y_size)
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(5))
        plt.subplot(2, 3, 3)
        plt.plot(test[2][0][0], test[2][0][2], color='blue')
        plt.plot(test[2][1][0], test[2][1][2], color='cyan')
        plt.plot(test[5][0][0], test[5][0][2], color='orange')
        plt.plot(test[5][1][0], test[5][1][2], color='green')
        plt.title(rf'SemBenchmarkSearchQueries', fontsize=title_size)
        plt.xticks(fontsize=x_size)
        plt.yticks(fontsize=y_size)
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(5))
        plt.subplot(2, 3, 4)
        plt.plot(test[0][0][0], test[0][0][1], color='blue')
        plt.plot(test[0][1][0], test[0][1][1], color='cyan')
        plt.plot(test[3][0][0], test[3][0][1], color='orange')
        plt.plot(test[3][1][0], test[3][1][1], color='green')
        plt.xlabel('Number of Queries', fontsize=left_label_size)
        plt.ylabel('Cache hit Rate', fontsize=y_size, fontweight='bold')
        plt.xticks(fontsize=x_size)
        plt.yticks(fontsize=y_size)
        plt.subplot(2, 3, 5)
        plt.plot(test[1][0][0], test[1][0][1], color='blue')
        plt.plot(test[1][1][0], test[1][1][1], color='cyan')
        plt.plot(test[4][0][0], test[4][0][1], color='orange')
        plt.plot(test[4][1][0], test[4][1][1], color='green')
        plt.xlabel('Number of Queries', fontsize=left_label_size)
        plt.xticks(fontsize=x_size)
        plt.yticks(fontsize=y_size)
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(5))
        plt.subplot(2, 3, 6)
        plt.plot(test[2][0][0], test[2][0][1], color='blue')
        plt.plot(test[2][1][0], test[2][1][1], color='cyan')
        plt.plot(test[5][0][0], test[5][0][1], color='orange')
        plt.plot(test[5][1][0], test[5][1][1], color='green')
        plt.xticks(fontsize=x_size)
        plt.yticks(fontsize=y_size)
        plt.xlabel('Number of Queries', fontsize=left_label_size)
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(5))
        plt.tight_layout(rect=[0, 0, 1, 0.9])
        fig.legend(handles, labels, loc='upper center', ncol=2, fontsize = 18, bbox_to_anchor=(0.5, 1.01))
        dir_path = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
        plt.savefig(rf"{dir_path}/pictures/final/comparison_by_cache_GPT.png")
        fig = plt.figure(figsize=(16, 8))
        plt.subplot(2, 3, 1)
        plt.plot(test[0][2][0], test[0][2][2], label='vCache Delta=0.1 at paraphrase-albert-small-v2', color='blue')
        plt.plot(test[0][3][0], test[0][3][2], label='vCache Delta=0.015 at paraphrase-albert-small-v2', color='cyan')
        plt.plot(test[3][2][0], test[3][2][2], label='vCache Delta=0.1 at e5-large-v2', color='orange')
        plt.plot(test[3][3][0], test[3][3][2], label='vCache Delta=0.015 at e5-large-v2', color='green')
        plt.ylabel('Cache Error Rate', fontsize=left_label_size, fontweight='bold')
        plt.title(rf'SemBenchmarkClassificationSorted', fontsize=title_size)
        plt.xticks(fontsize=x_size)
        plt.yticks(fontsize=y_size)
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.subplot(2, 3, 2)
        plt.plot(test[1][2][0], test[1][2][2], color='blue')
        plt.plot(test[1][3][0], test[1][3][2], color='cyan')
        plt.plot(test[4][2][0], test[4][2][2], color='orange')
        plt.plot(test[4][3][0], test[4][3][2], color='green')
        plt.title(rf'SemBenchmarkLmArena', fontsize=title_size)
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(5))
        plt.xticks(fontsize=x_size)
        plt.yticks(fontsize=y_size)
        plt.subplot(2, 3, 3)
        plt.plot(test[2][2][0], test[2][2][2], color='blue')
        plt.plot(test[2][3][0], test[2][3][2], color='cyan')
        plt.plot(test[5][2][0], test[5][2][2], color='orange')
        plt.plot(test[5][3][0], test[5][3][2], color='green')
        plt.title(rf'SemBenchmarkSearchQueries', fontsize=title_size)
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(5))
        plt.xticks(fontsize=x_size)
        plt.yticks(fontsize=y_size)
        plt.subplot(2, 3, 4)
        plt.plot(test[0][2][0], test[0][2][1], color='blue')
        plt.plot(test[0][3][0], test[0][3][1], color='cyan')
        plt.plot(test[3][2][0], test[3][2][1], color='orange')
        plt.plot(test[3][3][0], test[3][3][1], color='green')
        plt.xlabel('Number of Queries', fontsize=left_label_size)
        plt.ylabel('Cache hit Rate', fontsize=left_label_size, fontweight='bold')
        plt.xticks(fontsize=x_size)
        plt.yticks(fontsize=y_size)
        plt.subplot(2, 3, 5)
        plt.plot(test[1][2][0], test[1][2][1], color='blue')
        plt.plot(test[1][3][0], test[1][3][1], color='cyan')
        plt.plot(test[4][2][0], test[4][2][1], color='orange')
        plt.plot(test[4][3][0], test[4][3][1], color='green')
        plt.xlabel('Number of Queries', fontsize=left_label_size)
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(5))
        plt.xticks(fontsize=x_size)
        plt.yticks(fontsize=y_size)
        plt.subplot(2, 3, 6)
        plt.plot(test[2][2][0], test[2][2][1], color='blue')
        plt.plot(test[2][3][0], test[2][3][1], color='cyan')
        plt.plot(test[5][2][0], test[5][2][1], color='orange')
        plt.plot(test[5][3][0], test[5][3][1], color='green')
        plt.xlabel('Number of Queries', fontsize=left_label_size)
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(5))
        plt.xticks(fontsize=x_size)
        plt.yticks(fontsize=y_size)
        plt.tight_layout(rect=[0, 0, 1, 0.9])
        fig.legend(handles, labels, loc='upper center', ncol=2, fontsize = 18, bbox_to_anchor=(0.5, 1.01))
        plt.savefig(rf"{dir_path}/pictures/final/comparison_by_cache_vCache.png")

if __name__ == '__main__':
    '''
    dir_path = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
    os.makedirs(rf"{dir_path}/pictures", exist_ok=True)
    os.makedirs(rf"{dir_path}/pictures/final", exist_ok=True)
    datasets = ["SemBenchmarkClassificationSorted", "SemBenchmarkLmArena", "SemBenchmarkSearchQueries"]
    embeddings = ["paraphrase-albert-small-v2", "e5-large-v2"]
    for dataset in datasets:
        for embedding in embeddings:
            test = different_cache_compare(dataset, embedding)
            test.compare_all()'''
    test = compare_by_embedding_all("paraphrase-albert-small-v2")
    test = compare_by_embedding_all("e5-large-v2")
    test = compare_by_cache_all()