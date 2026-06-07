import matplotlib.pyplot as plt
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False
from list_store import list_store
import os
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde
import matplotlib
import numpy as np
from collections import defaultdict

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
        fig = plt.figure(figsize=(16, 11))
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
        #plt.tight_layout(rect=[0, 0, 1, 0.9])
        fig.legend(handles, labels, loc='upper center', ncol=3, fontsize = 18, bbox_to_anchor=(0.5, 1))
        fig.suptitle(rf'{embedding} 嵌入下性能表现', fontsize=22, y=0.05)
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
        fig = plt.figure(figsize=(16, 11))
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
        #plt.tight_layout(rect=[0, 0, 1, 0.9])
        fig.legend(handles, labels, loc='upper center', ncol=2, fontsize = 18, bbox_to_anchor=(0.5, 1.01))
        dir_path = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
        fig.suptitle(rf'不同嵌入模型下 GPTcache 和GPT Semantic Cache 性能表现', fontsize=22, y=0.05)
        plt.savefig(rf"{dir_path}/pictures/final/comparison_by_cache_GPT.png")
        fig = plt.figure(figsize=(16, 11))
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
        #plt.tight_layout(rect=[0, 0, 1, 0.9])
        fig.legend(handles, labels, loc='upper center', ncol=2, fontsize = 18, bbox_to_anchor=(0.5, 1.005))
        fig.suptitle(rf'不同嵌入模型下 vCache 性能表现', fontsize=22, y=0.05)
        plt.savefig(rf"{dir_path}/pictures/final/comparison_by_cache_vCache.png")

class compare_by_embedding_similarity:
    def __init__(self):
        datasets = ["SemBenchmarkClassificationSorted", "SemBenchmarkLmArena", "SemBenchmarkSearchQueries"]
        embeddings = ["paraphrase-albert-small-v2", "e5-large-v2"]
        self.data = defaultdict(dict)
        self.dir_path = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
        for i, dataset in enumerate(datasets):
            for j, embedding in enumerate(embeddings):
                self.data[i][j] = list_store(rf"{self.dir_path}/data/{dataset}_{embedding}_gpt_kde_similarities_cache.json")

    def draw_kde(self):
        matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'SimHei', 'WenQuanYi Zen Hei']
        matplotlib.rcParams['axes.unicode_minus'] = False
        fig = plt.figure(figsize=(16, 10))

        plt.subplot(2, 3, 1)
        plt.title(rf'SemBenchmarkClassificationSorted', fontsize=20)
        right_similarities = self.data[0][0].load_list()[0]
        wrong_similarities = self.data[0][0].load_list()[1]
        kde_right = gaussian_kde(right_similarities, bw_method='scott')
        kde_wrong = gaussian_kde(wrong_similarities, bw_method='scott')
        x = np.linspace(0, 1, 1000)
        y_right = kde_right(x)
        y_wrong = kde_wrong(x)
        '''mean_right = np.mean(right_similarities)
        mean_wrong = np.mean(wrong_similarities)
        plt.axvline(mean_right, color='green', linestyle='--')
        plt.axvline(mean_wrong, color='red', linestyle='--')'''
        plt.plot(x, y_right, label='right hit', color='green')
        plt.plot(x, y_wrong, label='wrong hit', color='red')
        plt.ylabel('pdf(paraphrase-albert-small-v2)', fontsize=16, fontweight='bold')
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        
        plt.subplot(2, 3, 2)
        plt.title(rf'SemBenchmarkLmArena', fontsize=20)
        right_similarities = self.data[1][0].load_list()[0]
        wrong_similarities = self.data[1][0].load_list()[1]
        kde_right = gaussian_kde(right_similarities, bw_method='scott')
        kde_wrong = gaussian_kde(wrong_similarities, bw_method='scott')
        x = np.linspace(0, 1, 1000)
        y_right = kde_right(x)
        y_wrong = kde_wrong(x)
        '''mean_right = np.mean(right_similarities)
        mean_wrong = np.mean(wrong_similarities)
        plt.axvline(mean_right, color='green', linestyle='--')
        plt.axvline(mean_wrong, color='red', linestyle='--')'''
        plt.plot(x, y_right, label='right hit', color='green')
        plt.plot(x, y_wrong, label='wrong hit', color='red')
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        plt.subplot(2, 3, 3)
        plt.title(rf'SemBenchmarkSearchQueries', fontsize=20)
        right_similarities = self.data[2][0].load_list()[0]
        wrong_similarities = self.data[2][0].load_list()[1]
        kde_right = gaussian_kde(right_similarities, bw_method='scott')
        kde_wrong = gaussian_kde(wrong_similarities, bw_method='scott')
        x = np.linspace(0, 1, 1000)
        y_right = kde_right(x)
        y_wrong = kde_wrong(x)
        plt.plot(x, y_right, color='green')
        plt.plot(x, y_wrong, color='red')
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        plt.subplot(2, 3, 4)
        right_similarities = self.data[0][1].load_list()[0]
        wrong_similarities = self.data[0][1].load_list()[1]
        kde_right = gaussian_kde(right_similarities, bw_method='scott')
        kde_wrong = gaussian_kde(wrong_similarities, bw_method='scott')
        x = np.linspace(min(right_similarities + wrong_similarities), max(right_similarities + wrong_similarities), 1000)
        y_right = kde_right(x)
        y_wrong = kde_wrong(x)
        '''mean_right = np.mean(right_similarities)
        mean_wrong = np.mean(wrong_similarities)
        plt.axvline(mean_right, color='green', linestyle='--')
        plt.axvline(mean_wrong, color='red', linestyle='--')'''
        plt.plot(x, y_right, label='right hit', color='green')
        plt.plot(x, y_wrong, label='wrong hit', color='red')
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.xlabel('similarity', fontsize=20)
        plt.ylabel('pdf(e5-large-v2)', fontsize=16, fontweight='bold')
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        plt.subplot(2, 3, 5)
        right_similarities = self.data[1][1].load_list()[0]
        wrong_similarities = self.data[1][1].load_list()[1]
        kde_right = gaussian_kde(right_similarities, bw_method='scott')
        kde_wrong = gaussian_kde(wrong_similarities, bw_method='scott')
        x = np.linspace(min(right_similarities + wrong_similarities), max(right_similarities + wrong_similarities), 1000)
        y_right = kde_right(x)
        y_wrong = kde_wrong(x)
        '''mean_right = np.mean(right_similarities)
        mean_wrong = np.mean(wrong_similarities)
        plt.axvline(mean_right, color='green', linestyle='--')
        plt.axvline(mean_wrong, color='red', linestyle='--')'''
        plt.plot(x, y_right, label='right hit', color='green')
        plt.plot(x, y_wrong, label='wrong hit', color='red')
        plt.xlabel('similarity', fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        plt.subplot(2, 3, 6)
        right_similarities = self.data[2][1].load_list()[0]
        wrong_similarities = self.data[2][1].load_list()[1]
        kde_right = gaussian_kde(right_similarities, bw_method='scott')
        kde_wrong = gaussian_kde(wrong_similarities, bw_method='scott')
        x = np.linspace(min(right_similarities + wrong_similarities), max(right_similarities + wrong_similarities), 1000)
        y_right = kde_right(x)
        y_wrong = kde_wrong(x)
        plt.plot(x, y_right, color='green')
        plt.plot(x, y_wrong, color='red')
        plt.xlabel('similarity', fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        fig.legend(handles, labels, loc='upper center', ncol=2, fontsize = 18, bbox_to_anchor=(0.5, 0.99))
        fig.suptitle(rf'GPT Semantic Cache 不同嵌入模型下余弦相似度高斯核密度估计', fontsize=20, y=0.05)
        plt.savefig(rf"{self.dir_path}/pictures/final/comparison_by_embedding_similarity.png")
        
class test_different_embedding:
    def __init__(self):
        self.dir_path = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
        datasets = ["SemBenchmarkClassificationSorted", "SemBenchmarkLmArena", "SemBenchmarkSearchQueries"]
        similarities = ["", "0.958"]
        embeddings = ["paraphrase-albert-small-v2", "e5-large-v2"]
        self.data = defaultdict(defaultdict)
        for i, dataset in enumerate(datasets):
            for j, embedding in enumerate(embeddings):
                self.data[i][j] = list_store(rf"{self.dir_path}/data/{dataset}_{embedding}_gpt_get_list_cache{similarities[j]}.json").load_list()
        self.add_data = defaultdict()
        for i, dataset in enumerate(datasets):
            self.add_data[i] = list_store(rf"{self.dir_path}/data/{dataset}_e5-large-v2_gpt_get_list_cache.json").load_list()
        
    def draw(self):
        fig = plt.figure(figsize=(18, 11))
        title_size = 20
        left_label_size = title_size - 2
        x_size = title_size - 4
        y_size = title_size - 4
        plt.subplot(2, 3, 1)
        plt.plot(self.data[0][0][0], self.data[0][0][2], label='paraphrase-albert-small-v2 with threshold 0.86', color='blue')
        plt.plot(self.data[0][1][0], self.data[0][1][2], label='e5-large-v2 with threshold 0.958', color='orange')
        plt.plot(self.add_data[0][0], self.add_data[0][2], label='e5-large-v2 with threshold 0.86', color='red')
        plt.ylabel('Cache Error Rate', fontsize=left_label_size, fontweight='bold')
        plt.title(rf'SemBenchmarkClassificationSorted', fontsize=title_size)
        plt.xticks(fontsize=x_size)
        plt.yticks(fontsize=y_size)

        plt.subplot(2, 3, 2)
        plt.plot(self.data[1][0][0], self.data[1][0][2], label='paraphrase-albert-small-v2 with threshold 0.86', color='blue')
        plt.plot(self.data[1][1][0], self.data[1][1][2], label='e5-large-v2 with threshold 0.958', color='orange')
        plt.plot(self.add_data[1][0], self.add_data[1][2], label='e5-large-v2 with threshold 0.86', color='red')
        plt.title(rf'SemBenchmarkLmArena', fontsize=title_size)
        plt.xticks(fontsize=x_size)
        plt.yticks(fontsize=y_size)
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(5))

        plt.subplot(2, 3, 3)
        plt.plot(self.data[2][0][0], self.data[2][0][2], label='paraphrase-albert-small-v2 with threshold 0.86', color='blue')
        plt.plot(self.data[2][1][0], self.data[2][1][2], label='e5-large-v2 with threshold 0.958', color='orange')
        plt.plot(self.add_data[2][0], self.add_data[2][2], label='e5-large-v2 with threshold 0.86', color='red')
        plt.title(rf'SemBenchmarkSearchQueries', fontsize=title_size)
        plt.xticks(fontsize=x_size)
        plt.yticks(fontsize=y_size)
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(5))

        plt.subplot(2, 3, 4)
        plt.plot(self.data[0][0][0], self.data[0][0][1], label='paraphrase-albert-small-v2 with threshold 0.86', color='blue')
        plt.plot(self.data[0][1][0], self.data[0][1][1], label='e5-large-v2 with threshold 0.958', color='orange')
        plt.plot(self.add_data[0][0], self.add_data[0][1], label='e5-large-v2 with threshold 0.86', color='red')
        plt.xlabel('Number of Queries', fontsize=left_label_size)
        plt.ylabel('Cache Hit Rate', fontsize=left_label_size, fontweight='bold')
        plt.xticks(fontsize=x_size)
        plt.yticks(fontsize=y_size)

        plt.subplot(2, 3, 5)
        plt.plot(self.data[1][0][0], self.data[1][0][1], label='paraphrase-albert-small-v2 with threshold 0.86', color='blue')
        plt.plot(self.data[1][1][0], self.data[1][1][1], label='e5-large-v2 with threshold 0.958', color='orange')
        plt.plot(self.add_data[1][0], self.add_data[1][1], label='e5-large-v2 with threshold 0.86', color='red')
        plt.xlabel('Number of Queries', fontsize=left_label_size)
        plt.xticks(fontsize=x_size)
        plt.yticks(fontsize=y_size)
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(5))

        plt.subplot(2, 3, 6)
        plt.plot(self.data[2][0][0], self.data[2][0][1], label='paraphrase-albert-small-v2 with threshold 0.86', color='blue')
        plt.plot(self.data[2][1][0], self.data[2][1][1], label='e5-large-v2 with threshold 0.958', color='orange')
        plt.plot(self.add_data[2][0], self.add_data[2][1], label='e5-large-v2 with threshold 0.86', color='red')
        plt.xlabel('Number of Queries', fontsize=left_label_size)
        plt.xticks(fontsize=x_size)
        plt.yticks(fontsize=y_size)
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(5))
        handles, labels = plt.gca().get_legend_handles_labels()

        fig.legend(handles, labels, loc='upper center', ncol=3, fontsize = 18, bbox_to_anchor=(0.5, 0.99))
        fig.suptitle(rf'GPT Semantic Cache 线性映射阈值效果对比图', fontsize=22, y=0.05)
        plt.savefig(rf"{self.dir_path}/pictures/final/comparison_by_different_embedding_with_adjusted_thresholds.png")

class test_different_vcache:
    def __init__(self):
        self.dir_path = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
        datasets = ["SemBenchmarkClassificationSorted", "SemBenchmarkLmArena", "SemBenchmarkSearchQueries"]
        embeddings = ["paraphrase-albert-small-v2", "e5-large-v2"]
        self.data = defaultdict(defaultdict)
        for i, dataset in enumerate(datasets):
            for j, embedding in enumerate(embeddings):
                self.data[i][j] = list_store(rf"{self.dir_path}/data/{dataset}_{embedding}_SimpleVCache_delta = 0.1_vcache_get_list_cache.json").load_list()
        self.add_data = defaultdict()
        for i, dataset in enumerate(datasets):
            self.add_data[i] = list_store(rf"{self.dir_path}/data/{dataset}_e5-large-v2_SimpleVCache_delta = 0.03_vcache_get_list_cache.json").load_list()

    def draw(self):
        fig = plt.figure(figsize=(18, 8))
        title_size = 20
        left_label_size = title_size - 2
        x_size = title_size - 4
        y_size = title_size - 4
        plt.subplot(2, 3, 1)
        plt.plot(self.data[0][0][0], self.data[0][0][2], label='vCache Delta=0.1 at paraphrase-albert-small-v2', color='blue')
        plt.plot(self.data[0][1][0], self.data[0][1][2], label='vCache Delta=0.1 at e5-large-v2', color='orange')
        plt.plot(self.add_data[0][0], self.add_data[0][2], label='vCache Delta=0.03 at e5-large-v2', color='red')
        plt.ylabel('Cache Error Rate', fontsize=left_label_size, fontweight='bold')
        plt.title(rf'SemBenchmarkClassificationSorted', fontsize=title_size)
        plt.xticks(fontsize=x_size)
        plt.yticks(fontsize=y_size)

        plt.subplot(2, 3, 2)
        plt.plot(self.data[1][0][0], self.data[1][0][2], label='vCache Delta=0.1 at paraphrase-albert-small-v2', color='blue')
        plt.plot(self.data[1][1][0], self.data[1][1][2], label='vCache Delta=0.1 at e5-large-v2', color='orange')
        plt.plot(self.add_data[1][0], self.add_data[1][2], label='vCache Delta=0.03 at e5-large-v2', color='red')
        plt.title(rf'SemBenchmarkLmArena', fontsize=title_size)
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(5))
        plt.xticks(fontsize=x_size)
        plt.yticks(fontsize=y_size)

        plt.subplot(2, 3, 3)
        plt.plot(self.data[2][0][0], self.data[2][0][2], label='vCache Delta=0.1 at paraphrase-albert-small-v2', color='blue')
        plt.plot(self.data[2][1][0], self.data[2][1][2], label='vCache Delta=0.1 at e5-large-v2', color='orange')
        plt.plot(self.add_data[2][0], self.add_data[2][2], label='vCache Delta=0.03 at e5-large-v2', color='red')
        plt.title(rf'SemBenchmarkSearchQueries', fontsize=title_size)
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(5))
        plt.xticks(fontsize=x_size)
        plt.yticks(fontsize=y_size)

        plt.subplot(2, 3, 4)
        plt.plot(self.data[0][0][0], self.data[0][0][1], label='vCache Delta=0.1 at paraphrase-albert-small-v2', color='blue')
        plt.plot(self.data[0][1][0], self.data[0][1][1], label='vCache Delta=0.1 at e5-large-v2', color='orange')
        plt.plot(self.add_data[0][0], self.add_data[0][1], label='vCache Delta=0.03 at e5-large-v2', color='red')   
        plt.xlabel('Number of Queries', fontsize=left_label_size)
        plt.ylabel('Cache Hit Rate', fontsize=left_label_size, fontweight='bold')
        plt.xticks(fontsize=x_size)
        plt.yticks(fontsize=y_size)
        
        plt.subplot(2, 3, 5)
        plt.plot(self.data[1][0][0], self.data[1][0][1], label='vCache Delta=0.1 at paraphrase-albert-small-v2', color='blue')
        plt.plot(self.data[1][1][0], self.data[1][1][1], label='vCache Delta=0.1 at e5-large-v2', color='orange')
        plt.plot(self.add_data[1][0], self.add_data[1][1], label='vCache Delta=0.03 at e5-large-v2', color='red')
        plt.xlabel('Number of Queries', fontsize=left_label_size)
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(5))
        plt.xticks(fontsize=x_size)
        plt.yticks(fontsize=y_size)

        plt.subplot(2, 3, 6)
        plt.plot(self.data[2][0][0], self.data[2][0][1], label='vCache Delta=0.1 at paraphrase-albert-small-v2', color='blue')
        plt.plot(self.data[2][1][0], self.data[2][1][1], label='vCache Delta=0.1 at e5-large-v2', color='orange')
        plt.plot(self.add_data[2][0], self.add_data[2][1], label='vCache Delta=0.03 at e5-large-v2', color='red')
        plt.xlabel('Number of Queries', fontsize=left_label_size)
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(5))
        plt.xticks(fontsize=x_size)
        plt.yticks(fontsize=y_size)

        handles, labels = plt.gca().get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=3, fontsize = 18, bbox_to_anchor=(0.5, 1.01))
        plt.savefig(rf"{self.dir_path}/pictures/final/comparison_by_different_vcache_with_adjusted_thresholds.png")


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
            test.compare_all()
    '''
    test = compare_by_embedding_all("paraphrase-albert-small-v2")
    test = compare_by_embedding_all("e5-large-v2")
    
    