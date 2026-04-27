from picture import picture_generator
from sigmod_cache import sigmod_cache
from embedding import embedding_generator
from list_store import list_store
from pre_process import pre_processor
from vector_database import milvus_db
from my_sqllite3 import LMarenaSQLiteManager, ClassificationSortedSQLiteManager, SearchQueriesSQLiteManager, vcache_hit_record_SQLiteManager
from sigmod_probality import sigmod_probality
from vcache_final import SimpleVCache
from sigmod_iid import sigmod_iid
import os

class cache_test:
    def __init__(self, dataset : str, embedding_model : str):
        self.dataset = dataset
        self.embedding_mod = embedding_model
        self.processor = pre_processor(dataset)
        self.embedding_generator = embedding_generator(embedding_model)
        self.dir_path = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
        self.milvus_db = milvus_db(rf"{self.dir_path}/milvus_cache.db")
        self.milvus_db.create_collection(dataset + "_collection", self.embedding_generator.dimension)
        self.cache_hit = 0
        self.right_hit = 0
        self.miss = 0
        self.hit_rate = []
        self.sample_counts = []
        self.error_rate = []
        print("开始预处理和嵌入数据集...", flush=True)
        ds = self.processor.pre_process_vector()
        self.ds = self.embedding_generator.embed_ds(ds, rf"{self.dir_path}/data/{self.dataset}_{embedding_model}_embedding")
        print("数据集预处理和嵌入完成。", flush=True)
        if dataset == "SemBenchmarkClassificationSorted":
            self.sqllite_db = ClassificationSortedSQLiteManager(rf"{self.dir_path}/sqlite_cache.db")
            self.key_name = ["response_llama_3_8b"]
        elif dataset == "SemBenchmarkLmArena":
            self.sqllite_db = LMarenaSQLiteManager(rf"{self.dir_path}/sqlite_cache.db")
            self.key_name = ["ID_Set"]
        elif dataset == "SemBenchmarkSearchQueries":
            self.sqllite_db = SearchQueriesSQLiteManager(rf"{self.dir_path}/sqlite_cache.db")
            self.key_name = ["id_set", "cluster_id"]
        else:
            raise ValueError("不支持的dataset")
        print("处理后的列名：", self.ds["train"].column_names)

    def test_self(self):
        pass
        
class gpt_cache_test(cache_test):
    def __init__(self, dataset, embedding_model, must_run = False):
        super().__init__(dataset, embedding_model)
        self.list_store = list_store(rf"{self.dir_path}/data/{self.dataset}_{self.embedding_mod}_gpt_get_list_cache.json")
        self.must_run = must_run

    def test_self(self):
        if os.path.exists(rf"{self.dir_path}/data/{self.dataset}_{self.embedding_mod}_gpt_get_list_cache.json") and not self.must_run:
            print("Loading results from cache...")
            load_result = self.list_store.load_list()
            self.sample_counts = load_result[0]
            self.hit_rate = load_result[1]
            self.error_rate = load_result[2]
        else:
            for i in range(len(self.ds["train"])):
                query_embedding = self.ds["train"][i]["embedding"]
                similar_ids = self.milvus_db.single_search_collection(self.dataset + "_collection", query_embedding, threshold=0.86)
                if similar_ids:
                    self.cache_hit += 1
                    if self.sqllite_db.search_by_id(similar_ids[0][0]) == [self.ds["train"][i][key] for key in self.key_name]:
                        self.right_hit += 1
                else:
                    self.miss += 1
                    self.milvus_db.insert_into_collection(self.dataset + "_collection", [query_embedding], [i])
                    self.sqllite_db.insert(i, [self.ds["train"][i][key] for key in self.key_name])
                self.hit_rate.append(self.cache_hit / (self.cache_hit + self.miss))
                self.error_rate.append((self.cache_hit - self.right_hit) / (self.cache_hit + self.miss))
                self.sample_counts.append(i + 1)
            self.list_store.save_list([self.sample_counts, self.hit_rate, self.error_rate])
        self.sqllite_db.close()
        self.milvus_db.close()
        self.pic_gen = picture_generator(self.sample_counts, self.hit_rate, self.error_rate)
        os.makedirs(rf"{self.dir_path}/pictures", exist_ok=True)
        self.pic_gen.plot_hit_rate(rf"{self.dir_path}/pictures/{self.dataset}_{self.embedding_mod}_gpt_hit_rate.png")
        self.pic_gen.plot_error_rate(rf"{self.dir_path}/pictures/{self.dataset}_{self.embedding_mod}_gpt__error_rate.png")
        print("Cache plots saved.")

class vcache_base(cache_test):
    def __init__(self, dataset, embedding_model, cache, must_run=False):
        super().__init__(dataset, embedding_model)
        self.cache = cache
        self.list_store = list_store(rf"{self.dir_path}/data/{self.dataset}_{self.embedding_mod}_{self.cache.__class__.__name__}_delta = {self.cache.delta}_vcache_get_list_cache.json")
        self.must_run = must_run

    def test_self(self):
        if os.path.exists(rf"{self.dir_path}/data/{self.dataset}_{self.embedding_mod}_{self.cache.__class__.__name__}_delta = {self.cache.delta}_vcache_get_list_cache.json") and not self.must_run:
            print("Loading results from cache...")
            load_result = self.list_store.load_list()
            self.sample_counts = load_result[0]
            self.hit_rate = load_result[1]
            self.error_rate = load_result[2]
        else:
            self.hit_db = vcache_hit_record_SQLiteManager(rf"{self.dir_path}/sqlite_cache.db")
            print("开始 VCache 测试", flush=True)
            for i in range(len(self.ds["train"])):
                query_embedding = self.ds["train"][i]["embedding"]
                similar_ids = self.milvus_db.single_search_collection(self.dataset + "_collection", query_embedding, threshold=-1)
                if similar_ids:
                    s_vals, c_vals = self.hit_db.search_by_id(similar_ids[0][0])
                    if self.cache.decide(similar_ids[0][1], s_vals, c_vals) == 'exploit':
                        self.cache_hit += 1
                        if self.sqllite_db.search_by_id(similar_ids[0][0]) == [self.ds["train"][i][key] for key in self.key_name]:
                            self.right_hit += 1
                    else:
                        self.miss += 1
                        s_vals.append(similar_ids[0][1])
                        if self.sqllite_db.search_by_id(similar_ids[0][0]) == [self.ds["train"][i][key] for key in self.key_name]:
                            c_vals.append(1)
                        else:
                            c_vals.append(0)
                            self.hit_db.add_or_update(i, [-1, 1], [0, 1])
                            self.milvus_db.insert_into_collection(self.dataset + "_collection", [query_embedding], [i])
                            self.sqllite_db.insert(i, [self.ds["train"][i][key] for key in self.key_name])
                        self.hit_db.add_or_update(similar_ids[0][0], s_vals, c_vals)
                else:
                    self.miss += 1
                    self.milvus_db.insert_into_collection(self.dataset + "_collection", [query_embedding], [i])
                    self.sqllite_db.insert(i, [self.ds["train"][i][key] for key in self.key_name])
                    self.hit_db.add_or_update(i, [-1, 1], [0, 1])
                self.hit_rate.append(self.cache_hit / (self.cache_hit + self.miss))
                self.error_rate.append((self.cache_hit - self.right_hit) / (self.cache_hit + self.miss))
                self.sample_counts.append(i + 1)
            self.list_store.save_list([self.sample_counts, self.hit_rate, self.error_rate])
        self.sqllite_db.close()
        self.hit_db.close()
        self.milvus_db.close()
        self.pic_gen = picture_generator(self.sample_counts, self.hit_rate, self.error_rate)
        os.makedirs(rf"{self.dir_path}/pictures", exist_ok=True)
        self.pic_gen.plot_hit_rate(rf"{self.dir_path}/pictures/{self.dataset}_{self.embedding_mod}_{self.cache.__class__.__name__}_delta = {self.cache.delta}_hit_rate.png")
        self.pic_gen.plot_error_rate(rf"{self.dir_path}/pictures/{self.dataset}_{self.embedding_mod}_{self.cache.__class__.__name__}_delta = {self.cache.delta}_error_rate.png")


class gpt_kde_test(cache_test):
    def test_self(self):
        self.hit_db = vcache_hit_record_SQLiteManager(rf"{self.dir_path}/sqlite_cache.db")
        for i in range(len(self.ds["train"])):
            query_embedding = self.ds["train"][i]["embedding"]
            similar_ids = self.milvus_db.single_search_collection(self.dataset + "_collection", query_embedding, threshold=-1) #这里不设置阈值，找到最相近的即可
            if similar_ids: # 只考虑命中的情况
                if self.sqllite_db.search_by_id(similar_ids[0][0]) == [self.ds["train"][i][key] for key in self.key_name]: # 如果命中正确
                    s_vals, c_vals = self.hit_db.search_by_id(similar_ids[0][0])
                    s_vals.append(similar_ids[0][1])
                    c_vals.append(1)
                    self.hit_db.add_or_update(similar_ids[0][0], s_vals, c_vals)
                else: # 如果命中错误
                    s_vals, c_vals = self.hit_db.search_by_id(similar_ids[0][0])
                    s_vals.append(similar_ids[0][1])
                    c_vals.append(0)
                    self.hit_db.add_or_update(similar_ids[0][0], s_vals, c_vals)
                    self.milvus_db.insert_into_collection(self.dataset + "_collection", [query_embedding], [i])
                    self.sqllite_db.insert(i, [self.ds["train"][i][key] for key in self.key_name])
                    self.hit_db.add_or_update(i, [], [])
            else: # 如果整个向量库中没有其他向量
                self.milvus_db.insert_into_collection(self.dataset + "_collection", [query_embedding], [i])
                self.sqllite_db.insert(i, [self.ds["train"][i][key] for key in self.key_name])
                self.hit_db.add_or_update(i, [], [])
        print("分布计算完成")
        records = self.hit_db.get_all_records()
        right_similarities = []
        wrong_similarities = []
        for record in records:
            _, s_vals, c_vals = record
            for s, c in zip(s_vals, c_vals):
                if c == 1:
                    right_similarities.append(s)
                else:
                    wrong_similarities.append(s)
        # 接下来使用核函数展示分布
        print("开始绘制分布图...")
        self.pic_gen = picture_generator([], [], [])
        self.pic_gen.kde_gen(right_similarities, wrong_similarities, rf"{self.dir_path}/pictures/{self.dataset}_{self.embedding_mod}_gpt_kde_similarities.png")
        self.hit_db.close()
        self.sqllite_db.close()
        self.milvus_db.close()

if __name__ == "__main__":
    dataset = "SemBenchmarkClassificationSorted"
    # SemBenchmarkClassificationSorted
    # SemBenchmarkLmArena
    # SemBenchmarkSearchQueries
    embedding_model = "paraphrase-albert-small-v2"
    # 'paraphrase-albert-small-v2' check
    # 'gte-large-en-v1.5',暂时不知道为什么不能使用
    # 'e5-large-v2' check 
    test = gpt_kde_test(dataset, embedding_model)
    test.test_self()
    
