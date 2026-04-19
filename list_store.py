import json

class list_store:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def save_list(self, data_list: list):
        with open(self.file_path, 'w') as f:
            json.dump(data_list, f)

    def load_list(self) -> list:
        try:
            with open(self.file_path, 'r') as f:
                data_list = json.load(f)
            return data_list
        except FileNotFoundError:
            return []
        
if __name__ == '__main__':
    # 简单测试
    import os
    store = list_store("test_list.json")
    my_list = [1, 2, 3, 4, 5]
    store.save_list(my_list)
    loaded_list = store.load_list()
    print(f"Loaded List: {loaded_list}")
    os.remove("test_list.json")  # 清理测试文件