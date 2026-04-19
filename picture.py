import matplotlib.pyplot as plt

class picture_generator:
    def __init__(self, sample_counts: list, hit_rate: list, error_rate: list):
        self.sample_counts = sample_counts
        self.hit_rate = hit_rate
        self.error_rate = error_rate

    def plot_hit_rate(self, save_path: str):
        plt.figure(figsize=(10, 6))
        plt.plot(self.sample_counts, self.hit_rate, label='Cache Hit Rate', color='blue')
        plt.xlabel('Number of Samples')
        plt.ylabel('Cache Hit Rate')
        plt.title('Cache Hit Rate vs Number of Samples')
        plt.ylim(0, max(self.hit_rate))
        plt.xlim(0, max(self.sample_counts))
        plt.legend()
        plt.grid()
        plt.savefig(save_path)

    def plot_error_rate(self, save_path: str):
        plt.figure(figsize=(10, 6))
        plt.plot(self.sample_counts, self.error_rate, label='Error Rate', color='red')
        plt.xlabel('Number of Samples')
        plt.ylabel('Error Rate')
        plt.title('Error Rate vs Number of Samples')
        plt.ylim(0, max(self.error_rate))
        plt.xlim(0, max(self.sample_counts))
        plt.legend()
        plt.grid()
        plt.savefig(save_path)

if __name__ == '__main__':
    import os
    sample_counts = [100, 500, 1000, 5000, 10000]
    hit_rate = [0.1, 0.3, 0.5, 0.7, 0.9]
    error_rate = [0.9, 0.7, 0.5, 0.3, 0.1]
    path = os.path.join(os.path.dirname(__file__), "pictures")
    os.makedirs(path, exist_ok=True)
    pic_gen = picture_generator(sample_counts, hit_rate, error_rate)
    pic_gen.plot_error_rate(os.path.join(path, "cache_error_rate1.png"))
    pic_gen.plot_hit_rate(os.path.join(path, "cache_hit_rate1.png"))
    print("Cache plots saved.")