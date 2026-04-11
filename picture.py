import matplotlib.pyplot as plt

def plot_hit_rate(sample_counts: list, hit_rate: list, save_path: str):
    plt.figure(figsize=(10, 6))
    plt.plot(sample_counts, hit_rate, label='Cache Hit Rate', color='blue')
    plt.xlabel('Number of Samples')
    plt.ylabel('Cache Hit Rate')
    plt.title('Cache Hit Rate vs Number of Samples')
    plt.ylim(0, 1)
    plt.xlim(0, max(sample_counts))
    plt.legend()
    plt.grid()
    plt.savefig(save_path)

def plot_error_rate(sample_counts: list, error_rate: list, save_path: str):
    plt.figure(figsize=(10, 6))
    plt.plot(sample_counts, error_rate, label='Error Rate', color='red')
    plt.xlabel('Number of Samples')
    plt.ylabel('Error Rate')
    plt.title('Error Rate vs Number of Samples')
    plt.ylim(0, max(error_rate))
    plt.xlim(0, max(sample_counts))
    plt.legend()
    plt.grid()
    plt.savefig(save_path)

if __name__ == '__main__':
    import os
    sample_counts = [100, 500, 1000, 5000, 10000]
    error_rate = [0.9, 0.7, 0.5, 0.3, 0.1]
    path = os.path.join(os.path.dirname(__file__), "pictures")
    os.makedirs(path, exist_ok=True)
    plot_error_rate(sample_counts, error_rate, os.path.join(path, "cache_error_rate1.png"))
    print("Cache error rate plot saved.")