import matplotlib.pyplot as plt

def plot_hit_rate(sample_counts: list, hit_rate: list, save_path: str):
    plt.figure(figsize=(10, 6))
    plt.plot(sample_counts, hit_rate, label='Cache Hit Rate', color='blue')
    plt.xlabel('Number of Samples')
    plt.ylabel('Cache Hit Rate')
    plt.title('Cache Hit Rate vs Number of Samples')
    plt.legend()
    plt.grid()
    plt.savefig(save_path)