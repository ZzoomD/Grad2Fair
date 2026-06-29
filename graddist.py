"""
    GradDist class
"""
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import numpy as np


class GradDist:
    def __init__(self, bw_method=0.15, x_grid=1000):
        super(GradDist, self).__init__()
        self.bw_method = bw_method
        self.x_grid = x_grid
    
    def compute(self, data):
        TOP_N = 2
        kde = gaussian_kde(data)
        kde.set_bandwidth(bw_method=self.bw_method)
        x_grid = np.linspace(min(data), max(data), self.x_grid)
        y_kde = kde(x_grid)
        peaks_indices, _ = find_peaks(y_kde)
        peak_heights = y_kde[peaks_indices]
        sorted_indices = np.argsort(peak_heights)[::-1]
        top_n_indices = sorted_indices[:TOP_N]
        final_indices = np.sort(peaks_indices[top_n_indices]) # 按x轴排序
        final_peak_values = x_grid[final_indices]
        
        if len(final_peak_values) == 2:
            raw_distance = final_peak_values[1] - final_peak_values[0]
            global_range = np.max(data) - np.min(data)
            normalized_distance = raw_distance / global_range
            
            return normalized_distance
        else:
            print(f"Not enough peaks! (only found {len(final_peak_values)})")
            return 0


if __name__ == '__main__':
    np.random.seed(42) # Fix the random seed for reproducibility
    data1 = np.random.normal(loc=10, scale=2, size=1000)
    data2 = np.random.normal(loc=20, scale=1.5, size=500)
    data3 = np.random.normal(loc=5, scale=0.5, size=50) 
    data = np.concatenate([data1, data2, data3])

    grad_dist = GradDist()
    normalized_distance = grad_dist.compute(data)
    print(f"Normalized Distance: {normalized_distance}")