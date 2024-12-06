import matplotlib.pyplot as plt
import numpy as np

def plot_sample_images(X, y, class_names, num_samples=5):
    plt.figure(figsize=(15, 5))
    for i in range(num_samples):
        idx = np.random.randint(0, len(X))
        image = X[idx]
        label = class_names[y[idx]]
        plt.subplot(1, num_samples, i+1)
        plt.imshow(image)
        plt.title(f'Class: {label}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
