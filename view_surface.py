import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2


def view_surface(path, size=(100, 100), cmap='gray', elev=60, azim=30):
    
    # Load image in grayscale
    img = cv2.imread(path, 0)
    
    if img is None:
        print(f"Error: Could not load image at '{path}'")
        return
    
    # Downscaling has a "smoothing" effect
    img = cv2.resize(img, size) / 255.0  # Normalize to 0-1
    
    # Create the x and y coordinate arrays (pixel indices)
    xx, yy = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    
    # Create the figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(projection='3d')
    
    # Plot the surface
    surf = ax.plot_surface(xx, yy, img, rstride=1, cstride=1, 
                           cmap=cmap, linewidth=0, antialiased=True)
    
    # Set viewing angle
    ax.view_init(elev=elev, azim=azim)
    
    # Labels and title
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_zlabel('Intensity')
    ax.set_title(f'3D Surface: {path}')
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Pixel Intensity')
    
    plt.tight_layout()
    plt.show()

