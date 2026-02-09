import cv2
import matplotlib.pyplot as plt
import numpy as np
from view_surface import view_surface

# Lab 01 - Image Processing Dashboard

# Load images
image = cv2.imread("flower.png")
image_c = cv2.imread("red_bird.png", cv2.IMREAD_COLOR)

# Print info
print(f"Flower Image - Type: {type(image)}, Dtype: {image.dtype}, Shape: {image.shape}")
print(f"First pixel (BGR): {image[0, 0]}")
print(f"Red Bird Image - Shape: {image_c.shape}")
print(f"First pixel (BGR): {image_c[0, 0]}")

# Convert BGR to RGB for matplotlib display
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_c_rgb = cv2.cvtColor(image_c, cv2.COLOR_BGR2RGB)

# Resize image
resized_image = cv2.resize(image_c, (200, 150))
resized_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
print(f"\nOriginal shape: {image_c.shape}")
print(f"Resized shape: {resized_image.shape}")

# Split into color channels
b, g, r = cv2.split(image_c)

# Merge back
merged_image = cv2.merge([b, g, r])
merged_rgb = cv2.cvtColor(merged_image, cv2.COLOR_BGR2RGB)

# Pure color channels
zeros = np.zeros(image_c.shape[0:2], dtype='uint8')
pure_blue = cv2.cvtColor(cv2.merge([b, zeros, zeros]), cv2.COLOR_BGR2RGB)
pure_green = cv2.cvtColor(cv2.merge([zeros, g, zeros]), cv2.COLOR_BGR2RGB)
pure_red = cv2.cvtColor(cv2.merge([zeros, zeros, r]), cv2.COLOR_BGR2RGB)

# Grayscale
gray_image = cv2.cvtColor(image_c, cv2.COLOR_BGR2GRAY)
print(f"\nOriginal image shape: {image_c.shape}")
print(f"Grayscale image shape: {gray_image.shape}")

# Save a copy
cv2.imwrite("flower_copy.png", image)

# Create Dashboard - All visualizations in one figure

fig = plt.figure(figsize=(16, 14))
fig.suptitle("Lab 01 - Image Processing Dashboard", fontsize=16, fontweight='bold')

# Row 1: Original images
ax1 = fig.add_subplot(4, 4, 1)
ax1.imshow(image_rgb)
ax1.set_title('Flower (Original)')
ax1.axis('off')

ax2 = fig.add_subplot(4, 4, 2)
ax2.imshow(image_c_rgb)
ax2.set_title('Red Bird (Original)')
ax2.axis('off')

ax3 = fig.add_subplot(4, 4, 3)
ax3.imshow(resized_rgb)
ax3.set_title('Resized (200x150)')
ax3.axis('off')

ax4 = fig.add_subplot(4, 4, 4)
ax4.imshow(gray_image, cmap='gray')
ax4.set_title('Grayscale')
ax4.axis('off')

# Row 2: Split channels (grayscale view)
ax5 = fig.add_subplot(4, 4, 5)
ax5.imshow(b, cmap='gray')
ax5.set_title('Blue Channel')
ax5.axis('off')

ax6 = fig.add_subplot(4, 4, 6)
ax6.imshow(g, cmap='gray')
ax6.set_title('Green Channel')
ax6.axis('off')

ax7 = fig.add_subplot(4, 4, 7)
ax7.imshow(r, cmap='gray')
ax7.set_title('Red Channel')
ax7.axis('off')

ax8 = fig.add_subplot(4, 4, 8)
ax8.imshow(merged_rgb)
ax8.set_title('Merged (Reconstructed)')
ax8.axis('off')

# Row 3: Pure color channels
ax9 = fig.add_subplot(4, 4, 9)
ax9.imshow(pure_blue)
ax9.set_title('Pure Blue')
ax9.axis('off')

ax10 = fig.add_subplot(4, 4, 10)
ax10.imshow(pure_green)
ax10.set_title('Pure Green')
ax10.axis('off')

ax11 = fig.add_subplot(4, 4, 11)
ax11.imshow(pure_red)
ax11.set_title('Pure Red')
ax11.axis('off')

# Row 3-4: Histograms (spanning 2 columns each)
ax12 = fig.add_subplot(4, 4, 12)
hist_gray = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
ax12.plot(hist_gray, color='black')
ax12.set_title('Grayscale Histogram')
ax12.set_xlabel('Intensity')
ax12.set_ylabel('Frequency')
ax12.set_xlim([0, 256])

# Row 4: Color histogram (larger, spanning multiple cells)
ax13 = fig.add_subplot(4, 2, 7)
colors = ('b', 'g', 'r')
for i, col in enumerate(colors):
    hist = cv2.calcHist([image_c], [i], None, [256], [0, 256])
    ax13.plot(hist, color=col, label=f'{col.upper()} channel')
ax13.set_title('Color Histogram (BGR channels)')
ax13.set_xlabel('Pixel Intensity (0-255)')
ax13.set_ylabel('Frequency')
ax13.set_xlim([0, 256])
ax13.legend()

plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.show()

# 3D Surface Visualization (separate window)
print("\n--- 3D Surface Visualization ---")
view_surface("red_bird.png")