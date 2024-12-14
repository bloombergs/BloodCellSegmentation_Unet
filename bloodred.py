import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

image_path = '/kaggle/input/maskandori/original_image.jpg'
mask_path = '/kaggle/input/maskandori/predicted_mask.png'

image = Image.open(image_path).convert("RGB")
mask = Image.open(mask_path).convert("L")

mask = mask.resize(image.size)

image_array = np.array(image)
mask_array = np.array(mask)

blood_red = [160, 20, 20]


for i in range(image_array.shape[0]):
    for j in range(image_array.shape[1]):
        if mask_array[i, j] == 255:
            original_pixel = image_array[i, j]
            
            blended_pixel = [
                int(original_pixel[0] * 0.7 + blood_red[0] * 0.3),
                int(original_pixel[1] * 0.7), 
                int(original_pixel[2] * 0.7 + blood_red[2] * 0.3) 
            ]
            image_array[i, j] = blended_pixel

modified_image = Image.fromarray(image_array)

modified_image.save("modified_blood_red_with_texture.jpg")

plt.imshow(modified_image)
plt.axis('off')
plt.show()
