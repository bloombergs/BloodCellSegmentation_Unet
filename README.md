# BloodCellSegmentation_Unet
I implements a U-Net architecture for the segmentation of blood cells from microscopy images using a dataset of blood cell images with their corresponding masks.The model is trained using a dataset of cell images and their binary segmentation masks, where the goal is to create new segmentation mask from unseen image.

Dataset = Blood Cell Segmentation Dataset (https://www.kaggle.com/datasets/jeetblahiri/bccd-dataset-with-mask)

# Preprocessing
Dataset: The model is trained using the Blood Cell Segmentation Dataset with Masks. You will need two directories: one containing original microscopy images and the other containing the corresponding segmentation masks.
Image Preprocessing: Images are resized to 512x512 and converted into tensors for input to the model.

# Model
The model uses the U-Net architecture, which consists of the following blocks:
Encoder: A series of convolutional blocks (convdown), each followed by max pooling to downsample the image and extract features.
Bottleneck: A convolutional block that processes the deepest feature representation.
Decoder: A series of transposed convolutional blocks (convup) to upsample the feature map and predict the segmentation mask.
Output: A final convolutional layer that outputs the segmentation mask with a single channel.

# Training
The model is trained for 2 epochs using AdamW optimizer and BCEWithLogitsLoss as the loss function. The model's parameters are updated to minimize the binary cross-entropy loss between the predicted mask and the ground truth mask.

# Inference and Result
After training, the model can be used for inference on unseen images. The image is passed through the trained model, and the resulting segmentation mask is processed and visualized using matplotlib.
![Screenshot 2024-12-14 193641](https://github.com/user-attachments/assets/70e38692-f719-4767-80a3-568340a3f096)

# Optional
After i have the Generated mask i decided that it'll be cool to use the mask to make the original image have a Red(Blood like) color.
in bloodred.py we can paste the image and mask to generate the Red Textured Image.
![original_image](https://github.com/user-attachments/assets/da2f5f1f-e10b-4baa-966b-d0ba18af1315)
![predicted_mask](https://github.com/user-attachments/assets/3fed7d33-df6e-4b68-8914-d5f28fb67167)
![modified_blood_red](https://github.com/user-attachments/assets/7d61746a-0227-40cf-9788-363ac82aa97c)

