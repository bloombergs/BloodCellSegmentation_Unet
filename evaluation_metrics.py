import cv2
import numpy as np
from scipy.spatial.distance import cdist

ground_truth_path = "YourGroundtruthPath"
generated_mask_path = "YourGeneratedPath"

ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
generated_mask = cv2.imread(generated_mask_path, cv2.IMREAD_GRAYSCALE)

ground_truth = np.where(ground_truth > 127, 1, 0)
generated_mask = np.where(generated_mask > 127, 1, 0)

intersection = np.sum(np.logical_and(ground_truth, generated_mask))
union = np.sum(np.logical_or(ground_truth, generated_mask))

contours_gt, _ = cv2.findContours(ground_truth.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_gen, _ = cv2.findContours(generated_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
gt_points = np.vstack([contour.reshape(-1, 2) for contour in contours_gt])
gen_points = np.vstack([contour.reshape(-1, 2) for contour in contours_gen])
distances = cdist(gt_points, gen_points, 'cityblock')
hd = max(np.min(distances, axis=0).max(), np.min(distances, axis=1).max())

image_height, image_width = ground_truth.shape
hd1 = hd/((image_height - 1)+(image_width - 1))
hd1 = 1 - hd1
print(f"Normalized Hausdorff: {hd1}")

iou = intersection / union
print(f"IOU: {iou}")

dice = (2 * intersection) / (np.sum(ground_truth) + np.sum(generated_mask))
print(f"Dice Similarity Coefficient: {dice}")

F8_Score = np.sqrt(hd1 * iou)
print(f"F8 Score = {F8_Score}")
