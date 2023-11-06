import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import os
from pathlib import Path

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def get_filename_without_extension(path_str):
    filename = os.path.basename(path_str)
    filename_without_extension = Path(filename).stem
    return filename_without_extension

def det_lines(image):
    cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply edge detection
    edges = cv2.Canny(image, 50, 150, apertureSize=3)

    # Apply Hough Line Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 30)

    # Iterate over the lines and draw them on the original image
    if lines is not None:
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)


img_file_name = "./notebooks/images/map_building1.png"
img_file_name_without_extension = get_filename_without_extension(img_file_name)
image = cv2.imread(img_file_name)
# plt.imshow(image)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# plt.imshow(image)
sam_checkpoint = "./models/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)

masks = mask_generator.generate(image)

# predictor = SamPredictor(sam)
# predictor.set_image(image)
# only surpport point or box format
# masks, _, _ = predictor.predict("I want buildings")

# Step 1: Check if the directory exists, if not, create it
if not os.path.exists("./seg_images"):
    os.makedirs("./seg_images")
else:
    # Delete all files in the directory
    for file in os.listdir("./seg_images"):
        os.remove(os.path.join("./seg_images", file))

# test how 2d index works in 3d shape array
# index_2d = np.array([[1,1],[0,0]])
# index_3d = np.zeros((2,2,3))
# index_3d[index_2d] = 1
# print(index_3d)

# Step 2: Loop over each annotation
for i, ann in enumerate(masks):
    # Step 2.1: Get the segmentation and convert it to a mask, the elements in m are True or False
    m = ann['segmentation']
    effective_area = np.count_nonzero(m) / (image.shape[0] * image.shape[1])
    if effective_area < 0.01:
        continue
    
    mask = np.zeros_like(image)
    mask[m] = 255

    # Step 2.2: Apply the mask to the image
    segmented_image = cv2.bitwise_and(image, mask)

    # Step 2.3: Save the segmented image
    cv2.imwrite(f"./seg_images/segmented_image_{img_file_name_without_extension}_{i}.png", segmented_image)
    
plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show()