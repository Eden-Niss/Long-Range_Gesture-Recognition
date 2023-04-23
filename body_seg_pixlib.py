from torchvision.io.image import read_image
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_segmentation_masks
from torchvision import transforms
import torchvision.transforms.functional as F
import PIL
import cv2
import numpy as np

def CropingMask(original_image, mask_image):
    imagergb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    color_mask = np.zeros_like(imagergb)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilate = cv2.dilate(mask_image, kernel, iterations=5)
    d = np.array(dilate, dtype=bool)
    color_mask[:, :, 0] += (d * 255).astype('uint8')
    color_mask[:, :, 1] += (d * 255).astype('uint8')
    color_mask[:, :, 2] += (d * 255).astype('uint8')
    res = ((imagergb / color_mask) * color_mask).astype('uint8')
    res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    positions = np.nonzero(res)
    top = positions[0].min()
    bottom = positions[0].max()
    left = positions[1].min()
    right = positions[1].max()
    # output = cv2.rectangle(cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
    #                        , (left, top), (right, bottom), (0, 255, 0), 1)
    cropped_image = res[top:bottom, left:right]
    return cropped_image

image = cv2.imread("4.png")
img = read_image("4.png")
# Step 1: Initialize model with the best available weights
weights = FCN_ResNet50_Weights.DEFAULT
sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}
model = fcn_resnet50(weights=weights)
model.eval()

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = preprocess(img).unsqueeze(0)

# Step 4: Use the model and visualize the prediction
prediction = model(batch)["out"]
normalized_masks = prediction.softmax(dim=1)
class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}
# mask = normalized_masks[0, class_to_idx["person"]]
boolean_masks = (normalized_masks.argmax(1) == sem_class_to_idx['person'])
transform = transforms.Resize((480, 640))
mask = transform(boolean_masks)
mask_np = mask.cpu().detach().numpy()


# img_with_masks = draw_segmentation_masks(img, masks=mask, alpha=0.7)
# img_with_masks = F.to_pil_image(img_with_masks)
# img_with_masks.show()


# show(dogs_with_masks)
#
# to_pil_image(mask).show()

