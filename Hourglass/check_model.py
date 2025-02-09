import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from RUNET import RUNet
# from simple_UNET import UNET
import os


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
path = '/home/roblab20/PycharmProjects/SuperResolution/results/before.png'
img = Image.open(path)

path2save = '/home/roblab20/PycharmProjects/SuperResolution/results'
before = os.path.join(path2save, "before.png")
after1 = os.path.join(path2save, "after_UNET_edge_loss.png")
after2 = os.path.join(path2save, "after_RUNE_from_longrangeT.png")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

lr_img = transform(img).unsqueeze(0)


wights_path = '/home/roblab20/PycharmProjects/SuperResolution/checkpoint/07_05_2023/1_net_Wed_Jul__5_21_51_32_2023.pt'
model = RUNet()

# wights_path = '/home/roblab20/PycharmProjects/SuperResolution/checkpoint/06_27_2023/10_net_Tue_Jun_27_01_25_13_2023.pt'
# model = RUNet()

model.load_state_dict(torch.load(wights_path, map_location=device))
model.to(device)
model.eval()

with torch.no_grad():
    output = model(lr_img.to(device))

mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
output = output.squeeze().cpu().numpy()

hr_img = output.transpose((1, 2, 0)) * std + mean
hr_img = np.clip(hr_img, 0, 1)
hr_img = (hr_img * 255).astype(np.uint8)
hr_img = Image.fromarray(hr_img)
# img.save(before)
# hr_img.save(after1)
img.show()
hr_img.show()
