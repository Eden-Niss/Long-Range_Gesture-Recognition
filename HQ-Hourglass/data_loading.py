from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os
import torch
from sklearn.model_selection import train_test_split


def get_filepaths(path):
    img_list = []
    lh_path = path + "/high"
    for file in os.listdir(lh_path):
        img_list.append(file)
    return img_list


class ClassDataset(Dataset):

    def __init__(self, train_dir, img_list, transform=None):
        self.LR_dir = train_dir + "/low/"
        self.HR_dir = train_dir + "/high/"
        self.transform = transform
        self.images = img_list

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        lr_image_path = self.LR_dir + self.images[index]
        hr_image_path = self.HR_dir + self.images[index]
        lr_img = Image.open(lr_image_path)
        # print(lr_img.size)
        hr_img = Image.open(hr_image_path)
        if self.transform:
            lr_img, hr_img = self.transform(lr_img), self.transform(hr_img)
        return lr_img, hr_img


def data_loaders(train_path):
    data_transforms = {
            'train':
                transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
            'validation':
                transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
            'test':
                transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        }

    train_img_list = get_filepaths(train_path)
    # print(len(train_img_list))

    train_data, test_data = train_test_split(train_img_list, test_size=0.1, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

    train_dataset = ClassDataset(train_path, train_data, data_transforms['train'])
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=2,
                                              shuffle=True, num_workers=1, drop_last=False)

    validation_dataset = ClassDataset(train_path, val_data, data_transforms['validation'])
    validationloader = torch.utils.data.DataLoader(validation_dataset, batch_size=2,
                                                   shuffle=True, num_workers=1, drop_last=False)

    test_dataset = ClassDataset(train_path, test_data, data_transforms['test'])
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                             shuffle=True, num_workers=1, drop_last=False)

    return trainloader, validationloader, testloader


if __name__ == '__main__':
    path = r'/home/roblab20/PycharmProjects/SuperResolution/data'
    trainloader, validationloader, testloader = data_loaders(path)
    for lr_image, hr_image in trainloader:
        a = hr_image.permute(2, 3, 1, 0).squeeze()
        print('a is ', a.size())
        print(type(hr_image))
        to_image = transforms.ToPILImage()
        l_image = to_image(lr_image.squeeze())
        hr_image = to_image(hr_image.squeeze())
        l_image.show()
        hr_image.show()
        break