from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os
import torch
import pickle
from sklearn.model_selection import train_test_split


def get_filepaths(path):
    name_img = []
    for root, directories, files in os.walk(path):
        if root.endswith('crop'):
            for filename in files:
                name_img.append(os.path.join(root, filename))
    return name_img

#
# def pre_processing(name_pkl):
#     dis = []
#     img_name = []
#     Dtrain = []
#     for cat in name_pkl:
#         meter_data = pickle.load(file=open(cat, "rb"))
#         Dtrain.append(meter_data)
#         # dis.append(meter_data['distance'])
#         # img_name.append(meter_data['id_name'])
#     return meter_data


class ClassDataset(Dataset):

    def __init__(self, train_dir, img_list, input_types, transform=None):
        self.train_dir = train_dir
        self.transform = transform
        self.input_types = input_types
        self.images = img_list

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_class = self.images[index].split("/")[-1].split("_")[0]
        I_matrix = torch.eye(len(self.input_types))
        label = I_matrix[self.input_types.index(image_class)]
        for i in range(1, 21):
            full_path = self.images[index]
            if os.path.exists(full_path):
                image_path = full_path
            else:
                continue
        img = Image.open(image_path)
        if self.transform:
            img = self.transform(img)
        return img, label


def data_loaders(args):
    data_transforms = {
            'train':
                transforms.Compose([
                    transforms.Resize((args.img_size, args.img_size)),
                    transforms.ColorJitter(brightness=.5, hue=.3),
                    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
            'validation':
                transforms.Compose([
                    transforms.Resize((args.img_size, args.img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
            'test':
                transforms.Compose([
                    transforms.Resize((args.img_size, args.img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        }

    train_path = args.root_train

    input_types = ['None', 'Point', 'Bad', 'Good', 'Stop', 'Come']
    train_img_list = get_filepaths(train_path)

    train_data, test_data = train_test_split(train_img_list, test_size=0.1, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

    train_dataset = ClassDataset(train_path, train_data, input_types, data_transforms['train'])
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.workers, drop_last=args.drop_last)

    validation_dataset = ClassDataset(train_path, val_data, input_types, data_transforms['validation'])
    validationloader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.batch_size,
                                                   shuffle=True, num_workers=args.workers, drop_last=args.drop_last)

    test_dataset = ClassDataset(train_path, test_data, input_types, data_transforms['test'])
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                   shuffle=True, num_workers=args.workers, drop_last=args.drop_last)

    return trainloader, validationloader, testloader


if __name__ == '__main__':
    path = r'/home/roblab20/PycharmProjects/PointClassification/data_rearranged/train'
    data_loaders(path, img_size=224)


