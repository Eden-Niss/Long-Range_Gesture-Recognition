from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os
import torch
import pickle


def get_filepaths(path):
    name_img = []
    for root, directories, files in os.walk(path):
        if root.endswith('image'):
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
        for i in range(2, 21):
            full_path = self.images[index]
            if os.path.exists(full_path):
                image_path = full_path
            else:
                continue
        img = Image.open(image_path)
        if self.transform:
            img = self.transform(img)
        return img, label


def data_loaders(args, test=False):
    data_transforms = {
            'train':
                transforms.Compose([
                    transforms.Resize((args.img_size, args.img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
            'jitter_aug':
                transforms.Compose([
                    transforms.Resize((args.img_size, args.img_size)),
                    transforms.ColorJitter(brightness=1.0, contrast=0.1, saturation=0.5, hue=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
            'blur':
                transforms.Compose([
                    transforms.Resize((args.img_size, args.img_size)),
                    transforms.ToTensor(),
                    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
            'flip':
                transforms.Compose([
                    transforms.Resize((args.img_size, args.img_size)),
                    transforms.RandomHorizontalFlip(p=1),
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

    if test:
        test_path = args.root_test
        test_img_list = ['Point', 'None', 'Bad', 'Good', 'Stop']

        test_dataset = ClassDataset(test_path, test_img_list, data_transforms['test'])

        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                                 shuffle=False, num_workers=0, drop_last=False)
        return testloader

    else:
        train_path = args.root_train

        input_types = ['Point', 'None', 'Bad', 'Good', 'Stop']
        train_img_list = get_filepaths(train_path)

        # validation_path = args.root_val
        # validation_img_list = ['Point', 'None', 'Bad', 'Good', 'Stop']

        train_dataset = ClassDataset(train_path, train_img_list, input_types, data_transforms['train'])
        jitter_dataset = ClassDataset(train_path, train_img_list, input_types, data_transforms['jitter_aug'])
        flip_dataset = ClassDataset(train_path, train_img_list, input_types, data_transforms['flip'])
        blur_dataset = ClassDataset(train_path, train_img_list, input_types, data_transforms['blur'])

        full_data_train = torch.utils.data.ConcatDataset([train_dataset, flip_dataset, jitter_dataset, blur_dataset])

        train_size = int(len(full_data_train) * 0.8)
        val_size = len(full_data_train) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_data_train, [train_size, val_size])

        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                  shuffle=True, num_workers=args.workers, drop_last=args.drop_last)

        # validation_dataset = ClassDataset(validation_path, validation_img_list, data_transforms['validation'])
        validationloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                                       shuffle=True, num_workers=args.workers, drop_last=args.drop_last)
        return trainloader, validationloader


if __name__ == '__main__':
    path = r'/home/roblab20/PycharmProjects/PointClassification/data_rearranged/train'
    data_loaders(path, img_size=224)


