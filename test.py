import torch
from data_loading import data_loaders
from Gesture_class import CNN
import argparse
import warnings
import torch.nn as nn
from pretrained_models import load_model4test
from tqdm import tqdm

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Test Config', add_help=False)

parser.add_argument('--root_train', default=r'/home/roblab20/PycharmProjects/LongRange/data/data_LongRANGE', metavar='DIR',
                    help='path to training dataset')
parser.add_argument('--csv_root', default=r'data/LongeRange_CSV/data_all.csv', metavar='DIR',
                    help='path to csv dataset')
parser.add_argument('--img_root', default=r'/home/roblab20/PycharmProjects/LongRange/data/data_LongRANGE', metavar='DIR',
                    help='path to csv dataset')
parser.add_argument('--root_checkpoint', default=r'checkpoint/Wide_ResNet/crop/05_30_2023/20_net_Tue_May_30_06_14_57_2023.pt',
                    metavar='DIR', help='path to training dataset')
parser.add_argument('--img_size', type=int, default=224, metavar='Size',
                    help='Image size for resize')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 16)')
parser.add_argument("--drop_last", default=True, type=str)
parser.add_argument('--num_classes', type=int, default=6,
                    help='Number of classes to classify')
parser.add_argument('--pretrained', default=True, type=bool,
                    help='Use pretrained model. (default: false)')
parser.add_argument('--pretrained_model', default='Wide_ResNet', type=str,
                    help='Pretrained model can be either: DenseNet; EfficientNet; GoogLeNet; VGG; Wide_ResNet')
parser.add_argument('-j', '--workers', type=int, default=1, metavar='N',
                    help='how many training processes to use (default: 2)')


def evaluate_model(test_set, model, device):
    gt_label = []
    preds = []
    criterion = nn.CrossEntropyLoss()

    pbar = tqdm(test_set, total=len(test_set))
    for img, label in pbar:
        test_x = img.to(device)
        label = label.to(device)

        gt_label.append(label[0].detach().cpu())

        with torch.no_grad():
            output = model(test_x)

        preds.append(output[0].detach().cpu())

    gt_label = torch.stack(gt_label)
    preds = torch.stack(preds)

    ce = criterion(gt_label[0], preds[0])
    equals = []
    for i in range(len(gt_label)):
        _, pred_class = torch.max(preds[i], dim=0)
        label_class = torch.nonzero(gt_label[i] == 1, as_tuple=False)
        eq = torch.eq(pred_class, label_class)
        equals.append(eq.item())
    integer_list = list(map(int, equals))
    acc = sum(integer_list) / len(integer_list)

    print(f'Test CE: {ce}. Test Accuracy: {acc}')


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    args_config = parser.parse_args()

    classes = {'None': 0, 'Point': 1, 'Bad': 2, 'Good': 3, 'Stop': 4}
    _, _, test_dataloader = data_loaders(args_config)

    model_path = args_config.root_checkpoint

    if args_config.pretrained:
        model = load_model4test(args_config.pretrained_model, args_config.num_classes, args_config.root_checkpoint)
        model.to(device)
    else:
        model = CNN(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()

    evaluate_model(test_dataloader, model, device)
