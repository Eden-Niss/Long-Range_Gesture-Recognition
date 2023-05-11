import torch
from data_loading import data_loaders
from Gesture_class import CNN
import argparse
import warnings
import torch.nn as nn

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Test Config', add_help=False)

parser.add_argument('--root_test', default=r'', metavar='DIR',
                    help='path to training dataset')
parser.add_argument('--root_checkpoint', default=r'', metavar='DIR',
                    help='path to training dataset')
parser.add_argument('--img_size', type=int, default=224, metavar='Size',
                    help='Image size for resize')


def evaluate_model(test_set, model, device):
    gt_label = []
    preds = []
    for img, label in test_set:
        test_x = img.to(device)
        label = label.to(device)

        gt_label.append(label.item())

        with torch.no_grad():
            output = model(test_x)

        preds.append(output.item())

    gt_label = torch.tensor(gt_label)
    preds = torch.tensor(preds)

    criterion = nn.CrossEntropyLoss()
    ce = criterion(gt_label, preds)

    _, pred_class = torch.max(preds, dim=1)
    label_class = torch.nonzero(gt_label == 1, as_tuple=False)[:, 1]
    acc = torch.sum(torch.eq(pred_class, label_class)).item() / len(label_class)

    print(f'Test CE: {ce}. Test Accuracy: {acc}')


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    args_config = parser.parse_args()

    _, _, test_dataloader = data_loaders(args_config, test=True)

    model_path = args_config.root_checkpoint
    model = CNN(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    evaluate_model(test_dataloader, model, device)
