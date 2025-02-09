import argparse
from RUNET import RUNet
from train import train
from data_loading import data_loaders
from test import test
# from simple_UNET import UNET
from torchvision.models import vgg19

parser = argparse.ArgumentParser(description='Training Config', add_help=False)

parser.add_argument('--dataset_root', type=str, default=r'/home/roblab20/PycharmProjects/SuperResolution/data', metavar='DIR',
                    help='path to csv dataset')
parser.add_argument('--saveM_path', default=r'/home/roblab20/PycharmProjects/SuperResolution/checkpoint', metavar='DIR',
                    help='path for save the weights in optimizer of the model')
parser.add_argument('--path2results', default=r'/home/roblab20/PycharmProjects/SuperResolution/results', metavar='DIR',
                    help='path for save the weights in optimizer of the model')
parser.add_argument('--batch_size', type=int, default=4, metavar='N',
                    help='input batch size for training (default: 16)')


# Optimizer parameters
parser.add_argument('--beta1', default=0.9156267464451984, type=float,
                    help='Optimizer beta1')
parser.add_argument('--beta2', default=0.9695956190145288, type=float,
                    help='Optimizer beta2')
parser.add_argument('--weight_decay', type=float, default=0.08,
                    help='weight decay')

parser.add_argument('--optim', type=str, default='Adam',
                    help='define optimizer type')
parser.add_argument('--lr', type=float, default=1e-05, metavar='LR',
                    help='learning rate')

parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 2)')

# # Misc
parser.add_argument('--img_size', type=int, default=224, metavar='Size',
                    help='Image size for resize')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('-j', '--workers', type=int, default=0, metavar='N',
                    help='how many training processes to use (default: 2)')
parser.add_argument('--device', type=str, default='cuda:0',
                    help='type "cpu" if there is no gpu')
parser.add_argument("--drop_last", default=True, type=str)
parser.add_argument('--pretrained', default=False, type=bool,
                    help='Use pretrained model. (default: false)')
parser.add_argument('--fine_tune', default=False, type=bool,
                    help='Use pretrained model. (default: false)')
parser.add_argument('--pretrained_model', default='GoogLeNet', type=str,
                    help='Pretrained model can be either: DenseNet; EfficientNet; GoogLeNet; VGG; Wide_ResNet')


def main(args_config):
    device = args_config.device
    train_dataloader, val_dataloader, test_dataloader = data_loaders(args_config.dataset_root)

    model = RUNet()
    # model = UNET()
    model.to(device)

    # VGG19
    vgg_model = vgg19(pretrained=True).features[:36]
    vgg_model.to(device)

    try:
        train(args_config, model, vgg_model, train_dataloader, val_dataloader) #train_dataloader
        test(args_config, model, vgg_model, test_dataloader)

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    args_config = parser.parse_args()
    main(args_config)
