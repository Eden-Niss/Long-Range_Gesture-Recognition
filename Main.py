import argparse
import logging
from Gesture_class import CNN2, train
# from dataloader import data_loaders
from data_loading import data_loaders
# from data_loading import data_loaders
from pretrained_models import pretrained_model_finetune, pretrained_model_whole
import wandb
from utils import test


parser = argparse.ArgumentParser(description='Training Config', add_help=False)

parser.add_argument('--csv_root', default=r'data/LongeRange_CSV/data_all_new2.csv', metavar='DIR',
                    help='path to csv dataset')
parser.add_argument('--img_root', default=r'/home/roblab20/PycharmProjects/LongRange/data/sr_data',
                    metavar='DIR',  help='path to csv dataset')
parser.add_argument('--root_train', default=r'/home/roblab20/PycharmProjects/LongRange/data/sr_data',
                    metavar='DIR', help='path to training dataset')
parser.add_argument('--root_val', default=r'', metavar='DIR',
                    help='path to training dataset')
parser.add_argument('--saveM_path', default=r'/home/roblab20/PycharmProjects/LongRange/checkpoint/DenseNet/SR_images/no_finetune',
                    metavar='DIR', help='path for save the weights in optimizer of the model')
parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 16)')
parser.add_argument('--criterion', default=r'rmse', metavar='CRI',
                    help='Criterion loss. (default: rmse)')
parser.add_argument('--num_classes', type=int, default=6,
                    help='Number of classes to classify')

# Optimizer parameters
parser.add_argument('--beta1', default=0.9156267464451984, type=float,
                    help='Optimizer beta1')
parser.add_argument('--beta2', default=0.9695956190145288, type=float,
                    help='Optimizer beta2')
parser.add_argument('--weight_decay', type=float, default=0.08,
                    help='weight decay')

parser.add_argument('--optim', type=str, default='Adam',
                    help='define optimizer type')
parser.add_argument('--scheduler', default='step', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')

# parser.add_argument('--lr', type=float, default=0.00184446, metavar='LR',
#                     help='learning rate')
parser.add_argument('--lr', type=float, default=1.1697595319788407e-05, metavar='LR',
                    help='learning rate')

parser.add_argument('--epochs', type=int, default=15, metavar='N',
                    help='number of epochs to train (default: 2)')

# # Misc
parser.add_argument('--img_size', type=int, default=224, metavar='Size',
                    help='Image size for resize')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log_interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--log_wandb', action='store_true', default=True,
                    help='log training and validation metrics to wandb')
parser.add_argument('-j', '--workers', type=int, default=2, metavar='N',
                    help='how many training processes to use (default: 2)')
parser.add_argument('--device', type=str, default='cuda:0',
                    help='type "cpu" if there is no gpu')
parser.add_argument("--drop_last", default=True, type=str)
parser.add_argument('--pretrained', default=True, type=bool,
                    help='Use pretrained model. (default: false)')
parser.add_argument('--fine_tune', default=False, type=bool,
                    help='Use pretrained model. (default: false)')
parser.add_argument('--pretrained_model', default='DenseNet', type=str,
                    help='Pretrained model can be either: DenseNet; EfficientNet; GoogLeNet; VGG; Wide_ResNet')


def main(args_config):
    classes = {'None': 0, 'Point': 1, 'Bad': 2, 'Good': 3, 'Stop': 4, 'Come': 5}
    # train_dataloader, val_dataloader, test_dataloader = data_loaders(args_config, classes)
    train_dataloader, val_dataloader, test_dataloader = data_loaders(args_config)

    if args_config.pretrained:
        if args_config.fine_tune:
            model = pretrained_model_finetune(args_config.pretrained_model, args_config.num_classes)
        else:
            model = pretrained_model_whole(args_config.pretrained_model, args_config.num_classes)
        model.to(args_config.device)
    else:
        model = CNN2(args_config.device)

    try:
        train(args_config, model, train_dataloader, val_dataloader) #train_dataloader
        test(args_config, model, test_dataloader)

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    args_config = parser.parse_args()
    # wandb.login()
    # wandb.init(project="my-new-test-project")
    # wandb.config = {
    #     "learning_rate": args_config.lr,
    #     "epochs": args_config.epochs,
    #     "batch_size": args_config.batch_size,
    # }
    main(args_config)
