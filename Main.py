import argparse
import logging
from Gesture_class import CNN, train
from data_loading import data_loaders
from pretrained_models import load_pretrained_model
import wandb


parser = argparse.ArgumentParser(description='Training Config', add_help=False)

parser.add_argument('--root_train', default=r'/home/roblab20/PycharmProjects/LongRange/data/data_LongRANGE', metavar='DIR',
                    help='path to training dataset')
parser.add_argument('--root_val', default=r'', metavar='DIR',
                    help='path to training dataset')
parser.add_argument('--saveM_path', default=r'/home/roblab20/PycharmProjects/LongRange/checkpoint', metavar='DIR',
                    help='path for save the weights in optimizer of the model')
parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 16)')
parser.add_argument('--criterion', default=r'rmse', metavar='CRI',
                    help='Criterion loss. (default: rmse)')
parser.add_argument('--pretrained', default=False, type=bool,
                    help='Use pretrained model. (default: false)')
parser.add_argument('--pretrained_model', default='DenseNet', type=str,
                    help='Pretrained model can be either: DenseNet; EfficientNet; GoogLeNet; VGG; Wide_ResNet')
parser.add_argument('--num_classes', type=int, default=5,
                    help='Number of classes to classify')

# Optimizer parameters
parser.add_argument('--beta1', default=0.9159433559021458, type=float,
                    help='Optimizer beta1')
parser.add_argument('--beta2', default=0.957699106385856, type=float,
                    help='Optimizer beta2')
parser.add_argument('--weight_decay', type=float, default=0.006126044388180182,
                    help='weight decay')
parser.add_argument('--optim', type=str, default='Adam',
                    help='define optimizer type')
parser.add_argument('--scheduler', default='step', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
parser.add_argument('--lr', type=float, default=9.853023135481731e-05, metavar='LR',
                    help='learning rate')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
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
parser.add_argument('-j', '--workers', type=int, default=0, metavar='N',
                    help='how many training processes to use (default: 2)')
parser.add_argument('--device', type=str, default='cuda:0',
                    help='type "cpu" if there is no gpu')
parser.add_argument("--drop_last", default=True, type=str)
parser.add_argument("--load_model", default=False, type=str)


def main(args_config):

    train_dataloader, val_dataloader = data_loaders(args_config)

    if args_config.pretrained:
        model = load_pretrained_model(args_config.pretrained_model, args_config.num_classes)
        model.to(args_config.device)
    else:
        model = CNN(args_config.device)

    try:
        train(args_config, model, train_dataloader, val_dataloader)

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