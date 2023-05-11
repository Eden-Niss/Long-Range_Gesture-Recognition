from torchvision.models import densenet121, DenseNet121_Weights, googlenet, GoogLeNet_Weights, vgg16, VGG16_Weights, \
    wide_resnet101_2, Wide_ResNet101_2_Weights
from efficientnet_pytorch import EfficientNet

from torch import nn


def load_pretrained_model(model_name, num_classes):
    """

    :param model_name: can be either of the following:
    DenseNet; EfficientNet; GoogLeNet; VGG; Wide_ResNet
    :return: the trained model
    """
    if model_name == 'DenseNet':
        WEIGHTS = DenseNet121_Weights.DEFAULT
        model = densenet121(weights=WEIGHTS)
        for param in model.parameters():
            param.requires_grad = False
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, num_classes)
    elif model_name == 'EfficientNet':
        model = EfficientNet.from_pretrained('efficientnet-b0')
        for param in model.parameters():
            param.requires_grad = False
        num_features = model._fc.in_features
        model._fc = nn.Linear(num_features, num_classes)
    elif model_name == 'GoogLeNet':
        WEIGHTS = GoogLeNet_Weights.DEFAULT
        model = googlenet(weights=WEIGHTS)
        for param in model.parameters():
            param.requires_grad = False
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif model_name == 'VGG':
        WEIGHTS = VGG16_Weights.DEFAULT
        model = vgg16(weights=WEIGHTS)
        for param in model.parameters():
            param.requires_grad = False
        num_features = model.classifier[6].in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif model_name == 'Wide_ResNet':
        WEIGHTS = Wide_ResNet101_2_Weights.DEFAULT
        model = wide_resnet101_2(weights=WEIGHTS)
        for param in model.parameters():
            param.requires_grad = False
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    else:
        raise ValueError("The pre-trained model must be either of the following:"
                         " DenseNet; EfficientNet; GoogLeNet; VGG; Wide_ResNet")
    return model
