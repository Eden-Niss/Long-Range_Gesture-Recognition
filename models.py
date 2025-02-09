import torch
from torch import nn


def pretrained_model_finetune(model_name, num_classes):
    """

    :param model_name: can be either of the following:
    DenseNet; EfficientNet; GoogLeNet; VGG; Wide_ResNet
    :return: the trained model
    """
    if model_name == 'DenseNet':
        from torchvision.models import densenet121, DenseNet121_Weights
        WEIGHTS = DenseNet121_Weights.DEFAULT
        model = densenet121(weights=WEIGHTS)
        for param in model.parameters():
            param.requires_grad = False
        num_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_features, num_classes),
            nn.Softmax()
        )
    elif model_name == 'EfficientNet':
        from efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_pretrained('efficientnet-b0')
        for param in model.parameters():
            param.requires_grad = False
        num_features = model._fc.in_features
        model._fc = nn.Sequential(
            nn.Linear(num_features, num_classes),
            nn.Softmax()
        )
    elif model_name == 'GoogLeNet':
        from torchvision.models import googlenet, GoogLeNet_Weights
        WEIGHTS = GoogLeNet_Weights.DEFAULT
        model = googlenet(weights=WEIGHTS)
        for param in model.parameters():
            param.requires_grad = False
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, num_classes),
            nn.Softmax()
        )
    elif model_name == 'VGG':
        from torchvision.models import vgg16, VGG16_Weights
        WEIGHTS = VGG16_Weights.DEFAULT
        model = vgg16(weights=WEIGHTS)
        for param in model.parameters():
            param.requires_grad = False
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
            nn.Linear(num_features, num_classes),
            nn.Softmax()
        )
    elif model_name == 'Wide_ResNet':
        from torchvision.models import wide_resnet101_2, Wide_ResNet101_2_Weights
        WEIGHTS = Wide_ResNet101_2_Weights.DEFAULT
        model = wide_resnet101_2(weights=WEIGHTS)
        for param in model.parameters():
            param.requires_grad = False
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, num_classes),
            nn.Softmax()
        )
    else:
        raise ValueError("The pre-trained model must be either of the following:"
                         " DenseNet; EfficientNet; GoogLeNet; VGG; Wide_ResNet")
    return model


def pretrained_model_whole(model_name, num_classes):
    """

    :param model_name: can be either of the following:
    DenseNet; EfficientNet; GoogLeNet; VGG; Wide_ResNet
    :return: the trained model
    """
    if model_name == 'DenseNet':
        from torchvision.models import densenet121, DenseNet121_Weights
        WEIGHTS = DenseNet121_Weights.DEFAULT
        model = densenet121(weights=WEIGHTS)
        num_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_features, num_classes),
            nn.Softmax()
        )
    elif model_name == 'EfficientNet':
        from efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_pretrained('efficientnet-b0')
        num_features = model._fc.in_features
        model._fc = nn.Sequential(
            nn.Linear(num_features, num_classes),
            nn.Softmax()
        )
    elif model_name == 'GoogLeNet':
        from torchvision.models import googlenet, GoogLeNet_Weights
        WEIGHTS = GoogLeNet_Weights.DEFAULT
        model = googlenet(weights=WEIGHTS)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, num_classes),
            nn.Softmax()
        )
    elif model_name == 'VGG':
        from torchvision.models import vgg16, VGG16_Weights
        WEIGHTS = VGG16_Weights.DEFAULT
        model = vgg16(weights=WEIGHTS)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
            nn.Linear(num_features, num_classes),
            nn.Softmax()
        )
    elif model_name == 'Wide_ResNet':
        from torchvision.models import wide_resnet101_2, Wide_ResNet101_2_Weights
        WEIGHTS = Wide_ResNet101_2_Weights.DEFAULT
        model = wide_resnet101_2(weights=WEIGHTS)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, num_classes),
            nn.Softmax()
        )
    else:
        raise ValueError("The pre-trained model must be either of the following:"
                         " DenseNet; EfficientNet; GoogLeNet; VGG; Wide_ResNet")
    return model


def load_model4test(model_name, num_classes, weights_path, device):
    """

    :param model_name: can be either of the following:
    DenseNet; EfficientNet; GoogLeNet; VGG; Wide_ResNet
    :return: the trained model
    """
    if model_name == 'DenseNet':
        from torchvision.models import densenet121, DenseNet121_Weights
        model = densenet121(pretrained=False)
        num_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_features, num_classes),
            nn.Softmax()
        )
        model.load_state_dict(torch.load(weights_path))

    elif model_name == 'EfficientNet':
        from efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_name('efficientnet-b0')
        num_features = model._fc.in_features
        model._fc = nn.Sequential(
            nn.Linear(num_features, num_classes),
            nn.Softmax()
        )
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)

    elif model_name == 'GoogLeNet':
        from torchvision.models import googlenet, GoogLeNet_Weights
        model = googlenet(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, num_classes),
            nn.Softmax()
        )
        model.load_state_dict(torch.load(weights_path), strict=False)

    elif model_name == 'VGG':
        from torchvision.models import vgg16, VGG16_Weights
        model = vgg16(pretrained=False)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
            nn.Linear(num_features, num_classes),
            nn.Softmax()
        )
        model.load_state_dict(torch.load(weights_path))

    elif model_name == 'Wide_ResNet':
        from torchvision.models import wide_resnet101_2, Wide_ResNet101_2_Weights
        model = wide_resnet101_2(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, num_classes),
            nn.Softmax()
        )
        model.load_state_dict(torch.load(weights_path))

    else:
        raise ValueError("The pre-trained model must be either of the following:"
                         " DenseNet; EfficientNet; GoogLeNet; VGG; Wide_ResNet")
    return model.to(device)