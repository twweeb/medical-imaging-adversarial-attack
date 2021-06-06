from torch import nn
from torchvision import models


def load_model(arch='resnet18', num_classes=1):
    _model = None
    if arch == 'vgg16':
        _model = models.vgg16(pretrained=False, )
        _model.classifier[6] = nn.Linear(4096, num_classes)

    elif arch == 'resnet18':
        _model = models.resnet18(pretrained=False)
        num_ftrs = _model.fc.in_features

        _model.fc = nn.Linear(num_ftrs, num_classes)

    elif arch == 'resnet50':
        _model = models.resnet50(pretrained=False)
        num_ftrs = _model.fc.in_features

        _model.fc = nn.Linear(num_ftrs, num_classes)

    elif arch == 'densenet121':
        _model = models.densenet121(pretrained=False)

        _model.fc = nn.Linear(1024, num_classes)

    elif arch == 'resnet101':
        _model = models.resnet101(pretrained=False)
        num_ftrs = _model.fc.in_features

        _model.fc = nn.Linear(num_ftrs, num_classes)
    return _model