from torchvision.models import *
from .resnext101_32x4d import resnext101_32x4d
from .inception_v4 import inception_v4
from .inception_resnet_v2 import inception_resnet_v2
from .wrn50_2 import wrn50_2
from .my_densenet import densenet161, densenet121, densenet169, densenet201
from .fbresnet200 import fbresnet200
from .resnet50_dsd import resnet50_dsd
import torch.nn


def create_model(model_name='resnet50', pretrained=True, num_classes=1000, **kwargs):
    if model_name == 'resnet50':
        if pretrained:
            model = resnet50(pretrained=True, **kwargs)
            model.fc = torch.nn.Linear(2048, num_classes)
        else:
            model = resnet50(num_classes=num_classes)
    elif model_name == 'resnet101':
        if pretrained:
            model = resnet101(pretrained=True, **kwargs)
            model.fc = torch.nn.Linear(2048, num_classes)
        else:
            model = resnet101(num_classes=num_classes, **kwargs)
    elif model_name == 'resnet152':
        if pretrained:
            model = resnet152(pretrained=True, **kwargs)
            model.fc = torch.nn.Linear(2048, num_classes)
        else:
            model = resnet152(num_classes=num_classes, **kwargs)
    elif model_name == 'densenet121':
        if pretrained:
            model = densenet121(pretrained=True, **kwargs)
            model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
        else:
            model = densenet121(num_classes=num_classes, **kwargs)
    elif model_name == 'densenet161':
        if pretrained:
            model = densenet161(pretrained=True, **kwargs)
            model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
        else:
            model = densenet161(num_classes=num_classes, **kwargs)
    elif model_name == 'densenet169':
        if pretrained:
            model = densenet169(pretrained=True, **kwargs)
            model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
        else:
            model = densenet169(num_classes=num_classes, **kwargs)
    elif model_name == 'densenet201':
        if pretrained:
            model = densenet201(pretrained=True, **kwargs)
            model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
        else:
            model = densenet201(num_classes=num_classes, **kwargs)
    elif model_name == 'inception_resnet_v2':
        if pretrained:
            model = inception_resnet_v2(pretrained=True, **kwargs)
            model.classif = torch.nn.Linear(model.classif.in_features, num_classes)
        else:
            model = inception_resnet_v2(num_classes=num_classes, **kwargs)
    elif model_name == 'inception_v4':
        if pretrained:
            model = inception_v4(pretrained=True, **kwargs)
            model.classif = torch.nn.Linear(model.classif.in_features, num_classes)
        else:
            model = inception_v4(num_classes=num_classes, **kwargs)
    elif model_name == 'resnext101_32x4d':
        #activation_fn = torch.nn.LeakyReLU(0.1)
        activation_fn = torch.nn.SELU()
        if pretrained:
            model = resnext101_32x4d(pretrained=True, activation_fn=activation_fn, **kwargs)
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        else:
            model = resnext101_32x4d(num_classes=num_classes, activation_fn=activation_fn, **kwargs)
    elif model_name == 'wrn50':
        if pretrained:
            model = wrn50_2(pretrained=True, **kwargs)
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        else:
            model = wrn50_2(num_classes=num_classes, **kwargs)
    elif model_name == 'fbresnet200':
        if pretrained:
            model = fbresnet200(pretrained=True, **kwargs)
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        else:
            model = fbresnet200(num_classes=num_classes, **kwargs)
    elif model_name == 'resnet50_dsd':
        if pretrained:
            model = resnet50_dsd(pretrained=True, **kwargs)
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        else:
            model = resnet50_dsd(num_classes=num_classes, **kwargs)
    else:
        assert False and "Invalid model"
    return model


FC_NETS = ['resnet', 'inception', 'wrn', 'fbresnet', 'resnext']


def get_model_fc(model, model_name):
    if any(model_name.startswith(x) for x in FC_NETS):
        return model.fc
    elif model_name.starts_with('densenet'):
        return model.classifier
    else:
        assert False and "Invalid model"


def set_model_fc(model, fc, model_name):
    if any(model_name.startswith(x) for x in FC_NETS):
        model.fc = fc
    elif model_name.starts_with('densenet'):
        model.classifier = fc
    else:
        assert False and "Invalid model"
