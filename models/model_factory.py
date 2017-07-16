from torchvision.models import *
from .resnext101_32x4d import resnext101_32x4d
from .inception_v4 import inception_v4
from .inception_resnet_v2 import inception_resnet_v2
from .wrn50_2 import wrn50_2
from .my_densenet import densenet161, densenet121, densenet169, densenet201
from .my_resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .fbresnet200 import fbresnet200
import torch.nn


def create_model(model_name='resnet50', pretrained=True, num_classes=1000, **kwargs):
    global_pool = kwargs.pop('global_pool', 'avg')
    if model_name == 'resnet18':
        if pretrained:
            model = resnet18(pretrained=True, **kwargs)
            model.reset_fc(num_classes, global_pool=global_pool)
        else:
            model = resnet18(num_classes=num_classes, global_pool=global_pool, **kwargs)
    if model_name == 'resnet34':
        if pretrained:
            model = resnet34(pretrained=True, **kwargs)
            model.reset_fc(num_classes, global_pool=global_pool)
        else:
            model = resnet34(num_classes=num_classes, global_pool=global_pool, **kwargs)
    if model_name == 'resnet50':
        if pretrained:
            model = resnet50(pretrained=True, **kwargs)
            model.reset_fc(num_classes, global_pool=global_pool)
        else:
            model = resnet50(num_classes=num_classes, global_pool=global_pool, **kwargs)
    elif model_name == 'resnet101':
        if pretrained:
            model = resnet101(pretrained=True, **kwargs)
            model.reset_fc(num_classes, global_pool=global_pool)
        else:
            model = resnet101(num_classes=num_classes, global_pool=global_pool, **kwargs)
    elif model_name == 'resnet152':
        if pretrained:
            model = resnet152(pretrained=True, **kwargs)
            model.reset_fc(num_classes, global_pool=global_pool)
        else:
            model = resnet152(num_classes=num_classes, global_pool=global_pool, **kwargs)
    elif model_name == 'densenet121':
        if pretrained:
            model = densenet121(pretrained=True, **kwargs)
            model.reset_fc(num_classes, global_pool=global_pool)
        else:
            model = densenet121(num_classes=num_classes, global_pool=global_pool, **kwargs)
    elif model_name == 'densenet161':
        if pretrained:
            model = densenet161(pretrained=True, **kwargs)
            model.reset_fc(num_classes, global_pool=global_pool)
        else:
            model = densenet161(num_classes=num_classes, global_pool=global_pool, **kwargs)
    elif model_name == 'densenet169':
        if pretrained:
            model = densenet169(pretrained=True, **kwargs)
            model.reset_fc(num_classes, global_pool=global_pool)
        else:
            model = densenet169(num_classes=num_classes, global_pool=global_pool, **kwargs)
    elif model_name == 'densenet201':
        if pretrained:
            model = densenet201(pretrained=True, **kwargs)
            model.reset_fc(num_classes, global_pool=global_pool)
        else:
            model = densenet201(num_classes=num_classes, global_pool=global_pool, **kwargs)
    elif model_name == 'inception_resnet_v2':
        if pretrained:
            model = inception_resnet_v2(pretrained=True, **kwargs)
            model.reset_fc(num_classes, global_pool=global_pool)
        else:
            model = inception_resnet_v2(num_classes=num_classes, global_pool=global_pool, **kwargs)
    elif model_name == 'inception_v4':
        if pretrained:
            model = inception_v4(pretrained=True, **kwargs)
            model.reset_fc(num_classes, global_pool=global_pool)
        else:
            model = inception_v4(num_classes=num_classes, global_pool=global_pool, **kwargs)
    elif model_name == 'resnext101_32x4d':
        kwargs.pop('activation_fn')
        activation_fn = torch.nn.LeakyReLU(0.1)  # torch.nn.SELU()
        if pretrained:
            model = resnext101_32x4d(pretrained=True, activation_fn=activation_fn, **kwargs)
            model.reset_fc(num_classes, global_pool=global_pool)
        else:
            model = resnext101_32x4d(
                num_classes=num_classes, activation_fn=activation_fn, global_pool=global_pool, **kwargs)
    elif model_name == 'wrn50':
        if pretrained:
            model = wrn50_2(pretrained=True, **kwargs)
            model.reset_fc(num_classes, global_pool=global_pool)
        else:
            model = wrn50_2(num_classes=num_classes, global_pool=global_pool, **kwargs)
    elif model_name == 'fbresnet200':
        if pretrained:
            model = fbresnet200(pretrained=True, **kwargs)
            model.reset_fc(num_classes, global_pool=global_pool)
        else:
            model = fbresnet200(num_classes=num_classes, global_pool=global_pool, **kwargs)
    else:
        assert False and "Invalid model"
    return model

