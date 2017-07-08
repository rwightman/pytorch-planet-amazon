import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
from functools import reduce
from collections import OrderedDict

model_urls = {
    'resnet50_dsd': 'https://www.dropbox.com/s/g8wl9ea9pekmdca/resnet50_dsd-b5ae1a3f.pth?dl=1'
}


class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input


class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))


class LambdaMap(LambdaBase):
    def forward(self, input):
        return list(map(self.lambda_func, self.forward_prepare(input)))


class LambdaReduce(LambdaBase):
    def forward(self, input):
        return reduce(self.lambda_func, self.forward_prepare(input))


def resnet50_dsd_features():
    return nn.Sequential(  # Sequential,
        nn.Conv2d(3, 64, (7, 7), (2, 2), (3, 3)),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d((3, 3), (2, 2), (1, 1)),
        nn.Sequential(  # Sequential,
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(64, 64, (1, 1)),
                              nn.BatchNorm2d(64),
                              nn.ReLU(),
                              nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
                              nn.BatchNorm2d(64),
                              nn.ReLU(),
                              nn.Conv2d(64, 256, (1, 1)),
                          ),
                          nn.Conv2d(64, 256, (1, 1)),
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                nn.BatchNorm2d(256),
                nn.ReLU(),
            ),
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(256, 64, (1, 1)),
                              nn.BatchNorm2d(64),
                              nn.ReLU(),
                              nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
                              nn.BatchNorm2d(64),
                              nn.ReLU(),
                              nn.Conv2d(64, 256, (1, 1)),
                          ),
                          Lambda(lambda x: x),  # Identity,
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                nn.BatchNorm2d(256),
                nn.ReLU(),
            ),
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(256, 64, (1, 1)),
                              nn.BatchNorm2d(64),
                              nn.ReLU(),
                              nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
                              nn.BatchNorm2d(64),
                              nn.ReLU(),
                              nn.Conv2d(64, 256, (1, 1)),
                          ),
                          Lambda(lambda x: x),  # Identity,
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                nn.BatchNorm2d(256),
                nn.ReLU(),
            ),
        ),
        nn.Sequential(  # Sequential,
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(256, 128, (1, 1)),
                              nn.BatchNorm2d(128),
                              nn.ReLU(),
                              nn.Conv2d(128, 128, (3, 3), (2, 2), (1, 1)),
                              nn.BatchNorm2d(128),
                              nn.ReLU(),
                              nn.Conv2d(128, 512, (1, 1)),
                          ),
                          nn.Conv2d(256, 512, (1, 1), (2, 2)),
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                nn.BatchNorm2d(512),
                nn.ReLU(),
            ),
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(512, 128, (1, 1)),
                              nn.BatchNorm2d(128),
                              nn.ReLU(),
                              nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1)),
                              nn.BatchNorm2d(128),
                              nn.ReLU(),
                              nn.Conv2d(128, 512, (1, 1)),
                          ),
                          Lambda(lambda x: x),  # Identity,
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                nn.BatchNorm2d(512),
                nn.ReLU(),
            ),
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(512, 128, (1, 1)),
                              nn.BatchNorm2d(128),
                              nn.ReLU(),
                              nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1)),
                              nn.BatchNorm2d(128),
                              nn.ReLU(),
                              nn.Conv2d(128, 512, (1, 1)),
                          ),
                          Lambda(lambda x: x),  # Identity,
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                nn.BatchNorm2d(512),
                nn.ReLU(),
            ),
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(512, 128, (1, 1)),
                              nn.BatchNorm2d(128),
                              nn.ReLU(),
                              nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1)),
                              nn.BatchNorm2d(128),
                              nn.ReLU(),
                              nn.Conv2d(128, 512, (1, 1)),
                          ),
                          Lambda(lambda x: x),  # Identity,
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                nn.BatchNorm2d(512),
                nn.ReLU(),
            ),
        ),
        nn.Sequential(  # Sequential,
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(512, 256, (1, 1)),
                              nn.BatchNorm2d(256),
                              nn.ReLU(),
                              nn.Conv2d(256, 256, (3, 3), (2, 2), (1, 1)),
                              nn.BatchNorm2d(256),
                              nn.ReLU(),
                              nn.Conv2d(256, 1024, (1, 1)),
                          ),
                          nn.Conv2d(512, 1024, (1, 1), (2, 2)),
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                nn.BatchNorm2d(1024),
                nn.ReLU(),
            ),
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 256, (1, 1)),
                              nn.BatchNorm2d(256),
                              nn.ReLU(),
                              nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
                              nn.BatchNorm2d(256),
                              nn.ReLU(),
                              nn.Conv2d(256, 1024, (1, 1)),
                          ),
                          Lambda(lambda x: x),  # Identity,
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                nn.BatchNorm2d(1024),
                nn.ReLU(),
            ),
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 256, (1, 1)),
                              nn.BatchNorm2d(256),
                              nn.ReLU(),
                              nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
                              nn.BatchNorm2d(256),
                              nn.ReLU(),
                              nn.Conv2d(256, 1024, (1, 1)),
                          ),
                          Lambda(lambda x: x),  # Identity,
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                nn.BatchNorm2d(1024),
                nn.ReLU(),
            ),
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 256, (1, 1)),
                              nn.BatchNorm2d(256),
                              nn.ReLU(),
                              nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
                              nn.BatchNorm2d(256),
                              nn.ReLU(),
                              nn.Conv2d(256, 1024, (1, 1)),
                          ),
                          Lambda(lambda x: x),  # Identity,
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                nn.BatchNorm2d(1024),
                nn.ReLU(),
            ),
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 256, (1, 1)),
                              nn.BatchNorm2d(256),
                              nn.ReLU(),
                              nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
                              nn.BatchNorm2d(256),
                              nn.ReLU(),
                              nn.Conv2d(256, 1024, (1, 1)),
                          ),
                          Lambda(lambda x: x),  # Identity,
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                nn.BatchNorm2d(1024),
                nn.ReLU(),
            ),
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 256, (1, 1)),
                              nn.BatchNorm2d(256),
                              nn.ReLU(),
                              nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
                              nn.BatchNorm2d(256),
                              nn.ReLU(),
                              nn.Conv2d(256, 1024, (1, 1)),
                          ),
                          Lambda(lambda x: x),  # Identity,
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                nn.BatchNorm2d(1024),
                nn.ReLU(),
            ),
        ),
        nn.Sequential(  # Sequential,
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 512, (1, 1)),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                              nn.Conv2d(512, 512, (3, 3), (2, 2), (1, 1)),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                              nn.Conv2d(512, 2048, (1, 1)),
                          ),
                          nn.Conv2d(1024, 2048, (1, 1), (2, 2)),
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                nn.BatchNorm2d(2048),
                nn.ReLU(),
            ),
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(2048, 512, (1, 1)),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                              nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                              nn.Conv2d(512, 2048, (1, 1)),
                          ),
                          Lambda(lambda x: x),  # Identity,
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                nn.BatchNorm2d(2048),
                nn.ReLU(),
            ),
            nn.Sequential(  # Sequential,
                LambdaMap(lambda x: x,  # ConcatTable,
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(2048, 512, (1, 1)),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                              nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                              nn.Conv2d(512, 2048, (1, 1)),
                          ),
                          Lambda(lambda x: x),  # Identity,
                          ),
                LambdaReduce(lambda x, y: x + y),  # CAddTable,
                nn.BatchNorm2d(2048),
                nn.ReLU(),
            ),
        ),
    )

class ResNet50Dsd(nn.Module):

    def __init__(self, num_classes=1000, activation_fn=nn.ReLU(), drop_rate=0.):
        super(ResNet50Dsd, self).__init__()
        self.drop_rate = drop_rate
        self.features = resnet50_dsd_features() #activation_fn=activation_fn)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input):
        x = self.features(input)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        if self.drop_rate > 0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.fc(x)
        return x


def resnet50_dsd(pretrained=False, num_classes=1000, **kwargs):
    model = ResNet50Dsd(num_classes=num_classes, **kwargs)
    if pretrained:
        # Remap pretrained weights to match our class module with features + fc
        pretrained_weights = model_zoo.load_url(model_urls['resnet50_dsd'])
        feature_keys = filter(lambda k: '10.1.' not in k, pretrained_weights.keys())
        remapped_weights = OrderedDict()
        for k in feature_keys:
            remapped_weights['features.' + k] = pretrained_weights[k]
        remapped_weights['fc.weight'] = pretrained_weights['10.1.weight']
        remapped_weights['fc.bias'] = pretrained_weights['10.1.bias']
        model.load_state_dict(remapped_weights)
    return model