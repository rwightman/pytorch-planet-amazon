import torch
import torch.nn
import torch.nn.functional as F
import torch.autograd as autograd


class BayesianMultiLabelLoss(torch.nn.MultiLabelSoftMarginLoss):
    def __init__(self, num_classes, num_samples=256, weight=None, size_average=True):
        super(BayesianMultiLabelLoss, self).__init__(weight=weight, size_average=size_average)
        self.num_samples = num_samples
        self.num_classes = num_classes

    def forward(self, input, target):
        input_log_var = input[:, self.num_classes:]
        input_std = torch.sqrt(torch.exp(input_log_var))  # -input_log_var?
        input_pred = input[:, :self.num_classes]
        samples_var = autograd.Variable(torch.randn((self.num_samples,) + input_pred.size()).cuda())
        return torch.stack(
            [F.binary_cross_entropy(
                torch.sigmoid(input_pred + sample * input_std),
                target,
                self.weight,
                self.size_average) for sample in samples_var],
            dim=0).mean()


