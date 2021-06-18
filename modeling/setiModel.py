import timm
import torch.nn as nn

class setiModel(nn.Module):
    def __init__(self, opt, pretrained=False):
        super().__init__()
        self.opt = opt
        self.model = timm.create_model(self.opt.arch, pretrained=pretrained, in_chans=1)
        if self.opt.arch.startswith('nfnet'):
            self.n_features = self.model.head.fc.in_features
            self.model.head.fc = nn.Linear(self.n_features, self.opt.target_size)
        elif self.opt.arch.startswith('efficientnetv2_l'):
            self.n_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(self.n_features, self.opt.target_size)
        elif self.opt.arch.startswith('resnet'):
            self.n_features = self.model.fc.in_features
            self.model.fc = nn.Linear(self.n_features, self.opt.target_size)
    def forward(self, x):
        output = self.model(x)
        return output