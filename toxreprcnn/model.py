import timm
import torch
import torch.nn as nn


class ToxReprCNNModel(nn.Module):
    def __init__(self, model_name: str, pretrained: bool = False, num_classes: int = 0):
        super().__init__()
        self.model = timm.create_model(
            model_name=model_name, pretrained=pretrained, num_classes=0)
        self.n_features = self.model.in_features
        self.n_classes = num_classes
        self.fc = nn.Linear(self.n_features, num_classes)

    def forward(self, x: torch.Tensor):
        x = self.model(x)
        x = self.fc(x)
        return x

    def forward_features(self, x: torch.Tensor):
        x = self.model(x)
        return x


class FrozenEffnetB4Model(nn.Module):
    def __init__(self, depth, num_classes):
        super().__init__()
        self.model = timm.create_model("tf_efficientnet_b4_ns", num_classes=0, pretrained=True)
        block_list = []
        for i in range(depth):
            block_list.append(self.model.blocks[6-depth+1+i])
            self.model.blocks[6-depth+1+i] = nn.Identity()
        conv_head = self.model.conv_head
        bn2 = self.model.bn2
        global_pool = self.model.global_pool
        self.head = nn.Sequential(*block_list, conv_head, bn2, global_pool)
        self.model.conv_head = nn.Identity()
        self.model.bn2 = nn.Identity()
        self.model.global_pool = nn.Identity()
        self.classifier = nn.Linear(1792, num_classes)
        for params in self.model.parameters():
            params.requires_grad = False
    
    def train(self, train_flag=True):
        super().train(train_flag)
        self.model.eval()
    
    def forward(self, x):
        x = self.model(x)
        x = self.head(x)
        x = self.classifier(x)
        return x

class FrozenEffnetB4ModelMO(nn.Module):
    def __init__(self, depth, num_classes):
        super().__init__()
        self.depth = depth
        self.model = timm.create_model("tf_efficientnet_b4_ns", num_classes=0, pretrained=True)
        block_list = []
        for i in range(depth):
            block_list.append(self.model.blocks[6-depth+1+i])
            self.model.blocks[6-depth+1+i] = nn.Identity()
        conv_head = self.model.conv_head
        bn2 = self.model.bn2
        global_pool = self.model.global_pool
        self.pool = global_pool
        self.head = nn.Sequential(*block_list, conv_head, bn2, global_pool)
        self.model.conv_head = nn.Identity()
        self.model.bn2 = nn.Identity()
        self.model.global_pool = nn.Identity()
        self.classifier = nn.Linear(1792, num_classes)
        for params in self.model.parameters():
            params.requires_grad = False
    
    def train(self, train_flag=True):
        super().train(train_flag)
        self.model.eval()

    def forward(self, x):
        res = []
        x = self.model.conv_stem(x)
        x = self.model.bn1(x)
        res.append(self.pool(x))
        for i in range(7-self.depth):
            x = self.model.blocks[i](x)
            res.append(self.pool(x))
        for i in range(self.depth):
            x = self.head[i](x)
            res.append(self.pool(x))
        x = self.head[-3](x)
        x = self.head[-2](x)
        res.append(self.pool(x))
        return res

class EffnetB4ModelMO(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.model = timm.create_model("tf_efficientnet_b4_ns", num_classes=num_classes, pretrained=pretrained)
        self.layers = [nn.Sequential(self.model.conv_stem, self.model.bn1),
                        self.model.blocks[0],
                        self.model.blocks[1],
                        self.model.blocks[2],
                        self.model.blocks[3],
                        self.model.blocks[4],
                        self.model.blocks[5],
                        self.model.blocks[6],
                        nn.Sequential(self.model.conv_head, self.model.bn2)]
        self.pool = self.model.global_pool
    def forward(self, x):
        ret = []
        for layer in self.layers:
            x = layer(x)
            ret.append(self.pool(x.clone()))
        return ret
    
class FrozenResNet50Model(nn.Module):
    def __init__(self, depth, num_classes):
        assert 0<=depth<=5
        super().__init__()
        self.model = timm.create_model("resnetaa50", num_classes=0, pretrained=True)
        layer_list = []
        if depth>=1:
            layer_list = [self.model.layer4] + layer_list
            self.model.layer4 = nn.Identity()
        if depth>=2:
            layer_list = [self.model.layer3] + layer_list
            self.model.layer3 = nn.Identity()
        if depth>=3:
            layer_list = [self.model.layer2] + layer_list
            self.model.layer2 = nn.Identity()
        if depth>=4:
            layer_list = [self.model.layer1] + layer_list
            self.model.layer1 = nn.Identity()
        if depth>=5:
            layer_list = [self.model.conv1, self.model.bn1, self.model.act1, self.model.max_pool] + layer_list
            self.model.conv1 = nn.Identity()
            self.model.bn1 = nn.Identity()
            self.model.act1 = nn.Identity()
            self.model.max_pool = nn.Identity()
        global_pool = self.model.global_pool
        self.model.global_pool = nn.Identity()
        self.head = nn.Sequential(*layer_list, global_pool)
        if num_classes>0:
            self.classifier = nn.Linear(2048, num_classes)
        else:
            self.classifier = nn.Identity()
        for params in self.model.parameters():
            params.requires_grad = False
    
    def train(self, train_flag=True):
        super().train(train_flag)
        self.model.eval()
    
    def forward(self, x):
        x = self.model(x)
        x = self.head(x)
        x = self.classifier(x)
        return x

    def forward_features(self, x):
        x = self.model(x)
        x = self.head(x)
        return x

class FrozenResNet50ModelMO(nn.Module):
    def __init__(self, depth, num_classes):
        assert 0<=depth<=5
        super().__init__()
        self.model = timm.create_model("resnetaa50", num_classes=0, pretrained=True)
        layer_list = []
        self.layers = [[self.model.conv1, self.model.bn1, self.model.act1, self.model.maxpool],
                       [self.model.layer1],
                       [self.model.layer2],
                       [self.model.layer3],
                       [self.model.layer4]]
        self.pool = self.model.global_pool
        if depth>=1:
            layer_list = [self.model.layer4] + layer_list
            self.model.layer4 = nn.Identity()
        if depth>=2:
            layer_list = [self.model.layer3] + layer_list
            self.model.layer3 = nn.Identity()
        if depth>=3:
            layer_list = [self.model.layer2] + layer_list
            self.model.layer2 = nn.Identity()
        if depth>=4:
            layer_list = [self.model.layer1] + layer_list
            self.model.layer1 = nn.Identity()
        if depth>=5:
            layer_list = [self.model.conv1, self.model.bn1, self.model.act1, self.model.maxpool] + layer_list
            self.model.conv1 = nn.Identity()
            self.model.bn1 = nn.Identity()
            self.model.act1 = nn.Identity()
            self.model.max_pool = nn.Identity()
        global_pool = self.model.global_pool
        self.model.global_pool = nn.Identity()
        self.head = nn.Sequential(*layer_list, global_pool)
        if num_classes>0:
            self.classifier = nn.Linear(2048, num_classes)
        else:
            self.classifier = nn.Identity()
        for params in self.model.parameters():
            params.requires_grad = False
    
    def train(self, train_flag=True):
        super().train(train_flag)
        self.model.eval()
    
    def forward(self, x):
        ret = []
        for layer in self.layers:
            for m in layer:
                x = m(x)
            ret.append(self.pool(x.clone()))
        return ret

class ResNet50ModelMO(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model("resnetaa50", num_classes=num_classes, pretrained=True)
        self.layers = [[self.model.conv1, self.model.bn1, self.model.act1, self.model.maxpool],
                       [self.model.layer1],
                       [self.model.layer2],
                       [self.model.layer3],
                       [self.model.layer4]]
        self.pool = self.model.global_pool
        
    def forward(self, x):
        ret = []
        for layer in self.layers:
            for m in layer:
                x = m(x)
            ret.append(self.pool(x.clone()))
        return ret

