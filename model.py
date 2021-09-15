import torch
from torch import nn
import timm

from torch.nn import functional as F
from torch.nn.parameter import Parameter
import cv2
from torchsummary import summary

#ArgMargin
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        return cosine


class Backbone(nn.Module):
    def __init__(self, name = 'resnet18', pretrained =  True):
        super(Backbone, self).__init__()
        self.name = name
        self.net = timm.create_model(self.name, pretrained)

        if 'regnet' in name:
            self.out_feature = self.net.fc.in_features
        elif 'csp' in name:
            self.out_feature = self.net.head.fc.in_features
        elif 'res' in name: #works also for resnest
                self.out_features = self.net.fc.in_features
        elif 'efficientnet' in name:
            self.out_features = self.net.classifier.in_features
        elif 'densenet' in name:
            self.out_features = self.net.classifier.in_features
        elif 'senet' in name:
            self.out_features = self.net.fc.in_features
        elif 'inception' in name:
            self.out_features = self.net.last_linear.in_features

        else:
            self.out_features = self.net.classifier.in_features

    def forward(self, x):
        x = self.net.forward_features(x)
        return x

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable = True):
        super(GeM,self).__init__()
        if p_trainable:
            self.p = nn.Parameter(torch.ones(1)*p)
        else:
            self.p = p

        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

class Net(nn.Module):
    def __init__(self,  args, pretrained = True):
        super(Net, self).__init__()
        self.args = args
        self.backbone = Backbone(self.args["backbone"], pretrained = pretrained)

        if self.args["pool"] == "gem":
            self.global_pool = GeM(p_trainable = self.args["p_trainable"])

        elif self.args["pool"] == "identity":
            self.global_pool = torch.nn.Identity()
        else:
            self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.embedding_size = self.args["embedding_size"] #before output

        if self.args["neck"] == "option-D":
            self.neck = nn.Sequential(
                    nn.Linear(self.backbone.out_features, self.embedding_size, bias = True),
                    nn.BatchNorm1d(self.embedding_size),
                    torch.nn.PReLU()
                    )

        elif args["neck"] == "option-F": #using drop-out
            self.neck = nn.Sequential(
                    nn.Dropout(0.3),
                    nn.Linear(self.backbone.out_features, self.embedding_size, bias = True),
                    nn.BatchNorm1d(self.embedding_size),
                    torch.nn.PReLU()
                    )
            
        self.head = ArcMarginProduct(self.embedding_size, self.args["n_classes"])

        if self.args["pretrained_weights"] is not None:
            self.load_state_dict(torch.load(self.args["pretrained_weights"], map_location = 'cpu'), strict = False)
            print("Weights loaded from: {}".format(self.args["pretrained_weights"]))


    def forward(self, x, get_embeddings = False, get_attentions = False):
        x = self.backbone(x) #output before Gem-pooling

        x = self.global_pool(x)

        x = x[:,:,0,0]
        x = self.neck(x)
        logits = self.head(x) #cosine
        return logits
        #print(logits)
        if get_embeddings:
            return {'logits': logits, 'embeddings': x}
        else:
            return {'logits': logits}

if __name__ == '__main__':
    
    args = { 
    'seed':1138,
    'save_weights_only':True,

    'resume_from_checkpoint': None,
    'pretrained_weights':None,

    'normalization':'imagenet',

    'backbone':'gluon_seresnext101_32x4d',
    'embedding_size': 512,
    'pool': 'gem',

    'p_trainable': True,

    'neck': 'option-D',
    'head':'arc_margin',

    'crit': "focal",
    'loss':'arcface',

    'n_classes':32,

    }
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    model = Net(args)
    model = model.to(device)
    summary(model, (3, 512,512))


