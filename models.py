import torch
import torch.nn as nn
import torchvision

#### RESNET50
def make_resnet50(pretrained=True):
    model = torchvision.models.resnet50(pretrained=pretrained)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.name ='resnet50'
    if not pretrained:
        model.load_state_dict(torch.load('./models/best_model_resnet50.pth'))
    return model

#### RESNET50 FT
def make_resnet50_ft(model_name, pretrained=True):
    model = torchvision.models.resnet50(pretrained=pretrained)
    for name, param in model.named_parameters():
        if "layer4" not in name and "fc" not in name:
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.name ='resnet50_ft'
    if not pretrained:
        model.load_state_dict(torch.load(model_name, map_location='cuda:0'))
    return model

#### RESNET18 FT
def make_resnet18_ft(pretrained=True):
    model = torchvision.models.resnet18(pretrained=pretrained)
    model.name ='resnet18_ft'
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.fc.in_features, 1)
        )
    if not pretrained:
        model.load_state_dict(torch.load('./best_model_resnet18_ft_full.pth', map_location='cuda:0'))
    return model

#### VGG16
def make_vgg16(pretrained=True):
    model = torchvision.models.vgg16(pretrained=pretrained)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[6] = torch.nn.Linear(4096, 1)
    model.name = 'vgg16'
    if not pretrained:
        model.load_state_dict(torch.load('./models/best_model_vgg16.pth'))
    return model

#### VGG16 FT
def make_vgg16_ft(pretrained=True):
    model = torchvision.models.vgg16(pretrained=pretrained)
    for param in model.parameters():
        param.requires_grad = False
    model.features[-3].requires_grad = True
    model.features[-2].requires_grad = True
    model.features[-1].requires_grad = True
    model.classifier[-1].requires_grad = True
    model.classifier[-1] = torch.nn.Linear(4096, 1)
    model.name = 'vgg16_ft'
    if not pretrained:
        model.load_state_dict(torch.load('./models/best_model_vgg16_ft.pth'))
    return model

#### MOBILENETV2
def make_mobilenetv2(pretrained=True):
    model = torchvision.models.mobilenet_v2(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_ftrs, 1)
    model.name = 'mobilenetv2'
    if not pretrained:
        model.load_state_dict(torch.load('./models/best_model_mobilenetv2.pth'))
    return model

#### INCEPTIONV3
def make_inceptionv3():
    model = torchvision.models.inception_v3(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model

#### CONVNEXT SMALL
def make_convnextsmall(pretrained=True):
    model = torchvision.models.convnext_small(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
    model.name ='convnextsmall'
    if not pretrained:
        model.load_state_dict(torch.load('./models/best_model_convnextsmall.pth'))
    return model

#### CONVNEXT SMALL FT
def make_convnextsmall_ft(pretrained=True):
    model = torchvision.models.convnext_small(pretrained=True)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
    model.name ='convnextsmall_ft'
    if not pretrained:
        model.load_state_dict(torch.load('./models/best_model_convnextsmall.pth'))
    return model

#### EFFICIENTNETB0
def make_efficientnetb0(model_name='', pretrained=True):
    model = torchvision.models.efficientnet_b0(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    model.name ='efficientnetb0'
    if not pretrained:
        model.load_state_dict(torch.load(model_name))
    return model

#### EFFICIENTNETB0_2
def make_efficientnetb0_2(model_name='', pretrained=True):
    model = torchvision.models.efficientnet_b0(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.name ='efficientnetb0'
    if not pretrained:
        model.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))
    return model

#### EFFICIENTNETB3
def make_efficientnetb3(model_name='', pretrained=True):
    model = torchvision.models.efficientnet_b3(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    model.name ='efficientnetb3'
    if not pretrained:
        model.load_state_dict(torch.load(model_name))
    return model

#### EFFICIENTNETB3_2
def make_efficientnetb3_2(model_name='', pretrained=True):
    model = torchvision.models.efficientnet_b3(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.name ='efficientnetb3'
    if not pretrained:
        model.load_state_dict(torch.load(model_name))
    return model