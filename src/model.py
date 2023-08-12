"""
@author: Mamdaliof
"""





import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class Resnet():
    def __init__(self,model_name):
      
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model = model = torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=True)
        (self.model).to(self.device)
        self.model.fc = nn.Sequential()

    def get_model(self):
        return self.model

    def freezing_layers(self, frozen_layers:list, partial_frozen_layers:list = [], num:int = 0, print_layers = True):
        count = 1
        for name, param in self.model.get_model().named_parameters():
            name = name.split(".")
            if name[0] in frozen_layers:
                param.requires_grad=False
                if print_layers:
                    print(f"{name}: requires_grad={param.requires_grad}")
            elif (name[0] in partial_frozen_layers) and (count <= num):
                param.requires_grad=False
                if print_layers:
                    print(f"{name}: requires_grad={param.requires_grad}")
                count += 1
            elif print_layers:
                    print(f"{name}: requires_grad={param.requires_grad}")



    def add_MLP(self, layers, in_dim = 2048):

        classifier = nn.Sequential()
        for i in range(layers-1):
            out_dim = in_dim//(2**(10//layers))
            classifier.add_module(f"linear{i}", nn.Linear(in_dim, out_dim))
            classifier.add_module(f"Dropout{i}",nn.Dropout(0.2))
            in_dim = out_dim
        classifier.add_module(f"final linear", nn.Linear(in_dim, 2))
        self.model.fc = classifier


