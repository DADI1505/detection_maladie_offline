# models.py
import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes):
    """
    Crée un modèle ResNet18 pré-entraîné et adapte la dernière couche
    pour correspondre au nombre de classes de votre dataset.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model