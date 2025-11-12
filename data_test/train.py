# train.py
import torch
import torch.nn as nn
import torch.optim as optim

# Importation des modules locaux
from models import get_model
from dataset import get_dataloaders

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, model_save_path, device):
    """
    Fonction principale pour l'entraînement et la validation d'un modèle.
    """
    best_accuracy = 0.0
    
    for epoch in range(num_epochs):
        print(f"Époque {epoch+1}/{num_epochs}")
        print("-" * 10)

        # Phase d'entraînement
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        print(f"Entraînement : Perte = {epoch_loss:.4f}, Précision = {epoch_acc:.4f}")

        # Phase de validation
        model.eval()
        running_loss = 0.0
        running_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

        val_loss = running_loss / len(val_loader.dataset)
        val_acc = running_corrects.double() / len(val_loader.dataset)
        print(f"Validation : Perte = {val_loss:.4f}, Précision = {val_acc:.4f}")

        # Sauvegarde le meilleur modèle
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"Nouveau meilleur modèle sauvegardé avec une précision de {best_accuracy:.4f} !")
        
        print("\n")

    print(f"Meilleure précision de validation finale : {best_accuracy:.4f}")

if __name__ == '__main__':
    # --- Configuration des paramètres ---
    data_dir = './data_test/data'
    model_save_path = './best_model_weights.pth'
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 25
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation de l'appareil : {device}")

    # --- Préparation des données et du modèle ---
    train_loader, val_loader, class_names = get_dataloaders(data_dir, batch_size)
    num_classes = len(class_names)
    print(f"Classes détectées : {class_names}")

    model = get_model(num_classes).to(device)
    
    # --- Définition de la fonction de perte et de l'optimiseur ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- Lancement de l'entraînement ---
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, model_save_path, device)