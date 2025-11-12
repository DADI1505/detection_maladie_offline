# predict.py
import torch
from PIL import Image
from torchvision import transforms
import os

# Importation du module local
from models import get_model

def predict_image(model, image_path, class_names, device):
    """
    Charge un modèle entraîné, applique les transformations nécessaires à une image,
    et retourne la prédiction.
    """
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted_class = torch.max(probabilities, 1)
        
    return class_names[predicted_class.item()], probabilities[0][predicted_class].item()

if __name__ == '__main__':
    # --- Configuration ---
    model_path = './best_model_weights.pth'
    
    # Remplacez ceci par le chemin de votre image à prédire
    image_to_predict = './path/to/your/image.jpg'
    
    # Vous devez lister les noms de vos classes dans le même ordre que vos dossiers
    class_names = ['Bacterial_spot', 'Early_blight', 'healthy', 'Late_blight', 'Leaf_Mold', 'powdery_mildew', 
                   'Septoria_leaf_spot', 'Spider_mites_Two-spotted_spider_mite', 'Target_Spot', 'Tomato_mosaic_virus', 
                   'Tomato_Yellow_Leaf_Curl_Virus']
    
    num_classes = len(class_names)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # --- Chargement du modèle entraîné ---
    model = get_model(num_classes).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Modèle chargé depuis {model_path}")
    else:
        print(f"Erreur: Le fichier modèle {model_path} n'existe pas. Veuillez d'abord entraîner le modèle.")
        exit()

    # --- Prédiction sur l'image ---
    if os.path.exists(image_to_predict):
        predicted_class, confidence = predict_image(model, image_to_predict, class_names, device)
        print(f"Image : {image_to_predict}")
        print(f"Prédiction : {predicted_class}")
        print(f"Confiance : {confidence:.2%}")
    else:
        print(f"Erreur: Le fichier image {image_to_predict} n'existe pas.")