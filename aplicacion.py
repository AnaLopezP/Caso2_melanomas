import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import numpy as np
from train_test import model, device

# Definir las transformaciones para las imágenes de prueba
transform_test = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalización de las imágenes
])

# Cargar las imágenes de la carpeta procesadas_test
test_data = datasets.ImageFolder('procesadas_test', transform=transform_test)

# Crear un DataLoader para las imágenes de prueba
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Cargar el mejor modelo entrenado
model.load_state_dict(torch.load('best_model.pth'))
model.eval()  # Poner el modelo en modo de evaluación

# Lista para guardar las predicciones, probabilidades y nombres de archivos
all_preds = []
all_probs = []
image_paths = []

# Desactivar el cálculo de gradientes durante la predicción
with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(device)
        outputs = model(images)

        # Obtener probabilidades con sigmoid
        probs = torch.sigmoid(outputs)
        all_probs.extend(probs.cpu().numpy())  # Guardar probabilidades

        # Obtener predicciones (0 o 1) en función de un umbral de 0.5
        predicted = (probs > 0.5).float()
        all_preds.extend(predicted.cpu().numpy())  # Guardar predicciones

        # Guardar los nombres de los archivos para cada batch
        batch_image_paths = [os.path.basename(path[0]) for path in test_loader.dataset.samples]
        image_paths.extend(batch_image_paths)

# Convertir las listas a arrays numpy
all_probs = np.array(all_probs)
all_preds = np.array(all_preds)

# Mostrar los primeros 10 resultados
for i in range(10):
    print(f"Imagen: {image_paths[i]}, Predicción: {all_preds[i][0]}, Probabilidad: {all_probs[i][0]:.4f}")
