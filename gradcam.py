import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

def generate_gradcam(image, model, target_class=None):
    # Poner el modelo en modo de evaluación
    model.eval()
    
    gradients = None  # Variable para almacenar los gradientes
    
    # Hacer que los gradientes de la última capa convolucional sean accesibles
    def extract_gradients(module, gran_input, grad_output):
        nonlocal gradients
        gradients = grad_output
    
    conv_output = None  # Variable para almacenar las activaciones
    def extract_activations(module, input, output):
        nonlocal conv_output
        conv_output = output

    # Registrar los hooks
    model.conv4.register_forward_hook(extract_activations)
    model.conv4.register_full_backward_hook(extract_gradients)
    
    # Pasar la imagen por el modelo
    output = model(image)

    # Si no se especifica una clase objetivo, usaremos la predicción más alta
    if target_class is None:
        target_class = torch.argmax(output)
    
    # Backward pass para obtener los gradientes de la clase objetivo
    model.zero_grad()
    class_loss = output[:, target_class]
    class_loss.backward()
    
    # Obtener las activaciones de las características y los gradientes
    pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])
    conv_output = conv_output.detach()

    # Ponderar las activaciones con los gradientes obtenidos
    for i in range(conv_output.size(1)):
        conv_output[:, i, :, :] *= pooled_gradients[i]
    
    # Obtener la media de las activaciones a través del canal
    heatmap = torch.mean(conv_output, dim=1).squeeze()
    heatmap = np.maximum(heatmap.cpu(), 0)  # ReLU en el heatmap
    heatmap /= torch.max(heatmap)  # Normalizar entre 0 y 1

    return heatmap.numpy()

def overlay_heatmap(heatmap, image_path, alpha=0.5, colormap=cv2.COLORMAP_JET):
    # Cargar la imagen original
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Redimensionar el mapa de calor al tamaño de la imagen
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    
    # Aplicar el colormap al heatmap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), colormap)
    
    # Superponer el mapa de calor con la imagen original
    superimposed_img = cv2.addWeighted(heatmap_colored, alpha, original_image, 1 - alpha, 0)
    
    return superimposed_img
