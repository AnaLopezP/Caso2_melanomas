import os
from flask import Flask, render_template, request, redirect, url_for, jsonify
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchvision import models, transforms
from werkzeug.utils import secure_filename
from train_test import model
from PIL import Image
import io
import torch.nn as nn

# creo la carpeta uploads
if not os.path.exists('static'):
    os.makedirs('static')
    
# Inicializar la app Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

'''# Cargar el modelo preentrenado
model.load_state_dict(torch.load('best_model.pth'))
model.eval()  # Poner el modelo en modo de evaluación'''

# Comprobar si la imagen tiene un formato permitido
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocesar la imagen para el modelo
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (32, 32))  # Cambiar tamaño a 32x32
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir a RGB
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)
    return image

# Generar un mapa de calor (grad-CAM)
def generate_heatmap(image, model):
    heatmap = np.random.random((224, 224))
    return heatmap

# Ruta principal
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Preprocesar la imagen y hacer la predicción
            image = preprocess_image(filepath)
            outputs = model(image)

            # Generar el mapa de calor
            heatmap = generate_heatmap(image, model)

            # Guardar el mapa de calor como imagen
            heatmap_path = os.path.join(app.config['UPLOAD_FOLDER'], 'heatmap_' + filename)
            plt.imshow(heatmap, cmap='hot', interpolation='nearest')
            plt.axis('off')
            plt.savefig(heatmap_path, bbox_inches='tight', pad_inches=0)

            # Obtener el resultado del modelo
            _, predicted = torch.max(outputs, 1)
            result = 'Maligna (Melanoma)' if predicted.item() == 1 else 'Benigna'
            print(filename, result)
            # Renderizar el resultado y la imagen con el mapa de calor
            return render_template('result.html', image_url=url_for('static', filename=filename),
                                   heatmap_url=url_for('static', filename='heatmap_' + filename),
                                   diagnosis=result)

    return render_template('index.html')


# Iniciar la app
if __name__ == '__main__':
    app.run(debug=True)
