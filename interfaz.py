import os
from flask import Flask, render_template, request, redirect, url_for, jsonify
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import  transforms
from werkzeug.utils import secure_filename
from train_test import model
import torch.nn as nn
from gradcam import generate_gradcam, overlay_heatmap

# Cargar el modelo preentrenado
model.load_state_dict(torch.load('best_model.pth'))
model.eval()  # Poner el modelo en modo de evaluación  

# creo la carpeta uploads
if not os.path.exists('static'):
    os.makedirs('static')
    
# Inicializar la app Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

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
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = transform(image).unsqueeze(0)
    return image


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
            
            model.eval()
            
            #ponemos el modelo en evalucion
            outputs = model(image)

            # Aplicar sigmoide a la salida para obtener una probabilidad
            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).float()
            
            print("Probabilidad predicha:", probs.item())
            
            if predicted.item() == 1:
                result = 'Maligna (Melanoma)'
            elif predicted.item() == 0:
                result = 'Benigna'
            else: 
                result = "No lo sé pero soy coquette y me gusta el helado <3" 
            print(filename, result)
            
            # Generar el mapa de calor
            heatmap = generate_gradcam(image, model)
            heatmap_overlay = overlay_heatmap(heatmap, filepath)

            # Guardar el mapa de calor como imagen
            heatmap_path = os.path.join(app.config['UPLOAD_FOLDER'], 'heatmap_' + filename)
            plt.imshow(heatmap_overlay, cmap='hot', interpolation='nearest')
            plt.axis('off')
            plt.savefig(heatmap_path, bbox_inches='tight', pad_inches=0)
            
            # Renderizar el resultado y la imagen con el mapa de calor
            return render_template('result.html', image_url=url_for('static', filename=filename),
                                   heatmap_url=url_for('static', filename='heatmap_' + filename),
                                   diagnosis=result)

    return render_template('index.html')

# Iniciar la app
if __name__ == '__main__':
    app.run(debug=True)
