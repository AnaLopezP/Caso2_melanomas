# ---------PREPROCESAMIENTO DE LAS IMÁGENES---------------
import cv2
import os
import numpy as np

input_dir = 'train\Malignant'  # Carpeta de imágenes originales
input_dir_2 = 'train\Benign'  # Carpeta de imágenes originales
output_dir = 'procesadas\Malignant'  # Carpeta para guardar imágenes preprocesadas
output_dir_2 = 'procesadas\Benign'  # Carpeta para guardar imágenes preprocesadas

# Crear carpeta de salida si no existe
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Procesar todas las imágenes en el directorio
def procesar_img(ruta, carpeta_destino):
    for filename in os.listdir(ruta):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # Extensiones comunes de imágenes
            # Cargar imagen
            img_path = os.path.join(ruta, filename)
            img = cv2.imread(img_path)

            # Verificar que la imagen se haya cargado correctamente
            if img is None:
                print(f"No se pudo cargar la imagen: {filename}")
                continue

            # Paso 1: Filtro Gaussiano (eliminar ruido)
            img_filtrada = cv2.GaussianBlur(img, (5, 5), 0)

            # Paso 2: Redimensionar la imagen a 224x224
            img_redimensionada = cv2.resize(img_filtrada, (224, 224))

            # Paso 3: Normalizar los píxeles (dividir entre 255)
            img_normalizada = img_redimensionada / 255.0
            
            # 4. Convertir a escala de grises
            #img_gris = cv2.cvtColor(img_normalizada, cv2.COLOR_BGR2GRAY)

            # Paso 5: Guardar imagen preprocesada
            carpeta_destino = os.path.join(output_dir, filename)
            cv2.imwrite(carpeta_destino, img_normalizada)

            print(f"Imagen preprocesada guardada: {carpeta_destino}")

procesar_img(input_dir, output_dir)
procesar_img(input_dir_2, output_dir_2)