# ---------PREPROCESAMIENTO DE LAS IMÁGENES---------------
import cv2
import os
import numpy as np

input_dir = 'train\Malignant'  # Carpeta de imágenes originales
output_dir = 'procesadas\Malignant'  # Carpeta para guardar imágenes preprocesadas
input_dir_2 = 'train\Benign'  # Carpeta de imágenes originales
output_dir_2 = 'procesadas\Benign'  # Carpeta para guardar imágenes preprocesadas


def preprocesar_imagen(ruta, destino):
    # Crear carpeta de salida si no existe
    if not os.path.exists(destino):
        os.makedirs(destino)

    # Procesar todas las imágenes en el directorio
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
            #img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Paso 5: Guardar imagen preprocesada
            output_path = os.path.join(destino, filename)
            cv2.imwrite(output_path, img_normalizada)

            print(f"Imagen preprocesada guardada: {output_path}")

preprocesar_imagen(input_dir, output_dir)
preprocesar_imagen(input_dir_2, output_dir_2)