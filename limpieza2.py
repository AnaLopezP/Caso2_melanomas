import os
import cv2
from PIL import Image
import numpy as np

# Función para limpiar la imagen usando un filtro Gaussiano
def limpiar_imagen(ruta_imagen):
    # Cargar la imagen
    img = cv2.imread(ruta_imagen)

    # Aplicar filtro Gaussiano para eliminar el ruido
    img_limpia = cv2.GaussianBlur(img, (5, 5), 0)
    
    return img_limpia

# Función para procesar y guardar las imágenes en la carpeta de destino
def procesar_imagenes(carpeta_origen, carpeta_destino):
    # Iterar sobre las subcarpetas (maligno y benigno)
    for clase in ['Malignant', 'Benign']:
        ruta_clase_origen = os.path.join(carpeta_origen, clase)
        ruta_clase_destino = os.path.join(carpeta_destino, clase)

        # Crear la carpeta de destino si no existe
        if not os.path.exists(ruta_clase_destino):
            os.makedirs(ruta_clase_destino)

        # Procesar cada imagen en la carpeta de clase
        for archivo in os.listdir(ruta_clase_origen):
            if archivo.endswith(('.jpg', '.jpeg', '.png')):
                ruta_imagen_origen = os.path.join(ruta_clase_origen, archivo)
                
                # Limpiar la imagen
                imagen_limpia = limpiar_imagen(ruta_imagen_origen)
                
                # Guardar la imagen limpia en la carpeta de destino
                ruta_imagen_destino = os.path.join(ruta_clase_destino, archivo)
                cv2.imwrite(ruta_imagen_destino, imagen_limpia)

                print(f"Imagen {archivo} procesada y guardada en {ruta_imagen_destino}")

# Rutas a las carpetas de origen y destino
carpeta_origen = 'test'  # Carpeta original con imágenes
carpeta_destino = 'procesado_test'  # Carpeta destino donde se guardarán las imágenes procesadas

# Procesar las imágenes y guardarlas
procesar_imagenes(carpeta_origen, carpeta_destino)
