import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Transformaciones: redimensionar las imágenes y normalizarlas
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Cambia el tamaño de las imágenes
    transforms.ToTensor(),  # Convierte la imagen a un tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normaliza los canales RGB
])

# Cargar el dataset desde carpetas
train_data = datasets.ImageFolder('train', transform=transform)
test_data = datasets.ImageFolder('test', transform=transform)

# Crear DataLoaders para cargar los datos en lotes
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Definir las capas convolucionales
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # 3 canales (RGB), 32 filtros, kernel 3x3
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        # Max Pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected (FC) layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)  # Ajusta según el tamaño final de tus imágenes
        self.fc2 = nn.Linear(256, 10)  # 10 clases de salida

        # Función de activación
        self.relu = nn.ReLU()

    def forward(self, x):
        # Pasar las imágenes a través de las capas convolucionales y de pooling
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))

        # Aplanar el tensor
        x = x.view(-1, 128 * 4 * 4)

        # Pasar por las capas fully connected
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

# Crear una instancia de la CNN
model = CNN()

# Definir la función de pérdida y el optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entrenar el modelo
epochs = 10
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        # Limpiar los gradientes del optimizador
        optimizer.zero_grad()

        # Forward pass (pasa las imágenes a través de la red)
        outputs = model(images)
        
        # Calcular la pérdida
        loss = criterion(outputs, labels)
        
        # Backward pass (retropropagación)
        loss.backward()
        
        # Actualizar los parámetros
        optimizer.step()

        # Acumular la pérdida
        running_loss += loss.item()

    print(f"Época [{epoch+1}/{epochs}], Pérdida: {running_loss / len(train_loader)}")

print("Entrenamiento completado")


correct = 0
total = 0

with torch.no_grad():  # No necesitamos calcular gradientes para la evaluación
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)  # Obtener la clase con mayor puntuación
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Precisión en el conjunto de prueba: {100 * correct / total}%")
