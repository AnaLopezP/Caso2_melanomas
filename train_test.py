import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt


# Revisar, está haciendo del 0.5 para abajo bien, pero las prob altas no. 
# Tocar el modelo para que esto no pase y siga clasificando bien

# Transformaciones de datos con data augmentation
transform_train = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),  # Agregar reflejo horizontal
    transforms.RandomRotation(10),      # Rotación aleatoria hasta 10 grados
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),  # Deformación aleatoria
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Variación de color
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalización estándar
])

transform_test = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # No aplicar augmentación a los datos de prueba
])
# Cargar el dataset desde carpetas
train_data = datasets.ImageFolder('procesadas', transform=transform_train)
test_data = datasets.ImageFolder('procesadas_test', transform=transform_test)

# Crear DataLoaders para cargar los datos en lotes
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Definir capas convolucionales con BatchNorm y más filtros
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Max Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers con Dropout
        self.fc1 = nn.Linear(256 * 2 * 2, 512)
        self.dropout = nn.Dropout(0.5)  # Dropout para prevenir sobreajuste
        self.fc2 = nn.Linear(512, 1)  # Salida para 2 clases (maligno/benigno)

        # Función de activación
        self.relu = nn.ReLU()

    def forward(self, x):
        # Capas convolucionales con BatchNorm y activación
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        
        # Aplanar el tensor
        x = x.view(-1, 256 * 2 * 2) 
        # Fully connected layers con Dropout
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# === Definir la función de pérdida (Focal Loss) ===
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, inputs, targets):
        BCE_loss = nn.BCEWithLogitsLoss()(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss

criterion = FocalLoss()

# Crear una instancia de la CNN
model = CNN().to(device)

# === Optimizer SGD con momentum y decay ===
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)

# Scheduler para reducir la tasa de aprendizaje dinámicamente
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


train_losses = []
train_accuracies = []

# Entrenar el modelo
epochs = 30
best_acc = 0.0
for epoch in range(epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    model.train()  # Poner el modelo en modo de entrenamiento
    
    for images, labels in train_loader:
        
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
        # Limpiar gradientes
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        
        # Calcular pérdida
        loss = criterion(outputs, labels)
        
        # Backward pass (retropropagación)
        loss.backward()
        
        # Actualizar parámetros
        optimizer.step()

        # Acumular la pérdida
        running_loss += loss.item()
        
        # Calcular precisión
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Ajustar la tasa de aprendizaje con el scheduler
    scheduler.step()
    
    # Guardar los resultados de la época
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total

    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

    # Guardar el mejor modelo
    if epoch_accuracy > best_acc:
        best_acc = epoch_accuracy
        torch.save(model.state_dict(), 'best_model.pth')

    print(f"Época [{epoch+1}/{epochs}], Pérdida: {running_loss / len(train_loader)}")

print("Entrenamiento completado")

# Evaluar el modelo
correct = 0
total = 0

all_labels = []
all_probs = []


model.eval()  # Poner el modelo en modo de evaluación

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
        outputs = model(images)
        
        # Obtener probabilidades con sigmoid
        probs = torch.sigmoid(outputs)
        all_probs.extend(probs.cpu().numpy())
        
        predicted = (probs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Guardar etiquetas reales
        all_labels.extend(labels.cpu().numpy())


print(f"Precisión en el conjunto de prueba: {100 * correct / total}%")



# Graficar la curva de pérdida (Loss)
plt.figure(figsize=(10,5))
plt.plot(train_losses, label='Pérdida de entrenamiento')
plt.title('Curva de Pérdida')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.show()

# Graficar la curva de precisión (Accuracy)
plt.figure(figsize=(10,5))
plt.plot(train_accuracies, label='Precisión de entrenamiento')
plt.title('Curva de Precisión')
plt.xlabel('Épocas')
plt.ylabel('Precisión (%)')
plt.legend()
plt.show()


# Convertir las listas de etiquetas y probabilidades a arrays numpy
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

# Calcular AUC-ROC
auc = roc_auc_score(all_labels, all_probs)
print(f"AUC-ROC: {auc:.2f}")

# Calcular la curva ROC
fpr, tpr, thresholds = roc_curve(all_labels, all_probs)

# Graficar la curva ROC
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'Curva ROC (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Línea diagonal de referencia
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de falsos positivos (FPR)')
plt.ylabel('Tasa de verdaderos positivos (TPR)')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()



'''# Inicializamos variables
all_labels = []
all_predictions = []

# Desactivar la gradiente para evaluación
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        # Guardar las etiquetas y las predicciones
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

# Convertimos a numpy arrays
all_labels = np.array(all_labels)
all_predictions = np.array(all_predictions)

# Calcular la matriz de confusión
conf_matrix = confusion_matrix(all_labels, all_predictions)

# Extraer los valores de la matriz de confusión
TN, FP, FN, TP = conf_matrix.ravel()

# Calcular precisión, sensibilidad y especificidad
precision = TP / (TP + FP)
sensibilidad = TP / (TP + FN)  # También llamada "recall"
especificidad = TN / (TN + FP)

# Imprimir las métricas
print(f"Precisión: {precision:.2f}")
print(f"Sensibilidad (Recall): {sensibilidad:.2f}")
print(f"Especificidad: {especificidad:.2f}")

# Inicializamos los valores predichos y las etiquetas reales
all_probs = []
all_labels = []

# Evaluar el modelo y obtener las probabilidades
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)  # Obtener probabilidades con softmax
        all_probs.extend(probs[:, 1].cpu().numpy())  # Probabilidad de la clase 1 (positiva)
        all_labels.extend(labels.cpu().numpy())

# Convertimos a arrays numpy
all_probs = np.array(all_probs)
all_labels = np.array(all_labels)

# Calcular AUC
auc = roc_auc_score(all_labels, all_probs)
print(f"AUC-ROC: {auc:.2f}")

# Calcular la curva ROC
fpr, tpr, thresholds = roc_curve(all_labels, all_probs)

# Si quieres graficar la curva ROC
import matplotlib.pyplot as plt

plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de falsos positivos (FPR)')
plt.ylabel('Tasa de verdaderos positivos (TPR)')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()

# Graficar la curva de pérdida (Loss)
plt.figure(figsize=(10,5))
plt.plot(train_losses, label='Pérdida de entrenamiento')
plt.title('Curva de Pérdida')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.show()

# Graficar la curva de precisión (Accuracy)
plt.figure(figsize=(10,5))
plt.plot(train_accuracies, label='Precisión de entrenamiento')
plt.title('Curva de Precisión')
plt.xlabel('Épocas')
plt.ylabel('Precisión (%)')
plt.legend()
plt.show()'''
