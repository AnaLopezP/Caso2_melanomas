import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

class SaveFeatures:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = output
    def remove(self):
        self.hook.remove()

def generate_gradcam(image_tensor, model, final_conv_layer_name='conv4'):
    model.eval()

    # Guardar los gradientes
    final_conv_layer = dict([*model.named_modules()])[final_conv_layer_name]
    features = SaveFeatures(final_conv_layer)

    output = model(image_tensor)
    output = torch.sigmoid(output)
    
    pred_class = output.argmax(dim=1)
    
    # Retropropagación del gradiente
    model.zero_grad()
    output[:, pred_class].backward()

    gradients = final_conv_layer.weight.grad.data
    activations = features.features.detach()

    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=1).squeeze()

    heatmap = np.maximum(heatmap.cpu(), 0)
    heatmap /= torch.max(heatmap)

    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, (image_tensor.shape[2], image_tensor.shape[3]))

    return heatmap

def apply_heatmap(image_path, heatmap):
    img = cv2.imread(image_path)
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img
    return superimposed_img
