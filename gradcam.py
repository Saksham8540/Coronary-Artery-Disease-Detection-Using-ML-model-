import torch
import cv2
import numpy as np
from torchvision import models
import os

def generate_gradcam(model, input_tensor, save_path="gradcam_outputs/temp_gradcam.png"):
    model.eval()

    # Define hooks to capture gradients and activations
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    def forward_hook(module, input, output):
        activations.append(output.detach())

    # Register hooks to the last convolutional layer (ResNet example: layer4[1].conv2)
    target_layer = model.layer4[1].conv2
    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_backward_hook(backward_hook)

    # Forward pass
    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()
    class_score = output[0, pred_class]

    # Backward pass
    model.zero_grad()
    class_score.backward()

    # Remove hooks
    handle_fwd.remove()
    handle_bwd.remove()

    # Get stored activations and gradients
    gradients = gradients[0]  # shape: [1, C, H, W]
    activations = activations[0]  # shape: [1, C, H, W]

    # Global Average Pooling of gradients
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])  # shape: [C]

    # Weight the channels by corresponding gradients
    for i in range(activations.shape[1]):
        activations[:, i, :, :] = activations[:, i, :, :] * pooled_gradients[i]

    # Generate heatmap
    heatmap = torch.mean(activations, dim=1).squeeze().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-6  # normalize to avoid division by zero

    # Convert heatmap to color map
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Convert input image back to NumPy
    input_img = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    input_img = np.clip(input_img * 255, 0, 255).astype(np.uint8)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)

    # Overlay heatmap on input image
    overlay = cv2.addWeighted(input_img, 0.6, heatmap_color, 0.4, 0)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save and return
    cv2.imwrite(save_path, overlay)
    return save_path