import numpy as np
from sklearn.metrics import mean_squared_error
import torch
import torch.nn.functional as F

# Define a function to register forward hooks for feature extraction
def forward_hook_registration(model, total_layers, layers, outputs):

    # Define a function to append the output of each layer to the outputs list
    def append_output(module, input, output):
        outputs.append(output)

    # Create a list to store the hook handles
    hd = [None] * total_layers
    i = 0

    # Register forward hooks for each layer
    for layer_type, layer_indices in layers.items():
        for layer in layer_indices:
            if layer_type == 'conv':
                hd[i] = model.features[layer][0].register_forward_hook(append_output)
                i += 1
            elif layer_type == 'fc':
                hd[i] = model.classifier[layer].register_forward_hook(append_output)
                i += 1
    return hd, outputs

# Define a function to remove the registered hooks
def remove_hooks(hd, total_layers):
    for i in range(total_layers):
        hd[i].remove()

def pad_to_desired_size(original_tensor, tensor_to_pad):
    desired_size = original_tensor.size(2)
    # Calculate the amount of padding needed for each dimension
    padding_needed = desired_size - tensor_to_pad.size(2)
    mean_value = torch.mean(original_tensor)
    # Pad the tensor with zeros
    padded_tensor = F.pad(tensor_to_pad, (0, padding_needed, 0, padding_needed), mode='constant', value=mean_value)

    return padded_tensor
