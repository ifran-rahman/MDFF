import numpy as np
import torch
import torch.nn.functional as F


# Define a function to register forward hooks for feature extraction
def forward_hook_registration(model, layers, outputs):
    """Register forward hooks that append each selected layer's output to `outputs`.

    layers: {'conv': [i, ...]} hooks model.features[i][0];
            {'fc': [j, ...]}   hooks model.classifier[j].
    """

    def append_output(module, input, output):
        outputs.append(output)

    handles = []
    for layer_type, layer_indices in layers.items():
        for layer in layer_indices:
            if layer_type == 'conv':
                handles.append(model.features[layer][0].register_forward_hook(append_output))
            elif layer_type == 'fc':
                handles.append(model.classifier[layer].register_forward_hook(append_output))
    return handles


# Define a function to remove the registered hooks
def remove_hooks(handles):
    for h in handles:
        h.remove()


def hooked_feature(model, layers):
    """Return f(x) -> output of the single hooked layer in `layers`."""
    def get_feature(x):
        outputs = []
        handles = forward_hook_registration(model, layers, outputs)
        model(x)
        remove_hooks(handles)
        return outputs[0]
    return get_feature


def pad_to_desired_size(original_tensor, tensor_to_pad):
    """Centre-pad `tensor_to_pad` to the spatial size of `original_tensor`.

    The padded border of each sample is filled with that sample's own mean of
    `original_tensor`, so a sample's features do not depend on the rest of the batch.
    """
    diff_h = original_tensor.size(2) - tensor_to_pad.size(2)
    diff_w = original_tensor.size(3) - tensor_to_pad.size(3)
    if diff_h < 0 or diff_w < 0:
        raise ValueError("tensor_to_pad has larger dimensions than original_tensor.")

    pad_top = diff_h // 2
    pad_left = diff_w // 2
    pads = (pad_left, diff_w - pad_left, pad_top, diff_h - pad_top)  # (left, right, top, bottom)

    mean_value = original_tensor.mean(dim=(1, 2, 3)).view(-1, 1, 1, 1)
    padded = F.pad(tensor_to_pad, pads, mode="constant", value=0.0)
    border = 1 - F.pad(torch.ones_like(tensor_to_pad[:, :1]), pads, mode="constant", value=0.0)
    return padded + border * mean_value


@torch.no_grad()
def extract_features(get_feature, loader, device, num_scales=1, scale_factor=2):
    """Run a frozen backbone over `loader` and return (N, D + 1) features with the target last.

    num_scales=1: single-scale features (Conv / Dense baselines).
    num_scales>1: MDFF - the image is repeatedly upscaled with bilinear interpolation,
    features are taken at every scale, the smaller maps are centre-padded to the
    largest one and all maps are fused by depth (channel) concatenation.
    """
    rows = []
    for image_tensor, targets in loader:
        image_list = [image_tensor]
        for _ in range(num_scales - 1):
            image_tensor = F.interpolate(image_tensor, scale_factor=scale_factor,
                                         mode='bilinear', align_corners=False)
            image_list.append(image_tensor)

        # Largest scale first, so its feature map sets the target size.
        features = [get_feature(x.to(device)) for x in image_list[::-1]]
        reference = features[0]
        if len(features) > 1:
            reference = torch.cat([reference] + [pad_to_desired_size(reference, f) for f in features[1:]], dim=1)
        flat = reference.reshape(len(targets), -1)

        rows.append(np.concatenate((flat.cpu().numpy(), targets.unsqueeze(1).numpy()), axis=1))
    return np.vstack(rows)
