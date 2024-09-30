import torch
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.models as models

def tensor_to_heatmap(tensor, title='Tensor Heatmap'):
    # Ensure tensor is 2D for a heatmap
    if tensor.dim() == 3:
        # If it's a batch of 1 image or a single image with channels, we'll take the first channel or convert to grayscale
        if tensor.size(0) == 1:
            tensor = tensor.squeeze(0)
        elif tensor.size(0) == 3:  # Assuming RGB, convert to grayscale for heatmap
            tensor = 0.299 * tensor[0] + 0.587 * tensor[1] + 0.114 * tensor[2]
        else:
            raise ValueError("Tensor should have 1 or 3 channels for visualization as a heatmap.")
    
    # Convert to numpy for plotting
    heatmap_data = tensor.detach().cpu().numpy()
    
    # If values are not in a suitable range, normalize them
    if heatmap_data.max() > 1 or heatmap_data.min() < 0:
        heatmap_data = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, cmap="YlGnBu", annot=False, cbar=True)
    plt.title(title)
    plt.show()



# Load pre-trained VGG model
vgg = models.vgg16(pretrained=True).eval()  # or any VGG variant

def visualize_layer_outputs(input_tensor, layer_name='features'):
    
    # List to hold the outputs of the layers we're interested in
    layer_outputs = []
    
    def hook_fn(module, input, output):
        layer_outputs.append(output)

    # Find the convolutional layers and register hooks
    for name, module in vgg.named_modules():
        if 'features' in name:  # Assuming you want to hook into the feature extraction part (conv layers)
            module.register_forward_hook(hook_fn)
    
    # Run the input through the model
    with torch.no_grad():
        output = vgg(input_tensor)
    
    # Visualize the outputs
    idx, out = reversed(enumerate(layer_outputs))[0]
    # for idx, out in enumerate(layer_outputs):
        # Let's visualize the activations of the first 64 filters for simplicity
        # out will be of shape [batch_size, num_filters, height, width]
    fig, axarr = plt.subplots(8, 8, figsize=(20, 20))
    for i in range(64):  # Here we're only showing 64 filters, adjust as needed
        ax = axarr[i//8, i%8]
        ax.imshow(out[0, i].detach().cpu().numpy(), cmap='viridis')
        ax.axis('off')
    plt.suptitle(f"Activations of Layer {idx} in {layer_name}")
    plt.show()
    
    return layer_outputs  # In case you want to use these activations for further processing
