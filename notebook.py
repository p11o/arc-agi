import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt

# Define the device to run on (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device {device}")
# Load the VGG19 model, we'll use features from this network for NST
vgg19 = models.vgg19(pretrained=True).features.to(device).eval()

# Freeze all VGG parameters since we're only optimizing the input image
for param in vgg19.parameters():
    param.requires_grad_(False)

# Image preprocessing
def load_image(img_path, max_size=400, shape=None):
    image = Image.open(img_path).convert('RGB')
    
    # Resize the image keeping aspect ratio
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    
    if shape is not None:
        size = shape
        
    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])

    image = in_transform(image)[:3,:,:].unsqueeze(0)
    return image.to(device)

# Load images
content_img = load_image('image/city.jpeg')
style_img = load_image('image/large-brush.jpeg')


def denormalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    for t, m, s in zip(tensor, mean, std):
        t = t.mul(s).add(m)
    return t

# Assuming you have the tensor of the image you want to visualize
# img_tensor = model_input_img # Your normalized tensor
# mean = (0.485, 0.456, 0.406)
# std = (0.229, 0.224, 0.225)
# denormalized_img = denormalize(img_tensor, mean, std)

# Function to convert tensor to image for visualization
def imshow(tensor, title=None):
    tensor = denormalize(tensor.clone())
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

# Display images
plt.figure()
imshow(style_img, title='Style Image')

plt.figure()
imshow(content_img, title='Content Image')

# Content and Style Losses
class ContentLoss(nn.Module):
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = self.gram_matrix(target_feature).detach()

    def gram_matrix(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        features = input.view(a * b, c * d)
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)

    def forward(self, input):
        G = self.gram_matrix(input)
        self.loss = nn.functional.mse_loss(G, self.target)
        return input

# Create model from VGG
def get_style_model_and_losses(vgg, content_img, style_img):
    content_layers = ['conv_4', 'conv_5']
    style_layers = ['conv_1', 'conv_2', 'conv_3', ]
    
    cnn = vgg
    model = nn.Sequential()
    content_losses = []
    style_losses = []

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # Trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

# Set up optimization
input_img = content_img.clone()

model, style_losses, content_losses = get_style_model_and_losses(vgg19, content_img, style_img)

# Training Loop
def run_style_transfer(content_img, style_img, input_img, model, style_losses, content_losses, num_steps=500,
                       style_weight=10000, content_weight=1):
    print('Building the style transfer model..')
    optimizer = optim.LBFGS([input_img.requires_grad_()], lr=0.005)
    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # Correct the values of updated input image
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print(f"run {run}:")
                print(f'Style Loss : {style_score.item():4f} Content Loss: {content_score.item():4f}')
                print()

            return style_score + content_score

        optimizer.step(closure)

    # A last correction...
    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img

# Now, run the style transfer
output = run_style_transfer(content_img, style_img, input_img, model, style_losses, content_losses)

# Visualize the result
plt.figure()
imshow(output, title='Output Image')

# If you want to save the result:
output_img = output.squeeze(0).cpu()
output_img = transforms.ToPILImage()(output_img)
output_img.save('style_transfer_result.jpg')

plt.ioff()
plt.show()
