import torch
from torch import nn
import torch_geometric.utils as utils
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from PIL import Image
from torchvision import transforms
from torch_geometric.data import Data
import numpy as np
import matplotlib.pyplot as plt
import torch_geometric
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        print(x.shape)
        x = x.view(-1, 3)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.relu(x)


def vit(input_dim, num_classes):
    model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    model.conv_proj = nn.Conv2d(input_dim, 768, kernel_size=(16, 16), stride=(16, 16))
    model_num_features = model.num_classes
    model.heads = nn.Sequential(
            nn.Linear(model_num_features, num_classes),
            nn.Softmax()
        )
    return model


class GViT(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(GViT, self).__init__()
        self.gnn = GCN(input_dim, hidden_dim1, hidden_dim2)
        self.transformer = vit(hidden_dim2, output_dim)

    def forward(self, x, edge_index):
        graphs = self.gnn(x, edge_index)
        width, height = 224, 224
        reshaped_graphs = graphs.view(height, width)
        output = self.transformer(reshaped_graphs)
        return output


if __name__ == '__main__':
    path = '/home/roblab20/PycharmProjects/LongRange/4.png'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    img = Image.open(path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    x = transform(img)
    model = GCN(3, 32, 1)
    model.to(device)

    image_width = 224
    image_height = 224


    def get_index(i, j):
        return j * image_width + i

    edge_index = []
    for i in range(image_width):
        for j in range(image_height):
            current_index = get_index(i, j)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    if 0 <= i + dx < image_width and 0 <= j + dy < image_height:
                        edge_index.append([current_index, get_index(i + dx, j + dy)])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t()

    # print('ok')
    result = model(x.to(device), edge_index.to(device))
    print('\n\n', result.shape)

    # Review results
    width, height = 224, 224
    reshaped_tensor = result.view(height, width)
    # Scale the pixel values to the range [0, 255]
    scaled_tensor = (reshaped_tensor * 255).byte()

    # Create a PIL Image from the tensor
    gray_image = Image.fromarray(scaled_tensor.cpu().numpy(), mode='L')
    plt.imshow(gray_image, cmap='gray')
    plt.axis('off')  # Turn off the axes
    plt.show()

    # print('\n\n', result)
    # vit(1, 6)