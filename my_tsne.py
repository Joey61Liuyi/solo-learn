# -*- coding: utf-8 -*-
# @Time    : 2023/3/16 22:35
# @Author  : LIU YI

import copy
import random
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import clip
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.manifold import TSNE
from collections import defaultdict
import torch.nn.functional as F
from torch.utils.data import TensorDataset
import math

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    result = out.reshape(b, *((1,) * (len(x_shape) - 1)))
    return result

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def inference(loader, model, device):
    feature_vector = []
    labels_vector = []
    for step, (x, y) in enumerate(loader):

        x = x.to(device)
        h = model(x)
        h = h.detach()
        # similarity = torch.matmul(h, h.t())
        # similarity /= torch.norm(h, dim=1)[:, None]
        # similarity /= torch.norm(h, dim=1)[None, :]
        #
        # # Plot heatmap
        # fig, ax = plt.subplots()
        # heatmap = ax.imshow(similarity.cpu().numpy(), cmap='Blues')
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_title('Cosine Similarity')
        # cbar = fig.colorbar(heatmap, ax=ax)
        # plt.savefig('random_input.png')
        # plt.show()
        h = h.squeeze()
        feature_vector.extend(h.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

        if step % 5 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        timesteps = 1000,
        beta_schedule = 'cosine',
    ):
        super().__init__()

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps)


        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas.to('cuda'))
        register_buffer('alphas_cumprod', alphas_cumprod.to('cuda'))
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev.to('cuda'))

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod).to('cuda'))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod).to('cuda'))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod).to('cuda'))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod).to('cuda'))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1).to('cuda'))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance.to('cuda'))

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)).to('cuda'))
        register_buffer('posterior_mean_coef1', (betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)).to('cuda'))
        register_buffer('posterior_mean_coef2', ((1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)).to('cuda'))

        # calculate p2 reweighting

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )


class MiniImageNetDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_label = None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.images = []
        self.labels = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if target_label == None:
                for image_name in os.listdir(class_dir):
                    image_path = os.path.join(class_dir, image_name)
                    self.images.append(image_path)
                    self.labels.append(self.class_to_idx[class_name])
            else:
                if self.class_to_idx[class_name] == target_label:
                    for image_name in os.listdir(class_dir):
                        image_path = os.path.join(class_dir, image_name)
                        self.images.append(image_path)
                        self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        label = self.labels[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def tSNE(h, label, name):
    color_dict = {0: 'blue', 1: 'green', 2: 'yellow', 3: 'red', 4: 'brown', 5: 'gray', 6: 'gold', 7: 'pink', 8: 'purple',
     9: 'orange', 10: 'black'}
    # color_dict = {'red': 0,  'black': 1}
    # color_dict = {}

    # for i in range(5):
    #     # Define the color in RGB format
    #     red = int(255 - (i * 255/4))  # gradually decrease red channel from 255 to 0
    #     green = 0  # no green channel
    #     blue = 0
    #     color = (red, green, blue)
    #     # Convert the color from RGB to hex format
    #     hex_color = '#{:02x}{:02x}{:02x}'.format(*color)
    #     # Add the color to the dictionary
    #     color_dict[i] = hex_color
    #
    # for i in range(5):
    #     # Define the color in RGB format
    #     red =  0# gradually decrease red channel from 255 to 0
    #     green = int(255 - (i * 255/4))   # no green channel
    #     blue = 0
    #     color = (red, green, blue)
    #     # Convert the color from RGB to hex format
    #     hex_color = '#{:02x}{:02x}{:02x}'.format(*color)
    #     # Add the color to the dictionary
    #     color_dict[i+5] = hex_color
    #
    # for i in range(5):
    #     # Define the color in RGB format
    #     red =  0# gradually decrease red channel from 255 to 0
    #     green = 0   # no green channel
    #     blue = int(255 - (i * 255/4))
    #     color = (red, green, blue)
    #     # Convert the color from RGB to hex format
    #     hex_color = '#{:02x}{:02x}{:02x}'.format(*color)
    #     # Add the color to the dictionary
    #     color_dict[i+10] = hex_color
    #
    #
    # for i in range(5):
    #     # Define the color in RGB format
    #     red =  0# gradually decrease red channel from 255 to 0
    #     green = int(255 - (i * 255/4)) # no green channel
    #     blue = int(255 - (i * 255/4))
    #     color = (red, green, blue)
    #     # Convert the color from RGB to hex format
    #     hex_color = '#{:02x}{:02x}{:02x}'.format(*color)
    #     # Add the color to the dictionary
    #     color_dict[i+15] = hex_color
    #
    # for i in range(5):
    #     # Define the color in RGB format
    #     red =  int(255 - (i * 255/4))# gradually decrease red channel from 255 to 0
    #     green = 0 # no green channel
    #     blue = int(255 - (i * 255/4))
    #     color = (red, green, blue)
    #     # Convert the color from RGB to hex format
    #     hex_color = '#{:02x}{:02x}{:02x}'.format(*color)
    #     # Add the color to the dictionary
    #     color_dict[i+20] = hex_color
    #
    # for i in range(5):
    #     # Define the color in RGB format
    #     red =  int(255 - (i * 255/4)) # gradually decrease red channel from 255 to 0
    #     green = int(255 - (i * 255/4)) # no green channel
    #     blue = 0
    #     color = (red, green, blue)
    #     # Convert the color from RGB to hex format
    #     hex_color = '#{:02x}{:02x}{:02x}'.format(*color)
    #     # Add the color to the dictionary
    #     color_dict[i+25] = hex_color


    data = np.array(h)
    target = np.array(label)
    tsne = TSNE(n_components=2, n_iter=500)

    data_tsne = tsne.fit_transform(data)
    x, y = data_tsne[:, 0], data_tsne[:, 1]

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    plt.xlim(-50, 50)
    plt.ylim(-50, 50)
    # current_axes = plt.axes()
    # current_axes.xaxis.set_visible(False)
    # current_axes.yaxis.set_visible(False)

    color_target = [color_dict[c] for c in target]
    plt.scatter(x, y, c=color_target, s=10)
    plt.xticks(fontsize=14, rotation=0)
    plt.yticks(fontsize=14, rotation=0)
    plt.savefig(name, dpi=1000)
    plt.show()


# Create the data loader

def test_result(test_loader, logreg, device):
    # Test fine-tuned model
    print("### Calculating final testing performance ###")
    logreg.eval()
    metrics = defaultdict(list)
    for step, (h, y) in enumerate(test_loader):
        h = h.to(device)
        y = y.to(device)

        outputs = logreg(h)

        # calculate accuracy and save metrics
        accuracy = (outputs.argmax(1) == y).sum().item() / y.size(0)
        metrics["Accuracy/test"].append(accuracy)

    for k, v in metrics.items():
        print(f"{k}: {np.array(v).mean():.4f}")
    return np.array(metrics["Accuracy/test"]).mean()


def test_down_stream_acc(train_images, train_labels, test_images, test_labels):

    # _, preprocess = clip.load("ViT-B/32", device='cuda')
    #
    #
    # # model, preprocess = clip.load("ViT-B/32", device='cuda')
    # mini_imagenet_dataset = MiniImageNetDataset(root_dir='./mini_imagenet', transform=preprocess)
    # mini_imagenet_dataloader = DataLoader(mini_imagenet_dataset, batch_size=500, shuffle=True, num_workers=0)
    # features, labels = inference(mini_imagenet_dataloader, resnet, 'cuda')
    # features = np.load('feature_clip_MI.npy')
    # labels = np.load('label_clip_MI.npy')

    # test_size = int(len(features)/6)
    # test_images, test_labels = features[:test_size], labels[:test_size]
    # train_images, train_labels = features[test_size:], labels[test_size:]



    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)

    train_dataloader = DataLoader(train_dataset, batch_size= 256, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    logreg = nn.Sequential(nn.Linear(512, 100))
    logreg = logreg.to('cuda')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=logreg.parameters(), lr=3e-3)

    logreg.train()
    for epoch in range(200):
        metrics = defaultdict(list)
        for step, (h, y) in enumerate(train_dataloader):

            h = h.to('cuda')
            y = y.to('cuda')

            outputs = logreg(h)

            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculate accuracy and save metrics
            accuracy = (outputs.argmax(1) == y).sum().item() / y.size(0)
            metrics["Loss/train"].append(loss.item())
            metrics["Accuracy/train"].append(accuracy)

        print(f"Epoch [{epoch}/{200}]: " + "\t".join(
            [f"{k}: {np.array(v).mean()}" for k, v in metrics.items()]))

        if epoch % 100 == 0:
            print("======epoch {}======".format(epoch))
            test_result(test_dataloader, logreg, 'cuda')

    result = test_result(test_dataloader, logreg, 'cuda')
    print(result)


def uniformity_test(images):
    # train_images = torch.load(method+'_'+dataset+'_train_feature.pt')
    # train_labels = torch.load(method+'_'+dataset+'_train_label.pt')
    # images = torch.load('noise_'+method+'_'+dataset+'_eval_feature.pt')
    print(uniform_loss(images.to('cuda')))


def draw_test_plot(clip_symble = True):


    method = 'simclr'
    dataset = 'cifar10'

    # train_images = torch.load(method+'_'+dataset+'_train_feature.pt')
    # train_labels = torch.load(method+'_'+dataset+'_train_label.pt')
    images = torch.load(method+'_'+dataset+'_eval_feature.pt')

    labels = torch.load(method+'_'+dataset+'_eval_label.pt')

    unique_labels = sorted(torch.unique(labels))
    unique_labels = [x for x in unique_labels if x < 10]

    # Create a new data tensor and label tensor to hold the reduced data
    reduced_data = torch.zeros((len(unique_labels) * 50, images.shape[1]))
    reduced_labels = torch.zeros(len(unique_labels) * 50)

    # Iterate over the unique labels and sample 50 samples each for labels in the range 0-9
    for i, label in enumerate(unique_labels):
        label_indices = torch.where(labels == label)[0]
        sample_indices = torch.randperm(len(label_indices))[:50]
        reduced_data[i * 50:(i + 1) * 50] = images[label_indices[sample_indices]]
        reduced_labels[i * 50:(i + 1) * 50] = label


    images = reduced_data.numpy()
    labels = reduced_labels.numpy()

    # images = torch.tensor(images)
    tSNE(images, labels, 'tep.png')
    center = np.mean(images, axis=0)
    distances = np.linalg.norm(images - center, axis=1)
    mean = np.mean(distances)
    std = np.std(distances)
    # radius = torch.max(distances)
    print(f'mean {mean}, std {std}')




def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()




if __name__ == '__main__':
    seed = 61

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


    prefix = 'noise_only_'
    method = 'byol'
    dataset = 'cifar100'

    train_images = torch.load(prefix+method+'_'+dataset+'_train_feature.pt')
    train_labels = torch.load(prefix+method+'_'+dataset+'_train_label.pt')
    test_images = torch.load(prefix+method+'_'+dataset+'_eval_feature.pt')
    test_labels = torch.load(prefix+method+'_'+dataset+'_eval_label.pt')

    test_down_stream_acc(train_images, train_labels, test_images, test_labels)
    # draw_test_plot(False)
    # draw_heat_map(True)
    uniformity_test(test_images)