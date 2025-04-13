import h5py
import random
import pandas as pd

from sklearn.manifold import TSNE

import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm.notebook import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

SEED = 0

torch.random.manual_seed(SEED)
random.seed(SEED)
##
## ------------------------------------------------------------------------------------------------------------
##
class HistopathologyDataset(Dataset):
    def __init__(self, dataset_path, transforms=None, name='train'):
        super(HistopathologyDataset, self).__init__()
        self.dataset_path = dataset_path
        self.transforms = transforms
        self.name = name

        with h5py.File(self.dataset_path, 'r', swmr=True) as hdf:
            self.image_ids = list(hdf.keys())
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        with h5py.File(self.dataset_path, 'r', swmr=True) as hdf:
            
            img_np = hdf[img_id]['img'][()]  
            img = torch.from_numpy(img_np).float()  
            
            if self.transforms:
                img = self.transforms(img)
            
            if self.name in ['train', 'val']:
                label = hdf[img_id]['label'][()]
                center = hdf[img_id]['metadata'][0]
                return img, label, center
            else:
                return img, img_id

##
## ------------------------------------------------------------------------------------------------------------
##

def extract_structural_features(dataset, batch_size=64, device="cpu"):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Filtres Sobel
    sobel_x = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=torch.float32, device=device).view(1,1,3,3)
    sobel_y = sobel_x.transpose(2,3)

    all_features = []
    all_centers = []
    all_labels = []

    for images, labels, centers in tqdm(dataloader):
        images = images.to(device)  # (B, 3, H, W)
        B = images.shape[0]

        r, g, b = images[:, 0], images[:, 1], images[:, 2]

        # Luminance perceptuelle
        luminance = 0.299 * r + 0.587 * g + 0.114 * b

        # Intensité & contraste
        intensity = luminance.mean(dim=(1,2))
        contrast = luminance.std(dim=(1,2))

        # Moyennes & std des canaux
        mean_rgb = images.mean(dim=(2,3))  # (B, 3)
        std_rgb = images.std(dim=(2,3))    # (B, 3)

        # Contours (sobel)
        luminance_4d = luminance.unsqueeze(1)  # (B,1,H,W)
        grad_x = F.conv2d(luminance_4d, sobel_x, padding=1)
        grad_y = F.conv2d(luminance_4d, sobel_y, padding=1)
        grad = torch.sqrt(grad_x**2 + grad_y**2).squeeze(1)
        edge_mean = grad.mean(dim=(1,2))
        edge_std = grad.std(dim=(1,2))

        # Saturation & brightness
        max_rgb = torch.max(images, dim=1)[0]
        min_rgb = torch.min(images, dim=1)[0]
        saturation = (max_rgb - min_rgb).mean(dim=(1,2))
        brightness = max_rgb.mean(dim=(1,2))

        # Entropie approximée
        eps = 1e-7
        norm_lum = luminance.clamp(min=eps, max=1.0)
        entropy = (-norm_lum * torch.log2(norm_lum)).mean(dim=(1,2))

        # Concaténer toutes les features (B, 13)
        batch_features = torch.cat([
            intensity.unsqueeze(1),
            contrast.unsqueeze(1),
            mean_rgb,
            std_rgb,
            edge_mean.unsqueeze(1),
            edge_std.unsqueeze(1),
            entropy.unsqueeze(1),
            saturation.unsqueeze(1),
            brightness.unsqueeze(1)
        ], dim=1)

        all_features.append(batch_features)
        all_labels.append(labels)
        all_centers.append(centers)

    # Tout concaténer
    all_features_tensor = torch.cat(all_features, dim=0).cpu()
    all_labels_tensor = torch.cat(all_labels, dim=0).cpu()
    all_centers_tensor = torch.cat(all_centers, dim=0).cpu()

    # Création finale du DataFrame
    df = pd.DataFrame(all_features_tensor.numpy(), columns=[
        "intensity", "contrast",
        "mean_r", "mean_g", "mean_b",
        "std_r", "std_g", "std_b",
        "edge_mean", "edge_std",
        "entropy", "saturation", "brightness"
    ])
    df["label"] = all_labels_tensor.numpy()
    df["center"] = all_centers_tensor.numpy()

    return df

##
## ------------------------------------------------------------------------------------------------------------
##
    
def get_features_from_model(model : torchvision.models, 
                 dataset: Dataset,
                 device : torch.device = "cpu",
                 batch_size : int = 32)-> pd.DataFrame :
    """
    Function to extract features from a pretrained model

    Args:
        model (torchvision.models): the pretrained model to extract features.
        dataset_path (str): path to the data to extract features from.
        transforms (torchvision.transforms, optional): Transform of the data. Defaults to None.
        device (torch.device, optional): Defaults to "cpu"
        batch_size (int, optional): Defaults to 32.

    Returns:
        pd.DataFrame: the dataframe with the extracted features, labels and center
    """
    model.to(device)
    model.eval()

    dataloader = tqdm(DataLoader(dataset, batch_size=batch_size, shuffle=False), unit='batch')

    extracted_features = []

    for imgs, labels, centers in dataloader:
        imgs = imgs.to(device)
        with torch.no_grad():
            features = model(imgs)  # shape: [B, D]
        
        features = features.cpu().numpy()
        centers = centers.numpy()
        labels = labels.numpy()

        for feat, c, l in zip(features, centers, labels):
            extracted_features.append(list(feat) + [c, l])

    # Build column names
    feature_dim = features.shape[1]
    columns = [f"feat_{i}" for i in range(feature_dim)] + ["center", "label"]

    return pd.DataFrame(extracted_features, columns=columns)

##
## ------------------------------------------------------------------------------------------------------------
##

def plot_tSNE(df : pd.DataFrame,
              subset_size : float = 1.,
              n_components : int = 2,
              perplexity : int = 30,
              max_iter : int = 1000,):
    """plot the t-SNE visualization per center (colored by label).
    The t-SNE is computed on the full dataset df but plotted for each center for a better 
    visualization of the clusters.

    Args:
        df (pd.DataFrame): _description_
        subset_size (float, optional): _description_. Defaults to 1..
        n_components (int, optional): _description_. Defaults to 2.
        perplexity (int, optional): _description_. Defaults to 30.
        max_iter (int, optional): _description_. Defaults to 1000.
    """
    # --- Sampling ---
    df_subset = df.sample(frac=subset_size, random_state=SEED) if subset_size < 1 else df

    # --- Feature extraction ---
    features_cols = [col for col in df_subset.columns if col.startswith('feat_')]
    X = df_subset[features_cols]

    # --- t-SNE ---
    tsne = TSNE(n_components=n_components, random_state=SEED, perplexity=perplexity, max_iter=max_iter)
    X_tsne = tsne.fit_transform(X)

    # --- Create tsne dataframe ---
    df_tsne = pd.DataFrame(X_tsne, columns=["TSNE1", "TSNE2"])
    df_tsne["center"] = df_subset["center"].values
    df_tsne["label"] = df_subset["label"].values  # still useful for shapes or analysis

    # --- Prepare the color palette for centers ---
    centers = sorted(df_tsne["center"].unique())
    palette = sns.color_palette("tab10", len(centers))  # or Set2, Paired, etc.
    center_to_color = dict(zip(centers, palette))

    # --- Plot all points in one plot, colored by center ---
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df_tsne,
        x="TSNE1", y="TSNE2",
        hue="center",
        style="label",
        palette=center_to_color,
        alpha=0.3, edgecolor='w', s=60, linewidth=0.5
    )

    # --- Overlay centroids with 'x' marker in same color ---
    for center in centers:
        subset = df_tsne[df_tsne["center"] == center]
        center_x = subset["TSNE1"].mean()
        center_y = subset["TSNE2"].mean()
        plt.scatter(center_x, center_y, 
                    color=center_to_color[center],
                    edgecolor='black', 
                    marker='P', s=120, linewidth=1,zorder=10)

    plt.title("t-SNE Visualization Colored by Center", fontsize=13)
    plt.xlabel("TSNE1")
    plt.ylabel("TSNE2")
    plt.grid(False)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.show()