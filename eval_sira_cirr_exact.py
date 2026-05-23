import os
import glob
import argparse
import torch
import yaml
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
from Dataset import CIRDataset, GalleryDataset
from tqdm import tqdm
import clip

# Disable PIL decompression bomb warning for large images
Image.MAX_IMAGE_PIXELS = None

from sira.sira_cirr_model import SIRACIRRModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='load_best_val', help='Path to model checkpoint')
parser.add_argument('--dataset', type=str, required=True, help='Dataset to evaluate on')
parser.add_argument('--backbone_size', type=str, default='B', choices=['B', 'L', 'H'], help='Size of the backbone')
parser.add_argument('--run_name', type=str, default='default', help='Run name')
parser.add_argument('--run_all', action='store_true', help='Run all models in the directory')
parser.add_argument('--split', type=str, default='test', choices=['val', 'test'], help='Split to evaluate on')
parser.add_argument('--use_distractors', type=bool, default=True, help='Use distractors in the gallery')
args = parser.parse_args()

cfg = yaml.safe_load(open('config.yaml', 'r'))

training_datasets = ''
match args.run_name.strip('B_').lower():
    case 'soup':
        training_datasets = 'cirr,cirr_r,hotels'
    case 'cirr' | 'default':
        training_datasets = 'cirr'
    case 'cirr_r':
        training_datasets = 'cirr_r'
    case 'hotels':
        training_datasets = 'hotels'
    case 'cirr_cirr_r':
        training_datasets = 'cirr,cirr_r'
    case 'cirr_hotels':
        training_datasets = 'cirr,hotels'
    case 'cirr_r_hotels':
        training_datasets = 'cirr_r,hotels'
    case _:
        raise ValueError('Invalid run name')

# Load model checkpoint paths
paths = []
if args.path == 'load_best_val':
    model_dir = os.path.join(cfg['model_dir'], f'{training_datasets}/{args.backbone_size}/{args.run_name}')
    print(f'Loading models from {model_dir}')
    models = glob.glob(f'{model_dir}/*.pt')  # Note: SIRA checkpoint uses .pt

    models = [m for m in models if 'last' not in m]
    if len(models) == 0:
        # Fallback to output_dir format
        models = glob.glob(os.path.join(cfg['model_dir'], 'sira_cirr_v3/*.pt'))
        
    paths = models
else:
    paths.append(args.path)

# Image Preprocessing
preproc = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.481, 0.457, 0.408], std=[0.268, 0.261, 0.275]),
])

# Load dataset
dataset_map = {
    'cirr': (cfg['cirr_data_path'], 'cirr'),
    'cirr_r': (cfg['cirr_data_path'], 'cirr_r'),
    'hotels': (cfg['hotel_data_path'], 'hotels')
}

if args.dataset not in dataset_map:
    raise ValueError(f"Dataset {args.dataset} not found")

data_path, dataset_name = dataset_map[args.dataset]
dataset = CIRDataset(data_path=data_path, split=args.split, dataset=dataset_name, preprocess=preproc)
gallery_dataset = GalleryDataset(data_path=data_path, split=args.split, dataset=dataset_name, preprocess=preproc, use_distractors=args.use_distractors)

# DataLoaders
num_workers = 16
batch_size = 256
test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
gallery_loader = torch.utils.data.DataLoader(gallery_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

# Load CLIP model once
if args.backbone_size == 'B':
    clip_model_name = "ViT-B/32"
elif args.backbone_size == 'L':
    clip_model_name = "ViT-L/14"
else:
    clip_model_name = "ViT-B/32"
    
print(f"Loading CLIP model {clip_model_name}")
clip_model, _ = clip.load(clip_model_name, device=device)
clip_model = clip_model.eval().float()

# Wrapper for SIRA-CIRR model to act like ConText model
class SIRAConTextWrapper(torch.nn.Module):
    def __init__(self, sira_model, clip_model, device):
        super().__init__()
        self.sira_model = sira_model
        self.clip_model = clip_model
        self.device = device
        
    def get_image_features(self, imgs):
        v = self.clip_model.encode_image(imgs)
        return F.normalize(v.float(), dim=-1)
        
    def forward(self, ref, cap):
        # cap is a list of strings
        text_tokens = clip.tokenize(cap, truncate=True).to(self.device)
        
        v_ref = self.clip_model.encode_image(ref)
        v_ref = F.normalize(v_ref.float(), dim=-1)
        
        t_cap = self.clip_model.encode_text(text_tokens)
        t_cap = F.normalize(t_cap.float(), dim=-1)
        
        z_query, _ = self.sira_model(v_ref, t_cap)
        return z_query

for path in paths:
    print(f'EVALUATING MODEL FROM {path}')
    ckpt = torch.load(path, map_location=device)
    config = ckpt.get("config", {})
    d_model = ckpt.get("d_model", 512)
    state_dict = ckpt["model_state_dict"]
    
    # Auto-detect version from state dict keys
    keys = list(state_dict.keys())
    if any(k.startswith("query_combiner") for k in keys):
        version = "v1"
    elif any(k.startswith("combiner.delta_net") for k in keys):
        version = "v2"
    else:
        version = "v3"
    print(f"Detected model version: {version}")
    
    sira_model = SIRACIRRModel(
        d_model=d_model,
        d_synergy=config.get("d_synergy", 64),
        gate_rank=config.get("gate_rank", 16),
        gate_init_bias=config.get("gate_init_bias", 0.0),
        dropout=config.get("dropout", 0.1),
        version=version,
    ).to(device)
    sira_model.load_state_dict(state_dict)
    sira_model.eval()
    
    model = SIRAConTextWrapper(sira_model, clip_model, device)

    # Compute Gallery Embeddings
    print('COMPUTING GALLERY EMBEDDINGS')
    gallery_embeddings, gallery_paths = [], []
    with torch.no_grad():
        for batch in tqdm(gallery_loader, desc="Gallery Processing"):
            imgs, paths_batch = batch['image'].to(device), batch['image_path']
            embeddings = model.get_image_features(imgs).cpu()
            gallery_embeddings.append(embeddings)
            gallery_paths.extend(paths_batch)

    assert len(gallery_paths) == len(list(set(gallery_paths)))  # Ensure no duplicates
    gallery_embeddings = torch.cat(gallery_embeddings)  # Stack all embeddings at once
    gallery_embeddings = F.normalize(gallery_embeddings, dim=-1).float()
    gallery_paths = np.array(gallery_paths)

    # Compute Test Embeddings
    print('COMPUTING VL EMBEDDINGS')
    target_paths, vl_embeddings = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test Processing"):
            ref, cap, target = batch['reference'].to(device), batch['caption'], batch['target_path']
            embeddings = model(ref, cap).cpu()
            vl_embeddings.append(embeddings)
            target_paths.extend(target)
    
    vl_embeddings = torch.cat(vl_embeddings)
    #normalize
    vl_embeddings = F.normalize(vl_embeddings, dim=-1).float()
    target_paths = np.array(target_paths)

    # Compute Similarities
    sims = torch.matmul(vl_embeddings, gallery_embeddings.T).numpy()

    # Compute Recalls
    print("COMPUTING RECALLS")
    recalls = [1, 5, 10, 50, 100]
    sorted_indices = np.argsort(-sims, axis=1)  # Sort in descending order

    for recall in recalls:
        top_k_indices = sorted_indices[:, :recall]
        correct = sum(target_paths[i] in gallery_paths[top_k_indices[i]] for i in range(len(target_paths)))
        print(f'Top-{recall} recall: {correct / len(target_paths):.4f}')

    # Compute Median Rank
    print("COMPUTING MEDIAN RANK")
    ranks = np.array([np.where(gallery_paths[sorted_indices[i]] == target_paths[i])[0][0] for i in range(len(target_paths))])
    print(f'Median rank: {np.median(ranks)}')

    print('#########################')
