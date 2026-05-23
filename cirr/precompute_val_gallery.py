import os
import json
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

class UniqueGalleryDataset(Dataset):
    def __init__(self, cirr_root, preprocess):
        self.cirr_root = Path(cirr_root)
        self.preprocess = preprocess
        
        # Load split file
        split_path = self.cirr_root / "cirr" / "image_splits" / "split.rc2.val.json"
        with open(split_path, "r") as f:
            self.img_paths = json.load(f)
            
        self.img_ids = sorted(list(self.img_paths.keys()))
        print(f"UniqueGalleryDataset: {len(self.img_ids)} unique images to encode")

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        rel_path = self.img_paths[img_id]
        if rel_path.startswith("./"):
            rel_path = rel_path[2:]
        img_path = self.cirr_root / rel_path
        
        try:
            image = self.preprocess(Image.open(img_path).convert("RGB"))
            return image, img_id, str(img_path)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # return zero tensor as fallback
            return torch.zeros(3, 224, 224), img_id, str(img_path)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cirr_root = "/home/otw/chiennhm/data/CIRR"
    output_dir = Path("cache/cirr_cls/val")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load CLIP
    import clip
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    
    dataset = UniqueGalleryDataset(cirr_root, preprocess)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)
    
    all_embeddings = []
    all_ids = []
    all_paths = []
    
    with torch.no_grad():
        for images, img_ids, img_paths in tqdm(loader, desc="Encoding gallery"):
            images = images.to(device)
            embeddings = model.encode_image(images)
            embeddings = F.normalize(embeddings.float(), dim=-1)
            all_embeddings.append(embeddings.cpu().half())
            all_ids.extend(img_ids)
            all_paths.extend(img_paths)
            
    all_embeddings = torch.cat(all_embeddings, dim=0)
    
    # Save embeddings and IDs
    torch.save(all_embeddings, output_dir / "gallery_val_embeddings.pt")
    
    gallery_info = {
        "img_ids": all_ids,
        "img_paths": all_paths
    }
    with open(output_dir / "gallery_val_info.json", "w") as f:
        json.dump(gallery_info, f, indent=2)
        
    print(f"Saved {all_embeddings.shape[0]} embeddings to {output_dir / 'gallery_val_embeddings.pt'}")

if __name__ == "__main__":
    main()
