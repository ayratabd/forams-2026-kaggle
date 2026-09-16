import os
import glob
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm

from model import HybridForamsModel

def apply_3d_nms(predictions, iou_threshold=0.5):
    """
    Applies 3D Non-Maximum Suppression to filter out overlapping centerpoint predictions.
    predictions: List of dicts, e.g., [{"center": (x, y, z), "score": 0.9, "class": 1}, ...]
    """
    predictions = sorted(predictions, key=lambda x: x["score"], reverse=True)
    
    keep = []
    for p in predictions:
        overlap = False
        for k in keep:
            dist = np.linalg.norm(np.array(p["center"]) - np.array(k["center"]))
            if dist < 48.0: # 48 voxel threshold from 3rd place writeup
                overlap = True
                break
        if not overlap:
            keep.append(p)
            
    return keep

def generate_submission(volume_predictions, output_path="submission.csv"):
    rows = []
    LABELS = [
        "Globigerina_bulloides", "Globigerinella_calida", "Globigerinella_siphonifera",
        "Globigerinita_glutinata", "Globigerinoides_conglobatus", "Globigerinoides_ruber",
        "Globigerinoides_rubescens", "Globorotalia_crassaformis", "Globorotalia_inflata",
        "Globorotalia_menardii", "Globorotalia_scitula", "Globorotalia_truncatulinoides",
        "Neogloboquadrina_dutertrei", "Neogloboquadrina_incompta", "Neogloboquadrina_pachyderma",
        "Orbulina_universa", "Pulleniatina_obliquiloculata", "Turborotalita_quinqueloba"
    ]
    # Reduce to 14 classes used by this specific competition split
    LABELS = LABELS[:14] 
    
    for filename, preds in volume_predictions.items():
        tokens = []
        for p in preds:
            cls_name = LABELS[p["class"]]
            x, y, z = p["center"]
            tokens.extend([cls_name, f"{z:.2f}", f"{y:.2f}", f"{x:.2f}"])
            
        rows.append({"filename": filename, "centerpoint": ";".join(tokens)})
        
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")

def run_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load best checkpoint
    checkpoints = glob.glob("checkpoints/*.ckpt")
    if not checkpoints:
        raise RuntimeError("No checkpoints found! Ensure training finished.")
    best_ckpt = sorted(checkpoints, key=lambda x: float(x.split("val_loss=")[1].split(".ckpt")[0]))[0]
    print(f"Loading best checkpoint: {best_ckpt}")
    
    model = HybridForamsModel.load_from_checkpoint(best_ckpt).to(device)
    model.eval()
    
    # 2. Iterate over test files
    test_files = sorted(glob.glob("data/test/*.tif"))
    if not test_files:
        raise RuntimeError("No test files found in data/test/")
        
    all_preds = {}
    
    for tf in tqdm(test_files, desc="Processing Volumes"):
        filename = os.path.basename(tf)
        vol = tifffile.imread(tf)
        D, H_orig, W_orig = vol.shape
        
        preds_for_vol = []
        
        # Z-axis Sliding Window
        Z_stride = 8
        for z in range(1, D - 1, Z_stride):
            slices = vol[z-1:z+2, :, :]
            image_tensor = torch.from_numpy(slices).float()
            
            # Interpolate to 224x224
            image_tensor = F.interpolate(
                image_tensor.unsqueeze(0).unsqueeze(0), 
                size=(3, 224, 224), 
                mode="trilinear", align_corners=False
            ).squeeze(0)
            
            if image_tensor.max() > 0:
                image_tensor = image_tensor / image_tensor.max()
                
            image_tensor = image_tensor.to(device)
            
            with torch.no_grad():
                with torch.autocast(device_type="cuda" if "cuda" in str(device) else "cpu"):
                    hm, off = model(image_tensor)
                hm = torch.sigmoid(hm)
                
            # Peak Extraction
            # hm shape: [1, 14, 64, 64]
            maxm = F.max_pool2d(hm, kernel_size=3, stride=1, padding=1)
            keep = (hm == maxm) & (hm > 0.3)
            
            # Find indices
            indices = torch.nonzero(keep[0]) # shape: [N, 3] -> (c, y, x)
            
            if len(indices) > 0:
                for idx in indices:
                    c, y_int, x_int = idx
                    score = hm[0, c, y_int, x_int].item()
                    
                    # Extract subpixel offset
                    off_x = off[0, 0, y_int, x_int].item()
                    off_y = off[0, 1, y_int, x_int].item()
                    
                    x_out = x_int.item() + off_x
                    y_out = y_int.item() + off_y
                    
                    # Map back to original resolution
                    x_orig = (x_out / 64.0) * W_orig
                    y_orig = (y_out / 64.0) * H_orig
                    
                    preds_for_vol.append({
                        "center": (x_orig, y_orig, z),
                        "score": score,
                        "class": c.item()
                    })
                    
        # Apply 3D NMS
        filtered = apply_3d_nms(preds_for_vol)
        all_preds[filename] = filtered
        
    generate_submission(all_preds)

if __name__ == "__main__":
    run_inference()
