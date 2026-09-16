import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoModel

def focal_heatmap_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 4.0,
    positive_threshold: float = 0.999,
) -> torch.Tensor:
    probability = torch.sigmoid(logits).clamp(1e-4, 1.0 - 1e-4)
    positive = (target >= positive_threshold).float()
    negative = 1.0 - positive
    positive_loss = (
        -((1.0 - probability) ** alpha) * torch.log(probability) * positive
    )
    negative_loss = (
        -((1.0 - target) ** beta)
        * (probability**alpha)
        * torch.log(1.0 - probability)
        * negative
    )
    return (positive_loss.sum() + negative_loss.sum()) / positive.sum().clamp(
        min=1.0
    )

def offset_l1_loss(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    expanded = mask.unsqueeze(1)
    denominator = expanded.sum().clamp(min=1.0) * 3.0
    return (torch.abs(prediction - target) * expanded).sum() / denominator

class HybridForamsModel(pl.LightningModule):
    def __init__(self, model_name="./models/dinov2-base", num_classes=14, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        
        # CenterNet Upsampling (16x16 -> 64x64)
        # Using two ConvTranspose2d layers with stride 2
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(hidden_size, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Heatmap Head (Outputs [B, 14, 64, 64])
        self.heatmap_head = nn.Conv2d(128, num_classes, kernel_size=1)
        
        # Offset Head (Outputs [B, 2, 64, 64])
        self.offset_head = nn.Conv2d(128, 2, kernel_size=1)
        
        # Initialize final convs like original CenterNet for stability
        self.heatmap_head.bias.data.fill_(-2.19)
        
    def forward(self, x):
        # DINOv2 returns last_hidden_state of shape [B, SeqLen, Hidden]
        # SeqLen for 224x224 with patch_size 14 is 1 (CLS) + 16x16 = 257
        outputs = self.backbone(pixel_values=x)
        features = outputs.last_hidden_state
        
        # Remove CLS token and reshape to spatial [B, Hidden, 16, 16]
        spatial_features = features[:, 1:, :] # [B, 256, 768]
        B, N, C = spatial_features.shape
        H_patch = int(N**0.5)
        spatial_features = spatial_features.permute(0, 2, 1).contiguous().view(B, C, H_patch, H_patch)
        
        # Upsample to [B, 128, 64, 64]
        up_features = self.upsample(spatial_features)
        
        heatmap_logits = self.heatmap_head(up_features)
        offsets = self.offset_head(up_features)
        
        return heatmap_logits, offsets

    def training_step(self, batch, batch_idx):
        images, targets = batch
        hm_logits, pred_offsets = self(images)
        
        target_hm = targets["heatmap"]
        target_offset = targets["offset"]
        target_mask = targets["mask"]
        
        loss_hm = focal_heatmap_loss(hm_logits, target_hm)
        loss_off = offset_l1_loss(pred_offsets, target_offset, target_mask)
        
        loss = loss_hm + 0.1 * loss_off
        
        self.log("train_loss", loss, prog_bar=True)
        self.log("hm_loss", loss_hm)
        self.log("off_loss", loss_off)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        hm_logits, pred_offsets = self(images)
        
        target_hm = targets["heatmap"]
        target_offset = targets["offset"]
        target_mask = targets["mask"]
        
        loss_hm = focal_heatmap_loss(hm_logits, target_hm)
        loss_off = offset_l1_loss(pred_offsets, target_offset, target_mask)
        
        loss = loss_hm + 0.1 * loss_off
        
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
