import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_metrics(csv_path, output_path):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    # Forward fill to handle NaNs where train and val metrics are recorded at different steps
    df = df.ffill()
    
    plt.figure(figsize=(10, 5))
    
    if 'train_loss' in df.columns:
        plt.plot(df['step'], df['train_loss'], label='Training Loss', color='blue', alpha=0.7)
    
    if 'val_loss' in df.columns:
        # val_loss is only recorded per epoch, so let's drop NaNs for it to plot points cleanly
        val_df = df.dropna(subset=['val_loss'])
        plt.plot(val_df['step'], val_df['val_loss'], label='Validation Loss', color='red', marker='o', linewidth=2)
        
    plt.title('Training and Validation Loss Curves (Dummy Placeholder Run)')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss Value')
    plt.ylim(0, 1) # Zoom in to show flatline
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    plot_metrics("lightning_logs/version_2/metrics.csv", "figures/dummy_loss_curves.png")
