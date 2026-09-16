import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def parse_tensorboard_events(log_dir):
    # Find the most recent version directory
    versions = sorted(glob.glob(os.path.join(log_dir, "version_*")), key=os.path.getmtime)
    if not versions:
        print("No tensorboard logs found.")
        return None
        
    latest_version = versions[-1]
    print(f"Reading logs from: {latest_version}")
    
    event_files = glob.glob(os.path.join(latest_version, "events.out.tfevents.*"))
    if not event_files:
        print("No event files found in the latest version.")
        return None
        
    event_file = event_files[0]
    
    # Load events
    ea = EventAccumulator(event_file)
    ea.Reload()
    
    # Extract train_loss and val_loss
    data = {}
    for tag in ["train_loss", "val_loss", "hm_loss", "off_loss"]:
        if tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            data[tag] = pd.DataFrame([(e.step, e.value) for e in events], columns=['step', tag])
    
    if not data:
        print("No loss data found in events.")
        return None
        
    # Merge on step
    df = None
    for tag, df_tag in data.items():
        if df is None:
            df = df_tag
        else:
            df = pd.merge(df, df_tag, on='step', how='outer')
            
    df = df.sort_values('step').reset_index(drop=True)
    return df

def plot_losses(df, output_path):
    if df is None or df.empty:
        return
        
    plt.figure(figsize=(10, 6))
    
    if 'train_loss' in df.columns:
        # Train loss might be noisy, optionally smooth it
        plt.plot(df['step'], df['train_loss'], label='Total Train Loss', color='blue', alpha=0.5)
        
    if 'val_loss' in df.columns:
        val_df = df.dropna(subset=['val_loss'])
        plt.plot(val_df['step'], val_df['val_loss'], label='Validation Loss', color='red', marker='o', linewidth=2)
        
    plt.title('True CenterNet Training and Validation Loss Curves')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    df = parse_tensorboard_events("lightning_logs")
    if df is not None:
        plot_losses(df, "figures/real_loss_curves.png")
