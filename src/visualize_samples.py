import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_samples(data_dir, output_path, num_samples=5):
    files = glob.glob(os.path.join(data_dir, "*.npz"))
    if not files:
        print(f"No .npz files found in {data_dir}")
        return
        
    random.seed(42)
    selected_files = random.sample(files, min(num_samples, len(files)))
    
    fig, axes = plt.subplots(2, num_samples, figsize=(5 * num_samples, 10))
    
    colors = plt.cm.tab20(np.linspace(0, 1, 14))
    
    for i, file_path in enumerate(selected_files):
        data = np.load(file_path)
        image = data["image"]  # (D, H, W)
        centers = data["centers"]  # (N, 3) -> [z, y, x]
        radii = data["radii"]      # (N,)
        classes = data["classes"]  # (N,)
        
        # Take a single horizontal slice from the middle of the 3D volume
        slice_idx = image.shape[0] // 2
        img_slice = image[slice_idx, :, :]
        
        # Filter for labels that physically intersect this slice
        # A shell intersects the slice if the distance from its Z-center to the slice is less than its radius
        active_indices = np.abs(centers[:, 0] - slice_idx) <= radii
        
        act_centers = centers[active_indices]
        act_radii = radii[active_indices]
        act_classes = classes[active_indices]
        
        # Top row: Raw slice
        ax_raw = axes[0, i]
        ax_raw.imshow(img_slice, cmap='gray')
        ax_raw.set_title(f"Sample {i+1} (Slice {slice_idx})\nRaw Image")
        ax_raw.axis('off')
        
        # Bottom row: Labeled slice
        ax_labeled = axes[1, i]
        ax_labeled.imshow(img_slice, cmap='gray')
        ax_labeled.set_title(f"{len(act_centers)} active shells")
        ax_labeled.axis('off')
        
        # Draw labels and centroids
        for center, radius, cls in zip(act_centers, act_radii, act_classes):
            z, y, x = center[0], center[1], center[2]
            color = colors[cls % 14]
            
            # Calculate the apparent 2D radius at this specific depth slice
            # r^2 = R^2 - dZ^2
            dz = abs(z - slice_idx)
            apparent_radius = np.sqrt(max(0, radius**2 - dz**2))
            
            # Draw the bounding circle (detection area)
            if apparent_radius > 0:
                circle = patches.Circle((x, y), apparent_radius, linewidth=1.5, edgecolor=color, facecolor='none', alpha=0.7)
                ax_labeled.add_patch(circle)
            
            # Draw the precise centroid (what we actually want to predict)
            ax_labeled.plot(x, y, marker='x', color='red', markersize=8, markeredgewidth=2)
            # Add small class label near the centroid
            ax_labeled.text(x + 2, y - 2, f"c{cls}", color=color, fontsize=9, weight='bold')

    # Create legend
    legend_elements = [
        patches.Patch(facecolor='none', edgecolor=colors[i], label=f'Class {i}')
        for i in range(14)
    ]
    # Add a legend entry for the centroid marker
    legend_elements.append(plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='red', markeredgecolor='red', markersize=8, label='Centroid Target'))
    
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=8, title="Annotations")
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    visualize_samples(
        data_dir="data/synthetic/train",
        output_path="figures/synthetic_slice_samples.png"
    )
