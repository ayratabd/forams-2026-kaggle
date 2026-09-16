import os
import subprocess
import sys
import argparse

def main():
    print("Generating synthetic data using 2nd place procedural packing logic...")
    
    # Path to the 2nd place repository we cloned
    repo_dir = os.path.abspath("references/2nd-place")
    
    # Path to our dataset
    train_root = os.path.abspath("data/train") # Extracted kaggle data
    
    # Output path for synthetic data
    output_dir = os.path.abspath("data/synthetic")
    
    script_path = os.path.join(repo_dir, "scripts", "build_detector_data.py")
    
    if not os.path.exists(script_path):
        print(f"Error: Could not find script at {script_path}")
        sys.exit(1)
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-scenes", type=int, default=3000)
    parser.add_argument("--validation-scenes", type=int, default=150)
    args = parser.parse_args()

    # The 2nd place script takes --train-root, --output, and --config
    config_path = os.path.join(repo_dir, "configs", "core_n.yaml")
    cmd = [
        "python", script_path,
        "--config", config_path,
        "--train-root", train_root,
        "--output", output_dir,
        "--train-scenes", str(args.train_scenes),
        "--validation-scenes", str(args.validation_scenes)
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    print(f"Synthetic data successfully generated in {output_dir}")

if __name__ == "__main__":
    main()
