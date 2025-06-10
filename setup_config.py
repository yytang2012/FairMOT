#!/usr/bin/env python3
"""
FairMOT Configuration Setup Tool
Easy way to configure data paths for FairMOT
"""
import os
import sys
from pathlib import Path

def setup_environment():
    """Interactive setup for FairMOT environment"""
    print("=== FairMOT Configuration Setup ===")
    print()
    
    # Get current config
    from config import config
    
    print("Current configuration:")
    config.print_config()
    print()
    
    print("You can configure FairMOT in several ways:")
    print("1. Set environment variables (recommended)")
    print("2. Create a .env file in the project root")
    print("3. Update the default paths in config.py")
    print()
    
    # Get user's data root
    current_data_root = config.DATA_ROOT
    print(f"Current data root: {current_data_root}")
    
    new_data_root = input("Enter new data root path (or press Enter to keep current): ").strip()
    
    if new_data_root:
        # Expand user path and make absolute
        new_data_root = os.path.abspath(os.path.expanduser(new_data_root))
        
        if not os.path.exists(new_data_root):
            create = input(f"Directory {new_data_root} doesn't exist. Create it? (y/n): ").lower().strip()
            if create == 'y':
                os.makedirs(new_data_root, exist_ok=True)
                print(f"Created directory: {new_data_root}")
            else:
                print("Setup cancelled.")
                return
        
        # Show how to set environment variable
        print()
        print("To use this configuration, set the environment variable:")
        print(f"export FAIRMOT_DATA_ROOT='{new_data_root}'")
        print()
        print("Or add it to your ~/.bashrc or ~/.zshrc:")
        print(f"echo 'export FAIRMOT_DATA_ROOT=\"{new_data_root}\"' >> ~/.bashrc")
        print()
        
        # Create .env file option
        create_env = input("Create .env file in project root? (y/n): ").lower().strip()
        if create_env == 'y':
            env_file = Path(__file__).parent / '.env'
            with open(env_file, 'w') as f:
                f.write(f"FAIRMOT_DATA_ROOT={new_data_root}\n")
                f.write(f"FAIRMOT_MODEL_DIR={config.MODEL_DIR}\n")
                f.write(f"FAIRMOT_EXP_DIR={config.EXP_DIR}\n")
            print(f"Created .env file: {env_file}")
            print("Note: You'll need to load this file manually or use python-dotenv")
    
    print()
    print("Setup complete!")
    print()
    print("Expected dataset structure:")
    print(f"{new_data_root or current_data_root}/")
    print("├── MOT17/")
    print("│   ├── images/")
    print("│   │   ├── train/")
    print("│   │   └── test/")
    print("│   └── labels_with_ids/")
    print("├── MOT20/")
    print("│   ├── images/")
    print("│   │   ├── train/")
    print("│   │   └── test/")
    print("│   └── labels_with_ids/")
    print("└── crowdhuman/")
    print("    ├── images/")
    print("    │   ├── train/")
    print("    │   └── val/")
    print("    └── labels_with_ids/")

def check_config():
    """Check current configuration and dataset availability"""
    print("=== FairMOT Configuration Check ===")
    
    from config import config
    config.print_config()
    
    print("\nDataset availability check:")
    
    # Check MOT17
    mot17_train = config.get_dataset_path('mot17', 'train')
    mot17_exists = os.path.exists(mot17_train)
    print(f"MOT17 train: {'✓' if mot17_exists else '✗'} {mot17_train}")
    
    # Check MOT20
    mot20_train = config.get_dataset_path('mot20', 'train')
    mot20_exists = os.path.exists(mot20_train)
    print(f"MOT20 train: {'✓' if mot20_exists else '✗'} {mot20_train}")
    
    # Check models directory
    model_dir_exists = os.path.exists(config.MODEL_DIR)
    print(f"Model directory: {'✓' if model_dir_exists else '✗'} {config.MODEL_DIR}")
    
    if not any([mot17_exists, mot20_exists]):
        print("\nWarning: No datasets found!")
        print("Run 'python setup_config.py setup' to configure paths.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "setup":
            setup_environment()
        elif command == "check":
            check_config()
        else:
            print("Usage: python setup_config.py [setup|check]")
    else:
        print("FairMOT Configuration Tool")
        print("Usage:")
        print("  python setup_config.py setup  - Interactive configuration setup")
        print("  python setup_config.py check  - Check current configuration")
        print()
        check_config()