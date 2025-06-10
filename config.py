"""
FairMOT Configuration Management
Centralized configuration for datasets, models, and paths
"""
import os
from pathlib import Path

class Config:
    """Centralized configuration management"""
    
    def __init__(self):
        # Project root directory
        self.PROJECT_ROOT = Path(__file__).parent.absolute()
        
        # Environment variable for data root, with fallback
        self.DATA_ROOT = os.environ.get('FAIRMOT_DATA_ROOT', self._get_default_data_root())
        
        # Model directory
        self.MODEL_DIR = os.environ.get('FAIRMOT_MODEL_DIR', str(self.PROJECT_ROOT / 'models'))
        
        # Experiment output directory
        self.EXP_DIR = os.environ.get('FAIRMOT_EXP_DIR', str(self.PROJECT_ROOT / 'exp'))
        
        # Ensure directories exist
        self._ensure_directories()
    
    def _get_default_data_root(self):
        """Get default data root based on platform and common locations"""
        possible_paths = [
            '/Users/yutao/Dataset/MOT/JDE',  # Current macOS path
            '~/Dataset/MOT/JDE',             # User home relative
            './data',                        # Project relative
            '/data/MOT',                     # Common Linux path
            '/home/dataset/MOT/JDE'          # Common server path
        ]
        
        for path in possible_paths:
            expanded_path = os.path.expanduser(path)
            if os.path.exists(expanded_path):
                return expanded_path
        
        # Fallback to project relative
        return str(self.PROJECT_ROOT / 'data')
    
    def _ensure_directories(self):
        """Ensure required directories exist"""
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        os.makedirs(self.EXP_DIR, exist_ok=True)
        
        # Create data directory if it doesn't exist
        if not os.path.exists(self.DATA_ROOT):
            print(f"Warning: Data root {self.DATA_ROOT} does not exist. Creating directory...")
            os.makedirs(self.DATA_ROOT, exist_ok=True)
    
    @property
    def mot17_root(self):
        """MOT17 dataset root"""
        return os.path.join(self.DATA_ROOT, 'MOT17')
    
    @property
    def mot20_root(self):
        """MOT20 dataset root"""
        return os.path.join(self.DATA_ROOT, 'MOT20')
    
    @property
    def crowdhuman_root(self):
        """CrowdHuman dataset root"""
        return os.path.join(self.DATA_ROOT, 'crowdhuman')
    
    def get_dataset_path(self, dataset_name, split='train'):
        """Get dataset path for specific dataset and split"""
        dataset_paths = {
            'mot17': {
                'train': os.path.join(self.mot17_root, 'images', 'train'),
                'test': os.path.join(self.mot17_root, 'images', 'test'),
                'labels': os.path.join(self.mot17_root, 'labels_with_ids', 'train')
            },
            'mot20': {
                'train': os.path.join(self.mot20_root, 'images', 'train'),
                'test': os.path.join(self.mot20_root, 'images', 'test'),
                'labels': os.path.join(self.mot20_root, 'labels_with_ids', 'train')
            },
            'crowdhuman': {
                'train': os.path.join(self.crowdhuman_root, 'images', 'train'),
                'val': os.path.join(self.crowdhuman_root, 'images', 'val'),
                'labels_train': os.path.join(self.crowdhuman_root, 'labels_with_ids', 'train'),
                'labels_val': os.path.join(self.crowdhuman_root, 'labels_with_ids', 'val')
            }
        }
        
        if dataset_name not in dataset_paths:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        if split not in dataset_paths[dataset_name]:
            raise ValueError(f"Unknown split '{split}' for dataset '{dataset_name}'")
        
        return dataset_paths[dataset_name][split]
    
    def print_config(self):
        """Print current configuration"""
        print("=== FairMOT Configuration ===")
        print(f"Project Root: {self.PROJECT_ROOT}")
        print(f"Data Root: {self.DATA_ROOT}")
        print(f"Model Directory: {self.MODEL_DIR}")
        print(f"Experiment Directory: {self.EXP_DIR}")
        print(f"MOT17 Root: {self.mot17_root}")
        print(f"MOT20 Root: {self.mot20_root}")
        print(f"CrowdHuman Root: {self.crowdhuman_root}")
        print("=============================")

# Global configuration instance
config = Config()

# For backward compatibility and easy access
def get_data_root():
    """Get the data root directory"""
    return config.DATA_ROOT

def get_model_dir():
    """Get the model directory"""
    return config.MODEL_DIR

def get_exp_dir():
    """Get the experiment directory"""
    return config.EXP_DIR