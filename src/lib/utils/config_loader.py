"""
Configuration loader with variable substitution support
"""
import json
import os
import re
import sys

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from config import config

def load_config_with_substitution(config_path):
    """
    Load JSON config file with variable substitution
    Supports ${VARIABLE_NAME} syntax
    """
    # Read the config file
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Define substitution variables
    substitutions = {
        'DATA_ROOT': config.DATA_ROOT,
        'MODEL_DIR': config.MODEL_DIR,
        'EXP_DIR': config.EXP_DIR,
        'PROJECT_ROOT': str(config.PROJECT_ROOT),
        'MOT17_ROOT': config.mot17_root,
        'MOT20_ROOT': config.mot20_root,
        'CROWDHUMAN_ROOT': config.crowdhuman_root
    }
    
    # Perform variable substitution
    def replace_var(match):
        var_name = match.group(1)
        if var_name in substitutions:
            return substitutions[var_name]
        elif var_name in os.environ:
            return os.environ[var_name]
        else:
            raise ValueError(f"Unknown variable: {var_name}")
    
    # Replace ${VAR_NAME} patterns
    content = re.sub(r'\$\{([^}]+)\}', replace_var, content)
    
    # Parse JSON
    return json.loads(content)

def load_data_config():
    """Load the main data configuration"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'cfg', 'data.json')
    return load_config_with_substitution(config_path)