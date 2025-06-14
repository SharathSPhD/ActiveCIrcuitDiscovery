#!/usr/bin/env python3
"""
Import Fixer for ActiveCircuitDiscovery Framework
Converts relative imports to absolute imports for proper execution
"""

import os
import re
from pathlib import Path

def fix_relative_imports(file_path):
    """Fix relative imports in a Python file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Pattern to match relative imports
    patterns = [
        (r'from \.\.([^.\s]+) import', r'from \1 import'),  # from ..module import
        (r'from \.([^.\s]+) import', r'from \1 import'),    # from .module import
        (r'from \.\.([^.\s]+)\.([^.\s]+) import', r'from \1.\2 import'),  # from ..module.submodule import
        (r'from \.([^.\s]+)\.([^.\s]+) import', r'from \1.\2 import'),    # from .module.submodule import
    ]
    
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)
    
    # Write back if changed
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed imports in: {file_path}")
        return True
    return False

def fix_all_imports(src_dir):
    """Fix imports in all Python files in src directory."""
    src_path = Path(src_dir)
    fixed_count = 0
    
    for py_file in src_path.rglob("*.py"):
        if py_file.name != "__init__.py":  # Skip __init__.py files
            if fix_relative_imports(py_file):
                fixed_count += 1
    
    print(f"Fixed imports in {fixed_count} files")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        src_directory = sys.argv[1]
    else:
        src_directory = "./src"
    
    fix_all_imports(src_directory)