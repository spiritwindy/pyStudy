import sys
import os

# 将目标目录路径添加到 sys.path
dir_path = '/path/to/your/directory'
if dir_path not in sys.path:
    sys.path.append(dir_path)