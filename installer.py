import subprocess
import os
from helpers.methods import download_model

# установка зависимости detectron2
subprocess.call(['pip', 'install', 'git+https://github.com/facebookresearch/detectron2@main#subdirectory=projects/DensePose'])
# установка зависимости wget
subprocess.call(['pip', 'install', 'wget'])

if not os.path.exists('data'):
    download_model('data')