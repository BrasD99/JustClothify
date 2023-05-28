import os
import shutil
import wget

def clear_path(path):
  if os.path.exists(path):
      shutil.rmtree(path)
  os.mkdir(path)

def download_model(data_path):
  clear_path(data_path)
  url = "https://raw.githubusercontent.com/facebookresearch/detectron2/main/projects/DensePose/configs/densepose_rcnn_R_50_FPN_WC1M_s1x.yaml"
  wget.download(url, os.path.join(data_path, 'config.yaml'))
  url = "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_WC1M_s1x/217144516/model_final_48a9d9.pkl"
  wget.download(url, os.path.join(data_path, 'weights.pkl'))
  url = "https://raw.githubusercontent.com/facebookresearch/detectron2/main/projects/DensePose/configs/Base-DensePose-RCNN-FPN.yaml"
  wget.download(url, os.path.join(data_path, 'Base-DensePose-RCNN-FPN.yaml'))
  print('Веса и файл конфигурации модели успешно загружены')