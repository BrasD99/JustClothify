import cv2
import imageio
import numpy as np
import os
import torch
from typing import Any, Dict

from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2.structures.instances import Instances

from densepose import add_densepose_config
from densepose.structures import (
    DensePoseChartPredictorOutput,
    DensePoseEmbeddingPredictorOutput
)
from densepose.vis.base import CompoundVisualizer
from densepose.vis.densepose_outputs_vertex import get_texture_atlases
from densepose.vis.densepose_results_textures import (
    DensePoseResultsVisualizerWithTexture as dp_iuv_texture,
    get_texture_atlas
)
from densepose.vis.extractor import (
    CompoundExtractor,
    create_extractor,
    DensePoseOutputsExtractor,
    DensePoseResultExtractor
)

from detectron2.data.detection_utils import read_image

class TextureProcessor:
  def __init__(self, input_dict):
    self.input_dict = input_dict
    self.config = self.get_config(self.input_dict['densepose']['config'],\
                    self.input_dict['densepose']['weights'])
    self.predictor = DefaultPredictor(self.config)
  
  def process_texture(self, image_filename, output_filename):
    image = cv2.imread(image_filename, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output = self.execute(image)[0]

    if 'pred_densepose' in output:
        texture = self.create_iuv(output, image)
        imageio.imwrite(output_filename, texture)
    
    else:
      print(f'Одежда не распознана для {image_filename}')

  def extract(self, guid):
    outputs = dict()
    person_filename = self.input_dict['person']
    for i, model_filename in enumerate(self.input_dict['models']):
      texture_name = f'{guid}_{i}.png'
      texture_fullname = os.path.join(self.input_dict["temp_path"], texture_name)
      self.process_texture(model_filename, texture_fullname)
      output_filename = os.path.join(self.input_dict["output_path"], f'{guid}-result.jpg')
      self.overlay_texture(texture_fullname, person_filename, output_filename)
      outputs[model_filename] = output_filename
    return outputs

  def overlay_texture(self, texture_name, original_image, output_name):
    texture_atlas = get_texture_atlas(texture_name)
    texture_atlases_dict = get_texture_atlases(None)
    vis = dp_iuv_texture(
        cfg=self.config,
        texture_atlas=texture_atlas,
        texture_atlases_dict=texture_atlases_dict
    )

    visualizers = [vis]
    extractor = create_extractor(vis)
    extractors = [extractor]

    visualizer = CompoundVisualizer(visualizers)
    extractor = CompoundExtractor(extractors)

    img = read_image(original_image, format="BGR")
    with torch.no_grad():
      outputs = self.predictor(img)["instances"]
    
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image = np.tile(image[:, :, np.newaxis], [1, 1, 3])
    data = extractor(outputs)
    image_vis = visualizer.visualize(image, data)

    cv2.imwrite(output_name, image_vis)


  def parse_iuv(self, result):
      i = result['pred_densepose'][0].labels.cpu().numpy().astype(float)
      uv = (result['pred_densepose'][0].uv.cpu().numpy() * 255.0).astype(float)
      iuv = np.stack((uv[1, :, :], uv[0, :, :], i))
      iuv = np.transpose(iuv, (1, 2, 0))
      return iuv

  def parse_bbox(self, result):
      return result["pred_boxes_XYXY"][0].cpu().numpy()

  def interpolate_tex(self, tex):
      # code is adopted from https://github.com/facebookresearch/DensePose/issues/68
      valid_mask = np.array((tex.sum(0) != 0) * 1, dtype='uint8')
      radius_increase = 10
      kernel = np.ones((radius_increase, radius_increase), np.uint8)
      dilated_mask = cv2.dilate(valid_mask, kernel, iterations=1)
      region_to_fill = dilated_mask - valid_mask
      invalid_region = 1 - valid_mask
      actual_part_max = tex.max()
      actual_part_min = tex.min()
      actual_part_uint = np.array(
          (tex - actual_part_min) / (actual_part_max - actual_part_min) * 255, dtype='uint8')
      actual_part_uint = cv2.inpaint(actual_part_uint.transpose((1, 2, 0)), invalid_region, 1,
                                    cv2.INPAINT_TELEA).transpose((2, 0, 1))
      actual_part = (actual_part_uint / 255.0) * \
          (actual_part_max - actual_part_min) + actual_part_min
      # only use dilated part
      actual_part = actual_part * dilated_mask

      return actual_part

  def concat_textures(self, array):
      texture = []
      for i in range(4):
          tmp = array[6 * i]
          for j in range(6 * i + 1, 6 * i + 6):
              tmp = np.concatenate((tmp, array[j]), axis=1)
          texture = tmp if len(texture) == 0 else np.concatenate(
              (texture, tmp), axis=0)
      return texture

  def get_texture(self, im, iuv, bbox, tex_part_size=200):
      im = im.transpose(2, 1, 0) / 255
      image_w, image_h = im.shape[1], im.shape[2]
      bbox[2] = bbox[2] - bbox[0]
      bbox[3] = bbox[3] - bbox[1]
      x, y, w, h = [int(v) for v in bbox]
      bg = np.zeros((image_h, image_w, 3))
      bg[y:y + h, x:x + w, :] = iuv
      iuv = bg
      iuv = iuv.transpose((2, 1, 0))
      i, u, v = iuv[2], iuv[1], iuv[0]

      n_parts = 22
      texture = np.zeros((n_parts, 3, tex_part_size, tex_part_size))

      for part_id in range(1, n_parts + 1):
          generated = np.zeros((3, tex_part_size, tex_part_size))

          x, y = u[i == part_id], v[i == part_id]

          tex_u_coo = (x * (tex_part_size - 1) / 255).astype(int)
          tex_v_coo = (y * (tex_part_size - 1) / 255).astype(int)

          tex_u_coo = np.clip(tex_u_coo, 0, tex_part_size - 1)
          tex_v_coo = np.clip(tex_v_coo, 0, tex_part_size - 1)

          for channel in range(3):
              generated[channel][tex_v_coo,
                                tex_u_coo] = im[channel][i == part_id]

          if np.sum(generated) > 0:
              generated = self.interpolate_tex(generated)

          texture[part_id - 1] = generated[:, ::-1, :]

      tex_concat = np.zeros((24, tex_part_size, tex_part_size, 3))
      for i in range(texture.shape[0]):
          tex_concat[i] = texture[i].transpose(2, 1, 0)
      tex = self.concat_textures(tex_concat)

      return tex

  def create_iuv(self, results, image):
      iuv = self.parse_iuv(results)
      bbox = self.parse_bbox(results)
      uv_texture = self.get_texture(image, iuv, bbox)
      uv_texture = uv_texture.transpose([1, 0, 2])
      return uv_texture

  def get_config(self, config_fpath, model_fpath):
      cfg = get_cfg()
      add_densepose_config(cfg)
      cfg.merge_from_file(config_fpath)
      cfg.MODEL.WEIGHTS = model_fpath
      cfg.MODEL.DEVICE = "cpu"
      cfg.freeze()
      return cfg

  def execute(self, image):
      context = {'results': []}
      with torch.no_grad():
          outputs = self.predictor(image)["instances"]
          self.execute_on_outputs(context, outputs)
      return context["results"]

  def execute_on_outputs(self, context: Dict[str, Any], outputs: Instances):
    result = {}
    if outputs.has("scores"):
        result["scores"] = outputs.get("scores").cpu()
    if outputs.has("pred_boxes"):
        result["pred_boxes_XYXY"] = outputs.get("pred_boxes").tensor.cpu()
        if outputs.has("pred_densepose"):
            if isinstance(outputs.pred_densepose, DensePoseChartPredictorOutput):
                extractor = DensePoseResultExtractor()
            elif isinstance(outputs.pred_densepose, DensePoseEmbeddingPredictorOutput):
                extractor = DensePoseOutputsExtractor()
            result["pred_densepose"] = extractor(outputs)[0]
    context["results"].append(result)