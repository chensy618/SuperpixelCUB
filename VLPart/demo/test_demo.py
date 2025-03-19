import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

import sys
sys.path.append('/mnt/d/Github/SuperpixelCUB/VLPart')
print(sys.path)
from vlpart.config import add_vlpart_config
from predictor import VisualizationDemo

# Constants
WINDOW_NAME = "image demo"
DEFAULT_CONFIG_FILE = "/mnt/d/Github/SuperpixelCUB/VLPart/configs/pascal_part/r50_pascalpart.yaml"
DEFAULT_WEIGHTS = "/mnt/d/Github/SuperpixelCUB/VLPart/checkpoints/r50_pascalpart.pth"
DEFAULT_VOCABULARY = "pascal_part"
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
# DEFAULT_INPUT_PATH = "/mnt/d/Github/SuperpixelCUB/CUB_200_data/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0003_796136.jpg"
DEFAULT_INPUT_PATH = "/mnt/d/Github/SuperpixelCUB/CUB_200_data/CUB_200_2011/images/031.Black_billed_Cuckoo/Black_Billed_Cuckoo_0092_795313.jpg"
DEFAULT_OUTPUT_PATH = "/mnt/d/Github/SuperpixelCUB/VLPart/output_images"


def setup_cfg():
    cfg = get_cfg()
    add_vlpart_config(cfg)
    cfg.merge_from_file(DEFAULT_CONFIG_FILE)
    cfg.MODEL.WEIGHTS = DEFAULT_WEIGHTS
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = DEFAULT_CONFIDENCE_THRESHOLD
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = DEFAULT_CONFIDENCE_THRESHOLD
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = DEFAULT_CONFIDENCE_THRESHOLD
    cfg.freeze()
    return cfg


def process_images(cfg):
    input_images = glob.glob(os.path.expanduser(DEFAULT_INPUT_PATH))
    assert input_images, "No input images found!"
    os.makedirs(DEFAULT_OUTPUT_PATH, exist_ok=True)

    demo = VisualizationDemo(cfg)
    for path in tqdm.tqdm(input_images):
        img = read_image(path, format="BGR")
        start_time = time.time()
        predictions, visualized_output = demo.run_on_image(img)
        print(
            "{}: {} in {:.2f}s".format(
                path,
                "detected {} instances".format(len(predictions["instances"]))
                if "instances" in predictions
                else "finished",
                time.time() - start_time,
            )
        )
        out_filename = os.path.join(DEFAULT_OUTPUT_PATH, os.path.basename(path))
        visualized_output.save(out_filename)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Running with default configuration")
    
    cfg = setup_cfg()
    process_images(cfg)
