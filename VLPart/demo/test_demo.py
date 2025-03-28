import argparse
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
DEFAULT_INPUT_PATH = "/mnt/d/Github/SuperpixelCUB/CUB_200_data/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0003_796136.jpg"
# DEFAULT_INPUT_PATH = "/mnt/d/Github/SuperpixelCUB/CUB_200_data/CUB_200_2011/images/031.Black_billed_Cuckoo/Black_Billed_Cuckoo_0092_795313.jpg"
DEFAULT_OUTPUT_PATH = "/mnt/d/Github/SuperpixelCUB/VLPart/output_images"
CUSTOM_VOCABULARY =  "bird beak, bird head, bird eye, bird leg, bird foot, bird wing, bird neck, bird tail,bird body"

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--vocabulary",
        default="pascal_part",
        choices=['pascal_part', 'partimagenet', 'paco',
                 'voc', 'coco', 'lvis',
                 'pascal_part_voc', 'lvis_paco', 'custom'],
        help="",
    )
    parser.add_argument(
        "--custom_vocabulary",
        default="",
        help="",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )

    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

# def setup_cfg():
#     cfg = get_cfg()
#     add_vlpart_config(cfg)
#     cfg.merge_from_file(DEFAULT_CONFIG_FILE)
#     # cfg.custom_vocabulary = CUSTOM_VOCABULARY
#     # cfg.vocabulary = "custom"
#     cfg.MODEL.WEIGHTS = DEFAULT_WEIGHTS
#     cfg.MODEL.RETINANET.SCORE_THRESH_TEST = DEFAULT_CONFIDENCE_THRESHOLD
#     cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = DEFAULT_CONFIDENCE_THRESHOLD
#     cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = DEFAULT_CONFIDENCE_THRESHOLD
#     cfg.freeze()
#     return cfg
def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_vlpart_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
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
    args = get_parser().parse_args()
    args.custom_vocabulary = CUSTOM_VOCABULARY
    args.vocabulary = "custom"  
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    process_images(cfg)
