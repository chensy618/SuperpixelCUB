import argparse
from collections import defaultdict
import glob
import multiprocessing as mp
from PIL import Image
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
from visualizer import _create_text_labels
import sys
sys.path.append('/mnt/d/Github/SuperpixelCUB/VLPart')
print(sys.path)
from vlpart.config import add_vlpart_config
from predictor import VisualizationDemo
from detectron2.data import MetadataCatalog

# Constants
WINDOW_NAME = "image demo"
DEFAULT_CONFIG_FILE = "/mnt/d/Github/SuperpixelCUB/VLPart/configs/pascal_part/r50_pascalpart.yaml"
DEFAULT_WEIGHTS = "/mnt/d/Github/SuperpixelCUB/VLPart/checkpoints/r50_pascalpart.pth"
DEFAULT_VOCABULARY = "pascal_part"
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
# DEFAULT_INPUT_PATH = "/mnt/d/Github/SuperpixelCUB/CUB_200_data/CUB_200_2011/images/176.Prairie_Warbler/Prairie_Warbler_0086_172534.jpg"
# DEFAULT_INPUT_PATH = '/mnt/d/Github/SuperpixelCUB/CUB_200_data/CUB_200_2011/images/080.Green_Kingfisher/Green_Kingfisher_0037_71113.jpg'
# DEFAULT_INPUT_PATH = '/mnt/d/Github/SuperpixelCUB/CUB_200_data/CUB_200_2011/images/175.Pine_Warbler/Pine_Warbler_0010_171239.jpg'
# DEFAULT_INPUT_PATH = '/mnt/d/Github/SuperpixelCUB/CUB_200_data/CUB_200_2011/images/078.Gray_Kingbird/Gray_Kingbird_0045_70256.jpg'
# DEFAULT_INPUT_PATH = "/mnt/d/Github/SuperpixelCUB/CUB_200_data/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0003_796136.jpg"
DEFAULT_INPUT_PATH = "/mnt/d/Github/SuperpixelCUB/CUB_200_data/CUB_200_2011/images/031.Black_billed_Cuckoo/Black_Billed_Cuckoo_0092_795313.jpg"
DEFAULT_OUTPUT_PATH = "/mnt/d/Github/SuperpixelCUB/VLPart/output_images"
CUSTOM_VOCABULARY =  "bird beak, bird head, bird eye, bird leg, bird foot, bird wing, bird neck, bird tail,bird body"
class_id_to_label = {
    0: "bird beak",
    1: "bird head",
    2: "bird eye",
    3: "bird leg",
    4: "bird foot",
    5: "bird wing",
    6: "bird neck",
    7: "bird tail",
    8: "bird body"
}
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
    # cfg.merge_from_file(args.config_file)
    cfg.merge_from_file(DEFAULT_CONFIG_FILE)
    # cfg.merge_from_list(args.opts)
    cfg.merge_from_list(['VIS.BOX', 'False'])
    # Set score_threshold for builtin models
    cfg.MODEL.WEIGHTS = DEFAULT_WEIGHTS
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = DEFAULT_CONFIDENCE_THRESHOLD
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = DEFAULT_CONFIDENCE_THRESHOLD
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = DEFAULT_CONFIDENCE_THRESHOLD
    cfg.freeze()
    return cfg

def refine_masks_by_semantics(semantic_groups, score_threshold=0.5):
    refined_masks = {}

    for label, mask_score_list in semantic_groups.items():
        # Sort masks by confidence descending
        sorted_masks = sorted(mask_score_list, key=lambda x: x[1], reverse=True)

        refined_mask = np.zeros_like(sorted_masks[0][0], dtype=np.uint8)
        confidence_map = np.zeros_like(refined_mask, dtype=np.float32)

        for mask, score in sorted_masks:
            if score < score_threshold:
                continue  # Skip low-confidence masks

            # Update only pixels with higher score than existing ones
            update_pixels = (mask & (score > confidence_map))
            refined_mask[update_pixels] = 1
            confidence_map[update_pixels] = score

        refined_masks[label] = refined_mask

    return refined_masks

def get_class_names():
    # Load the metadata for the dataset
    class_names_dict = {}
    metadata = MetadataCatalog.get("pascal_part_val")
    for i, name in enumerate(metadata.thing_classes):
        class_names_dict[i] = name
    print("class names:", class_names_dict)
    return class_names_dict

def generate_all_masks(predictions, visualized_output):
    # generate all semantic masks from the instance masks
    pred_masks = predictions["instances"].pred_masks.cpu().numpy()  # [N, H, W]
    pred_classes = predictions["instances"].pred_classes.cpu().numpy()  # [N]

    path = DEFAULT_INPUT_PATH
    input_img = read_image(path, format="BGR")
    height, width = input_img.shape[:2]
    semantic_mask = np.zeros((height, width), dtype=np.uint8)

    # use the current time as the base timestamp
    base_timestamp = str(int(time.time() * 1000))

    # save each instance mask separately
    for i, mask in enumerate(pred_masks):
        instance_mask = (mask * 255).astype(np.uint8)
        instance_class = pred_classes[i] + 1

        # output filename
        instance_filename = os.path.join(
            DEFAULT_OUTPUT_PATH, 
            f"{base_timestamp}_instance_{i}_class_{instance_class}.png"
        )
        Image.fromarray(instance_mask).save(instance_filename)

        # update semantic mask
        semantic_mask[mask == 1] = instance_class

    # save (semantic mask)
    semantic_filename = os.path.join(DEFAULT_OUTPUT_PATH, f"{base_timestamp}_semantic_mask.png")
    Image.fromarray(semantic_mask).save(semantic_filename)

    # visualization output
    vis_filename = os.path.join(DEFAULT_OUTPUT_PATH, f"{base_timestamp}_vis.jpg")
    visualized_output.save(vis_filename)
        
def process_images(cfg):
    input_images = glob.glob(os.path.expanduser(DEFAULT_INPUT_PATH))
    assert input_images, "No input images found!"
    os.makedirs(DEFAULT_OUTPUT_PATH, exist_ok=True)

    demo = VisualizationDemo(cfg)
    for path in tqdm.tqdm(input_images):
        img = read_image(path, format="BGR")
        start_time = time.time()
        predictions, visualized_output = demo.run_on_image(img)
        
        basename = os.path.splitext(os.path.basename(path))[0]

        # create a folder for the current image
        output_dir = os.path.join(DEFAULT_OUTPUT_PATH, basename)
        os.makedirs(output_dir, exist_ok=True)

        if "instances" in predictions:
            pred_instances = predictions["instances"]
            pred_masks = pred_instances.pred_masks.to("cpu").numpy()
            pred_classes = pred_instances.pred_classes.to("cpu").numpy()
            pred_scores = pred_instances.scores.to("cpu").numpy()
            
            semantic_groups = defaultdict(list)
            vocabulary_dict = get_class_names()
            
            for mask, cls, score in zip(pred_masks, pred_classes, pred_scores):
                pred_label = vocabulary_dict.get(cls, f"class_{cls}")
                semantic_groups[pred_label].append((mask, score))

            refined_masks = refine_masks_by_semantics(semantic_groups, score_threshold=0.55)

            # save refined masks separately per semantic class
            for label, refined_mask in refined_masks.items():
                mask_img = (refined_mask * 255).astype(np.uint8)
                mask_filename = os.path.join(output_dir, f"{basename}_{label}.png")
                cv2.imwrite(mask_filename, mask_img)

        # save the visualized output
        out_filename = os.path.join(output_dir, f"{basename}_vis.jpg")
        visualized_output.save(out_filename)
    # for path in tqdm.tqdm(input_images):
    #     img = read_image(path, format="BGR")
    #     start_time = time.time()
    #     predictions, visualized_output = demo.run_on_image(img)
    #     if "instances" in predictions:
    #         print(predictions["instances"].pred_classes)
    #         print(predictions["instances"].scores)
    #         print(predictions["instances"].pred_boxes)
    #         print(predictions["instances"].pred_masks)
    #         pred_instances = predictions["instances"]
    #         pred_masks = pred_instances.pred_masks.to("cpu").numpy()
    #         pred_classes = pred_instances.pred_classes.to("cpu").numpy()
    #         pred_scores = pred_instances.scores.to("cpu").numpy()
            
    #         # group by semantic class
    #         semantic_groups = defaultdict(list)
    #         vocabulary_dict = get_class_names()
    #         for mask, cls, score in zip(pred_masks, pred_classes, pred_scores):
    #             pred_label = vocabulary_dict.get(cls, f"class_{cls}")
    #             print("pred_label:", pred_label)
    #             semantic_groups[pred_label].append((mask, score))
    #         print("Semantic groups:", semantic_groups)
            
    #          # Refine masks
    #         refined_masks = refine_masks_by_semantics(semantic_groups, score_threshold=0.55)
            
    #         # Save refined masks separately per semantic class
    #         basename = os.path.splitext(os.path.basename(path))[0]
    #         print("basename:", basename)
    #         for label, refined_mask in refined_masks.items():
    #             mask_img = (refined_mask * 255).astype(np.uint8)
    #             mask_filename = os.path.join(DEFAULT_OUTPUT_PATH, f"{basename}_{label}.png")
    #             cv2.imwrite(mask_filename, mask_img)
    #     out_filename = os.path.join(DEFAULT_OUTPUT_PATH, os.path.basename(path))
    #     visualized_output.save(out_filename)


if __name__ == "__main__":

    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    # args.custom_vocabulary = CUSTOM_VOCABULARY
    # args.vocabulary = "custom"  
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    process_images(cfg)
