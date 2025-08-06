import os
import json
import cv2
import torch
import numpy as np
from tqdm import tqdm
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputs, fast_rcnn_inference_single_image

# Load VG Classes
data_path = 'data/genome/1600-400-20'

vg_classes = []
with open(os.path.join(data_path, 'objects_vocab.txt')) as f:
    for obj in f.readlines():
        vg_classes.append(obj.split(',')[0].lower().strip())

vg_attrs = []
with open(os.path.join(data_path, 'attributes_vocab.txt')) as f:
    for attr in f.readlines():
        vg_attrs.append(attr.split(',')[0].lower().strip())

MetadataCatalog.get("vg").thing_classes = vg_classes
MetadataCatalog.get("vg").attr_classes = vg_attrs

# Load configuration
cfg = get_cfg()
cfg.merge_from_file("../configs/VG-Detection/faster_rcnn_R_101_C4_attr_caffemaxpool.yaml")
cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
cfg.MODEL.WEIGHTS = "http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe_attr.pkl"
predictor = DefaultPredictor(cfg)

NUM_OBJECTS = 36  # Number of objects to detect

@torch.no_grad()
def detect_objects_and_attributes(image_path):
    im = cv2.imread(image_path)
    raw_height, raw_width = im.shape[:2]
    
    # Preprocess image
    image = predictor.transform_gen.get_transform(im).apply_image(im)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    inputs = [{"image": image, "height": raw_height, "width": raw_width}]
    images = predictor.model.preprocess_image(inputs)
    
    # Backbone and proposals
    features = predictor.model.backbone(images.tensor)
    proposals, _ = predictor.model.proposal_generator(images, features, None)
    proposal_boxes = [x.proposal_boxes for x in proposals]
    features = [features[f] for f in predictor.model.roi_heads.in_features]
    box_features = predictor.model.roi_heads._shared_roi_transform(features, proposal_boxes)
    feature_pooled = box_features.mean(dim=[2, 3])

    # Predictions
    pred_class_logits, pred_attr_logits, pred_proposal_deltas = predictor.model.roi_heads.box_predictor(feature_pooled)
    outputs = FastRCNNOutputs(
        predictor.model.roi_heads.box2box_transform,
        pred_class_logits,
        pred_proposal_deltas,
        proposals,
        predictor.model.roi_heads.smooth_l1_beta,
    )
    probs = outputs.predict_probs()[0]
    boxes = outputs.predict_boxes()[0]
    
    attr_prob = pred_attr_logits[..., :-1].softmax(-1)
    max_attr_prob, max_attr_label = attr_prob.max(-1)
    
    # Apply NMS and filter top NUM_OBJECTS
    for nms_thresh in np.arange(0.5, 1.0, 0.1):
        instances, ids = fast_rcnn_inference_single_image(
            boxes, probs, image.shape[1:], 
            score_thresh=0.2, nms_thresh=nms_thresh, topk_per_image=NUM_OBJECTS
        )
        if len(ids) == NUM_OBJECTS:
            break

    instances = detector_postprocess(instances, raw_height, raw_width)
    instances.attr_scores = max_attr_prob[ids].detach()
    instances.attr_classes = max_attr_label[ids].detach()
    
    result = []
    
    for i in range(len(instances)):
        # object_data = {
        #     # "box": instances.pred_boxes[i].tensor.tolist(),
        #     "class": vg_classes[instances.pred_classes[i]],
        #     # "score": float(instances.scores[i])
        # }
        # attr_data = {
        #     "attribute": vg_attrs[instances.attr_classes[i]],
        #     # "attr_score": float(instances.attr_scores[i])
        # }
        # combined_data = {**attr_data, **object_data}
        # print([len(instances.pred_classes),len(instances.attr_classes),i])
        result.append(f'''{vg_attrs[instances.attr_classes[i]]} {vg_classes[instances.pred_classes[i]]}''')
    #实现去重
    # result = [dict(t) for t in {tuple(d.items()) for d in result}]
    # result = [dict(t) for t in {tuple(d.items()) for d in result}]
    seen = set()
    unique_result = []
    for d in result:
        t = d
        if t not in seen:
            seen.add(t)
            unique_result.append(d)
    result = unique_result

    return result

def process_images_in_folder(input_folder, output_file, top_k=NUM_OBJECTS):
    results = []  # 用于存储所有图片的检测结果

    # for image_filename in os.listdir(input_folder):
    
    # 获取文件夹内所有图片文件名
    image_filenames = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # 使用 tqdm 添加进度条
    for image_filename in tqdm(image_filenames, desc="Processing Images", unit="image"):
        if image_filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(input_folder, image_filename)
            print(f"Processing {image_path}...")
            
            detection_result = detect_objects_and_attributes(image_path)
            
            # 将当前图片的结果添加到总结果中
            results.append({
                'image_filename': image_filename,
                'detections': detection_result
            })
    
    # 将所有结果写入一个 JSON 文件
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
        f.flush()
    
    print(f"Detection completed. Results saved in {output_file}")


if __name__ == "__main__":
    input_folder = "data/images"  # Path to the folder containing input images
    output_folder = "output/detections.json"  # Path to the folder to save JSON files
    process_images_in_folder(input_folder, output_folder)
