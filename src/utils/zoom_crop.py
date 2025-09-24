import re
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection

# load OWLv2 model
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(device)

def parse_prompt(prompt):
    txt = re.sub(r'(?i)^detect\s+', '', prompt).strip()
    txt = re.sub(r'(?i)\s+in\s+(the\s+)?image$', '', txt).strip()
    return [p.strip() for p in re.split(r'\s+and\s+|,', txt) if p.strip()]

# run OWLv2 detection
def owlv2_detect(img_bgr, queries, threshold=0.3):
    image_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).convert("RGB")
    inputs = processor(images=image_pil, text=queries, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_object_detection(
        outputs=outputs,
        target_sizes=[image_pil.size[::-1]],
        threshold=threshold
    )[0]

    boxes = results["boxes"].to(torch.float32).cpu().numpy()
    labels = [queries[i] for i in results["labels"]]
    scores = results["scores"].to(torch.float32).cpu().numpy()
    return boxes, labels, scores



def get_focused_crop(img, prompt, scale=0.5, buffer_ratio=0.2, threshold=0.3, allow_fallback=True):
    H, W = img.shape[:2]
    queries = parse_prompt(prompt)

    # downsample image
    small_img = cv2.resize(img, (int(W * scale), int(H * scale)))
    small_h, small_w = small_img.shape[:2]

    boxes, labels, scores = owlv2_detect(small_img, queries, threshold)

    # filter out boxes with height less than 40 pixels in original scale
    filtered_indices = []
    for i, box in enumerate(boxes):
        height = (box[3] - box[1]) / scale
        width = (box[2] - box[0]) / scale
        if height >= 40 and width >= 15:
            filtered_indices.append(i)
    
    boxes = boxes[filtered_indices]
    labels = [labels[i] for i in filtered_indices]
    scores = scores[filtered_indices]

    if not len(boxes):
        # fallback to top scoring box
        if allow_fallback and len(scores) > 0:
            top_idx = np.argmax(scores)
            boxes = [boxes[top_idx]]
            labels = [labels[top_idx]]
        else:
            print("No objects matched. Using full image.")
            return img.copy(), (0, 0, W, H), None

    # rescale boxes back to original image with bounds checking
    boxes_orig = boxes / scale
    for i, box in enumerate(boxes_orig):
        box[0] = np.clip(box[0], 0, W)  # x1
        box[1] = np.clip(box[1], 0, H)  # y1
        box[2] = np.clip(box[2], 0, W)  # x2
        box[3] = np.clip(box[3], 0, H)  # y2
    boxes = boxes_orig

    # union boxes
    x1 = int(min(b[0] for b in boxes))
    y1 = int(min(b[1] for b in boxes))
    x2 = int(max(b[2] for b in boxes))
    y2 = int(max(b[3] for b in boxes))

    # add buffer
    bw = int((x2 - x1) * buffer_ratio)
    bh = int((y2 - y1) * buffer_ratio)
    x1b = max(0, x1 - bw)
    y1b = max(0, y1 - bh)
    x2b = min(W, x2 + bw)
    y2b = min(H, y2 + bh)

    # check if we need to split the crop
    crops = split_wide_crop_smart(img, boxes, (x1b, y1b, x2b, y2b), scale)

    if len(crops) == 1:
        # single crop case
        crop, coords = crops[0]
        return crop, coords, labels
    else:
        # multiple crops case: return list of (crop, coords) pairs and labels
        return crops, labels



def split_wide_crop_smart(img, boxes, crop_coords, scale,
                          min_width_ratio=0.5, edge_gap_ratio=0.1):
    x1, y1, x2, y2 = crop_coords
    W, orig_w = x2 - x1, img.shape[1]
    safety_margin = 10  # pixels

    # handle edge cases with boxes
    if boxes is None or len(boxes) == 0:
        return [(img[y1:y2, x1:x2], crop_coords)]
    
    # ensure boxes is a proper numpy array
    boxes = np.array(boxes)
    if boxes.ndim == 1:
        boxes = boxes.reshape(1, -1)
    if boxes.shape[0] == 0:
        return [(img[y1:y2, x1:x2], crop_coords)]

    
    # check if any part of the box overlaps with the crop area
    in_crop = np.zeros(len(boxes), dtype=bool)
    for i, box in enumerate(boxes):
        left, top, right, bottom = box
        # box overlaps if it's not entirely outside the crop area
        box_in_crop = not (right < x1 or  # box is entirely to the left
                          left > x2 or   # box is entirely to the right
                          bottom < y1 or  # box is entirely above
                          top > y2)      # box is entirely below
        in_crop[i] = box_in_crop
    
    # get filtered boxes
    b = boxes[in_crop]

    # ensure b is always a 2D array
    if not isinstance(b, np.ndarray) or b.ndim == 0:
        b = np.empty((0, 4))
    elif b.ndim == 1:
        b = b.reshape(1, -1) if b.size > 0 else np.empty((0, 4))

    # get number of objects
    num_objects = b.shape[0]
    
    # if no objects or just one, return original crop
    if num_objects <= 1:
        return [(img[y1:y2, x1:x2], crop_coords)]

    # sort objects by their center x-coordinate
    objects = []
    for box in b:
        l, t, r, bottom = box
        center = (l + r) / 2
        objects.append({
            'left': l,
            'right': r,
            'center': center,
            'width': r - l
        })
    objects.sort(key=lambda x: x['center'])

    # find the largest gap between objects
    max_gap = 0
    best_split = None
    best_left_count = 0

    # only consider splits that are within the middle 50% of the image
    valid_split_min = x1 + W * 0.25
    valid_split_max = x2 - W * 0.25

    for i in range(len(objects) - 1):
        current = objects[i]
        next_obj = objects[i + 1]
        
        gap_start = current['right']
        gap_end = next_obj['left']
        gap_size = gap_end - gap_start
        gap_center = (gap_start + gap_end) / 2

        # only consider gaps in the middle 50% of the image
        if gap_center < valid_split_min or gap_center > valid_split_max:
            continue

        if gap_size > max_gap:
            max_gap = gap_size
            best_split = gap_center
            best_left_count = i + 1

    if max_gap < 2 * safety_margin:
        return [(img[y1:y2, x1:x2], crop_coords)]

    xi = int(best_split)
    return [
        (img[y1:y2, x1:xi], (x1, y1, xi, y2)),
        (img[y1:y2, xi:x2], (xi, y1, x2, y2)),
    ]





