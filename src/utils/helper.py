import json, ast
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_json(json_output):
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "```json":
            json_output = "\n".join(lines[i+1:])
            json_output = json_output.split("```")[0]
            break
    # fix missing closing braces
    open_braces = json_output.count('{')
    close_braces = json_output.count('}')
    if open_braces > close_braces:
        json_output += '}' * (open_braces - close_braces)
    return ast.literal_eval(json_output)


def get_annotated_image(im, masks):
    im_array = np.array(im)
    combined_image = im_array.copy()

    combined_color_mask = np.zeros_like(combined_image)

    for i in range(len(masks)):
        mask = masks[i]
        mask_bool = mask.squeeze().astype(bool)
        combined_color_mask[mask_bool] = [255, 255, 0]  # Yellow 
    alpha = 0.6 
    combined_blended = cv2.addWeighted(combined_image, 1, combined_color_mask, alpha, 0)
    combined_blended_pil = Image.fromarray(combined_blended)
    return combined_blended_pil


def unnormalize_boxes(box, width, height):
    unnormalized_box = []
    
    abs_y1 = int(int(box[0])/1000 * height)
    abs_x1 = int(int(box[1])/1000 * width)
    abs_y2 = int(int(box[2])/1000 * height)
    abs_x2 = int(int(box[3])/1000 * width)

    if abs_x1 > abs_x2:
        abs_x1, abs_x2 = abs_x2, abs_x1
    if abs_y1 > abs_y2:
        abs_y1, abs_y2 = abs_y2, abs_y1

    unnormalized_boxes = [abs_x1, abs_y1, abs_x2, abs_y2]
    return unnormalized_boxes


def unnormalize_points(point, width, height):
    unnormalized_point = []
    abs_y = int(int(point[0])/1000 * height)
    abs_x = int(int(point[1])/1000 * width)
    unnormalized_point = [abs_y, abs_x]
    return unnormalized_point


def extract_supervisor_boxes_from_json_new(json_output, width, height):
    missing_suggested_boxes = []
    extra_existing_boxes = []
    missing_suggested_points = []
    candidate_present = False
    json_output = json.loads(json_output)
        
    for missing_obj in json_output['instances']['missing_objects']:
        if 'box_2d' in missing_obj:
            missing_id = missing_obj['missing_object_id']
            missing_box = missing_obj['box_2d']
            missing_suggested_boxes.append(unnormalize_boxes(missing_box, width, height))

    # check if there are any extra objects with existing bboxes
    for extra_obj in json_output['instances']['false_positives']:
        if 'box_2d' in extra_obj:
            extra_existing_boxes.append(unnormalize_boxes(extra_obj['box_2d'], width, height))

    if missing_suggested_boxes or extra_existing_boxes: # to trigger the validator
        candidate_present = True
    return candidate_present, missing_suggested_boxes, extra_existing_boxes


def get_annotated_boxes(image, boxes, color='blue'):
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)

    count = 1
    for box in boxes:
        text = f"m_{count}"
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
        draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline=color, width=1)
        text_bbox = draw.textbbox((box[0], box[1] - 15 ), text, font=font)
        draw.rectangle([text_bbox[0], text_bbox[1], text_bbox[2], text_bbox[3]], fill='white')
        draw.text((box[0], box[1] - 15), text=text, fill=color, font=font)
        count += 1   
    return annotated_image


def get_annotated_boxes_with_id(box_id, image, boxes, color='blue'):
    box = unnormalize_boxes(boxes, image.size[0], image.size[1])
    draw = ImageDraw.Draw(image)
    text = box_id
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline=color, width=1)
    text_bbox = draw.textbbox((box[0], box[1] - 15), text, font=font)
    draw.rectangle([text_bbox[0], text_bbox[1], text_bbox[2], text_bbox[3]], fill='white')
    draw.text((box[0], box[1] - 15), text=text, fill="red", font=font)
    return image
