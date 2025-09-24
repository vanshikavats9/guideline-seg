import json
import numpy as np
from google import genai
from google.genai import types
from src.utils.helper import parse_json, get_annotated_image, unnormalize_boxes, get_annotated_boxes_with_id
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_fill_holes
import io
import time
import functools
from google.genai.errors import ServerError
import random
import torch

def retry_on_503(max_retries=1, base_delay=1.0, max_delay=60.0):
    # decorator to retry functions on 503 errors with exponential backoff
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except ServerError as e:
                    # check if it's a 503 error
                    if "503" in str(e) and "overloaded" in str(e).lower():
                        last_exception = e
                        if attempt < max_retries:
                            # calculate delay with exponential backoff
                            delay = min(base_delay * (2 ** attempt), max_delay)
                            print(f"503 error encountered. Retrying in {delay:.1f} seconds... (attempt {attempt + 1}/{max_retries + 1})")
                            time.sleep(delay)
                            continue
                        else:
                            print(f"Max retries ({max_retries}) exceeded for 503 error. Giving up.")
                    else:
                        raise
                except Exception as e:
                    raise
            
            # exhausted all retries
            raise last_exception
            
        return wrapper
    return decorator

def ensure_string(text_or_list):
    if isinstance(text_or_list, list):
        return "\n".join(text_or_list)
    return text_or_list

# --- Worker ---
class Worker:
    def __init__(self, policy, client, sam, model, seed, device=None):
        self.policy = policy
        self.client = client
        self.sam = sam
        self.model = model
        self.seed = seed
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Worker initialized on {self.device}")

    def __del__(self):
        # cleanup method to free GPU memory
        if hasattr(self, 'sam'):
            del self.sam
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'client'):
            del self.client
        with torch.cuda.device(self.device):
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    # produce bounding boxes for an image using a given prompt
    @retry_on_503(max_retries=1, base_delay=1.0, max_delay=60.0)
    def produce_bounding_boxes(self, image, image_name, cropnum=0):
        guidelines = self.policy["guidelines"]
        guidelines_text = "\n".join(
            f"- {g['id']}: {g['description']}" for g in guidelines
        )
        role_description = ensure_string(self.policy["system_instructions"]["instructions"]["worker_init"]["role_description"])
        system_prompts = ensure_string(self.policy["system_instructions"]["instructions"]["worker_init"]["system_prompts"])

        bounding_box_system_instructions = "\n\n".join([
            role_description,
            system_prompts,
        ])

        # convert PIL image to bytes
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        input_prompt = f"Detect pedestrians in the image. \n {guidelines_text}"

        contents = [
            types.Part.from_text(text="INPUT_IMAGE:\n"),
            types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
            types.Part.from_text(text="USER_PROMPT:\n" + input_prompt)
        ]

        # produce bounding boxes
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                system_instruction=bounding_box_system_instructions,
                temperature=0.5,
                max_output_tokens=4096,
                seed=self.seed,
                response_mime_type="application/json",
            ),
        )

        bounding_boxes_raw = response.text
        bounding_boxes_json = parse_json(response.text)

        # transform the response to the expected format
        if isinstance(bounding_boxes_json, list):
            transformed_json = {
                "instances": []
            }
            for idx, item in enumerate(bounding_boxes_json):
                transformed_item = {
                    "id": f"sub_{idx}",
                    "label": item["label"],
                    "box_2d": item["box_2d"]
                }
                transformed_json["instances"].append(transformed_item)
            bounding_boxes_json = transformed_json
        elif isinstance(bounding_boxes_json, dict) and "instances" not in bounding_boxes_json:
            instances = []
            for key, value in bounding_boxes_json.items():
                if isinstance(value, dict) and "box_2d" in value:
                    instances.append({
                        "id": f"sub_{len(instances)}",
                        "label": value["label"],
                        "box_2d": value["box_2d"]
                    })
            if instances:
                bounding_boxes_json = {"instances": instances}
       

        bounding_boxes_json = json.dumps(bounding_boxes_json)
        bounding_boxes_raw = f"```json\n{bounding_boxes_json}\n```"
        
        return bounding_boxes_raw, bounding_boxes_json
    


    # run SAM segmentation on image using the bounding boxes produced by the worker
    def run_segmentation(self, image, image_name, cropnum=0):
        org_image = image.copy()
        width, height = image.size
        bounding_boxes_raw, bounding_boxes_json = self.produce_bounding_boxes(image, image_name, cropnum)
        sam_boxes = []
        draw = ImageDraw.Draw(image)

        boxes_data = json.loads(bounding_boxes_json)["instances"]

        for i, bbox in enumerate(boxes_data):    
            box = bbox["box_2d"]
            # convert normalized coordinates to absolute coordinates
            abs_y1 = int(int(box[0])/1000 * height)
            abs_x1 = int(int(box[1])/1000 * width)
            abs_y2 = int(int(box[2])/1000 * height)
            abs_x2 = int(int(box[3])/1000 * width)

            if abs_x1 > abs_x2:
                abs_x1, abs_x2 = abs_x2, abs_x1
            if abs_y1 > abs_y2:
                abs_y1, abs_y2 = abs_y2, abs_y1

            color = random.choice(["red", "cyan", "lime", "magenta"])
            draw.rectangle(
                    ((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=1
                )
            if "label" in bbox:
                text = f"sub_{i}"
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
                text_bbox = draw.textbbox((abs_x1, abs_y1 - 15 ), text, font=font)
                draw.rectangle([text_bbox[0], text_bbox[1], text_bbox[2], text_bbox[3]], fill='white')
                draw.text((abs_x1, abs_y1 - 15), text=text, fill="red", font=font)

            sam_boxes.append([abs_x1, abs_y1, abs_x2, abs_y2])

        masks, _, _ = self.sam.predict(org_image, xyxy=sam_boxes)
        annotated_image = get_annotated_image(image, masks)

        return bounding_boxes_raw, masks, annotated_image, bounding_boxes_json
    


    # apply supervisor feedback to the bounding boxes
    def apply_supervisor_feedback(self, 
                                  image, 
                                  worker_boxes, 
                                  supervisor_output, supervisor_annotated_image, 
                                  img_emb_subject, img_emb_candidate, 
                                  worker_masks, 
                                  candidate_present, refinement_present, 
                                  iter, 
                                  image_name,
                                  cropnum):

        org_image = image.copy()
        annotated_image = image.copy()
        worker_boxes = json.loads(worker_boxes)
        
        # ensure worker_masks is a numpy array for proper indexing
        if not isinstance(worker_masks, np.ndarray):
            worker_masks = np.array(worker_masks)

        count_errors = 0
        
        supervisor_json = json.loads(supervisor_output)
        supervisor_boxes = supervisor_json["instances"]
        sam_boxes = []

        

        if refinement_present:
            # handle refinement suggestions if present
            if "refinements" in supervisor_json and supervisor_json["refinements"]:
                print("REFINEMENT TRUE")
                count_errors += len(supervisor_json["refinements"])

                bounding_box_system_instructions = (
                    "ROLE DESCRIPTION: "
                    + ensure_string(self.policy["system_instructions"]["instructions"]["worker_subsequent"]["role_description"])
                    + "\n\n# SYSTEM PROMPTS: "
                    + ensure_string(self.policy["system_instructions"]["instructions"]["worker_subsequent"]["system_prompts"])
                    + "\n```"
                )
                
                buf1 = io.BytesIO()
                org_image.save(buf1, format="JPEG")
                org_bytes = buf1.getvalue()

                buf2 = io.BytesIO()
                supervisor_annotated_image.save(buf2, format="JPEG")
                sup_bytes = buf2.getvalue()                 

                user_text = "\n".join([
                    "**NOTE: apply the refinements to fix the boxes.**",
                    "WORKER PREVIOUS BOXES:",
                    "```json\n" + json.dumps(worker_boxes, indent=2) + "\n```",
                    "REFINEMENTS:",
                    "```json\n" + json.dumps(supervisor_json["refinements"], indent=2) + "\n```"
                ])

                response_schema = {
                    "type": "object",
                    "properties": {
                        "updated_instances": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {
                                        "type": "string",
                                        "description": "Unique identifier for the detected object"
                                    },
                                    "label": {
                                        "type": "string", 
                                        "description": "Object class label (e.g. m_1, e_1, e_sub_2, etc)"
                                    },
                                    "box_2d": {
                                        "type": "array",
                                        "items": {
                                            "type": "number"
                                        },
                                        "minItems": 4,
                                        "maxItems": 4,
                                        "description": "Bounding box coordinates [ymin, xmin, ymax, xmax]"
                                    }
                                },
                                "required": ["id", "label", "box_2d"],
                                "propertyOrdering": ["id", "label", "box_2d"]
                            }
                        }
                    },
                    "required": ["updated_instances"],
                    "propertyOrdering": ["updated_instances"]
                }


                contents = [
                    types.Part.from_text(text="ORIGINAL_IMAGE:"),
                    types.Part.from_bytes(data=org_bytes, mime_type="image/jpeg"),
                    types.Part.from_text(text="ANNOTATED_IMAGE:"),
                    types.Part.from_bytes(data=sup_bytes,mime_type="image/jpeg"),
                    types.Part.from_text(text=user_text),
                ]

                response = self._make_refinement_api_call(
                    model=self.model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=bounding_box_system_instructions,
                        temperature=0.5,
                        seed=self.seed,
                        max_output_tokens=2048,
                        response_mime_type="application/json",
                        response_schema=response_schema,
                        thinking_config=types.ThinkingConfig(thinking_budget=0)
                        
                    ),
                )

                refined_boxes_raw = response.text
                refined_boxes_json = parse_json(refined_boxes_raw)
                refined_boxes_json = json.dumps(refined_boxes_json)
                refined_boxes_json = json.loads(refined_boxes_json)
                
                updated_boxes = {}

                for box in refined_boxes_json["updated_instances"]:
                    updated_boxes[box["id"]] = box
                
                # update existing boxes and add new ones to worker_boxes
                for box_id, box_data in updated_boxes.items():
                    # find and update box if it exists in worker_boxes
                    for instance in worker_boxes["instances"]:
                        if instance["id"] == box_id:
                            instance["box_2d"] = box_data["box_2d"]
                            sam_boxes.append(unnormalize_boxes(box_data["box_2d"], image.size[0], image.size[1]))
                            break

        if candidate_present:
            extra_obj_flag = False
            # apply the accepted supervisor candidates to the annotations
            for key in img_emb_candidate.keys():
                if key.startswith("m_"):
                    # get box from missing objects
                    for missing_obj in supervisor_boxes["missing_objects"]:
                        if missing_obj["missing_object_id"] == key:
                            count_errors += 1
                            box = missing_obj["box_2d"]
                            sam_box = unnormalize_boxes(box, image.size[0], image.size[1])
                            # crop the image to the sam_box and save
                            x1, y1, x2, y2 = sam_box
                            cropped_img = image.crop((x1, y1, x2, y2))
                            width = x2-x1
                            height = y2-y1
                            x1 = int(max(0, x1-width*0.05))
                            y1 = int(max(0, y1-height*0.05))
                            x2 = int(min(image.size[0], x2+width*0.05))
                            y2 = int(min(image.size[1], y2+height*0.05))
                            # crop_save_path = f"crop_{key}.png"
                            # cropped_img.save(crop_save_path, quality=95, dpi=(300, 300))
                            sam_boxes.append(sam_box)

        if len(sam_boxes) > 0:
            print("SAM boxes TRUE")
            print(sam_boxes)
            masks, _, _ = self.sam.predict(org_image, xyxy=sam_boxes)
            worker_masks = np.max(worker_masks, axis=0, keepdims=True)
            all_masks = np.concatenate([worker_masks, masks], axis=0)

        else:
            all_masks = worker_masks

        annotated_image = get_annotated_image(org_image, all_masks)

        # fill holes in the mask
        if all_masks is not None:
            all_masks = np.array([binary_fill_holes(mask).astype(np.uint8) for mask in all_masks])

    
        extra_obj_flag = False
        # remove false positives from worker boxes
        if candidate_present:
            for key in img_emb_candidate.keys():
                worker_masks = all_masks
                if key.startswith("e_"):
                    print("EXTRA DETECTED")
                    # get box from extra objects
                    for extra_obj in supervisor_boxes["false_positives"]:
                        if extra_obj["false_positive_id"] == key:
                            count_errors += 1
                            extra_obj_flag = True
                            unnormalized_extra_box = extra_obj["box_2d"]
                            box = unnormalize_boxes(extra_obj["box_2d"], image.size[0], image.size[1])
                            x1, y1, x2, y2 = box
                            width = x2-x1
                            height = y2-y1
                            x1 = int(max(0, x1-width*0.08))
                            y1 = int(max(0, y1-height*0.08))
                            x2 = int(min(image.size[0], x2+width*0.08))
                            y2 = int(min(image.size[1], y2+height*0.08))

                            # save img
                            # cropped_img = image.crop((x1, y1, x2, y2))
                            # crop_save_path = f"crop_{key}.png"
                            # cropped_img.save(crop_save_path, quality=95, dpi=(300, 300))
                            
                            # remove from mask 
                            temporary_mask = worker_masks[:, :, y1:y2, x1:x2] # slice all masks in the region
                            temporary_image = org_image.copy().crop((x1, y1, x2, y2))  # use crop method for PIL Image

                            mask_extra_arr = []

                            for i in range(temporary_mask.shape[0]):
                                single = temporary_mask[i,0]         
                                inds  = np.argwhere(single==1)
                                # print(inds)
                                if inds.size:
                                    yx = inds.mean(axis=0).astype(int)         
                                    neg  = np.array([[yx[1], yx[0]]])          

                                    if single[yx[0], yx[1]] == 1:
                                        print("mean neg", neg)
                                    else:
                                        # fallback to box center
                                        neg = np.array([[ (x1+x2)//2, (y1+y2)//2 ]])
                                        print("fallback center neg", neg)

                                    neg_label = np.array([0])
                                    mask_extra, _, _ = self.sam.predict_extra(temporary_image, single[None], neg_points=neg, neg_label=neg_label)
                                    mask_extra_arr.append(mask_extra)
                                else:
                                    mask_extra_arr.append(np.zeros((1, temporary_image.size[1], temporary_image.size[0])))

                            mask_extra_arr = np.array(mask_extra_arr)

                            # # temp_mask_path = f"old_mask.png"
                            # mask2d = np.max(temporary_mask, axis=0).squeeze()
                            # # print(mask2d.shape)
                            # mask_uint8 = (mask2d * 255).astype('uint8') 
                            # Image.fromarray(mask_uint8).save(temp_mask_path)

                            # new_mask_path = f"new_mask.png"
                            # mask_extra_arr = np.max(mask_extra_arr, axis=0).squeeze()
                            # # print(mask_extra_arr.shape)
                            # mask_uint8 = (mask_extra_arr * 255).astype('uint8') 
                            # Image.fromarray(mask_uint8).save(new_mask_path)

                            worker_masks[:, :, y1:y2, x1:x2] = mask_extra_arr
                       
                            annotated_image = get_annotated_image(org_image, worker_masks)
                            key_extra = key

                            if key.startswith("e_sub") or key.startswith("e_m") or key.startswith("e_cand"):
                                key_to_remove = key[2:]
                                # Find the index of the item to remove
                                for i, instance in enumerate(worker_boxes["instances"]):
                                    if instance["id"] == key_to_remove:
                                        del worker_boxes["instances"][i]
                                        break
                            all_masks = worker_masks


        
        
        for instance in worker_boxes["instances"]:
                    box_id = instance["id"]
                    color = random.choice(["red", "cyan", "lime", "magenta"])
                    if box_id.startswith("sub_"):
                        annotated_image = get_annotated_boxes_with_id(box_id, annotated_image, instance["box_2d"], color=color)
        #             if box_id.startswith("m_") or box_id.startswith("cand_"):
        #                 annotated_image = get_annotated_boxes_with_id(box_id, annotated_image, instance["box_2d"], color='blue')
                    # elif box_id.startswith("e_"):
                    #     annotated_image = get_annotated_boxes_with_id(box_id, annotated_image, instance["box_2d"], color='deeppink')


        '''Do it only for visualizing'''
        # if extra_obj_flag:
        #     annotated_image = get_annotated_boxes_with_id(key_extra, annotated_image, unnormalized_extra_box, color='deeppink')
        
        worker_boxes_json = json.dumps(worker_boxes)
        worker_boxes_raw = "```json\n" + worker_boxes_json + "\n```"

        # merge all masks into a single mask (0/1) for the whole image
        if all_masks is not None and len(all_masks) > 0:
            merged_mask = np.max(all_masks, axis=0).squeeze()
            mask_01 = (merged_mask > 0).astype(np.uint8)
            mask_255 = (mask_01 * 255).astype(np.uint8)
            w_mask_255 = Image.fromarray(mask_255)
        else:
            # fallback: create an empty mask with the same size as org_image
            w_mask_255 = Image.fromarray(np.zeros((org_image.size[1], org_image.size[0]), dtype=np.uint8))

        return w_mask_255, worker_boxes_raw, worker_boxes_json, annotated_image, all_masks

    @retry_on_503(max_retries=1, base_delay=1.0, max_delay=60.0)
    def _make_refinement_api_call(self, model, contents, config):
        return self.client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )



        