import json
import time
import functools
import io
from typing import Tuple, Dict, Any
from PIL import Image
from google import genai
from google.genai import types
from google.genai.errors import ServerError
from typing import Tuple, Dict, Any, List
import torch

from src.utils.helper import (
    parse_json,
    extract_supervisor_boxes_from_json_new,
    get_annotated_boxes
)


def retry_on_503(max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            delay = base_delay
            while True:
                try:
                    return func(*args, **kwargs)
                except ServerError as e:
                    if getattr(e, 'code', None) == 503 and retries < max_retries:
                        time.sleep(delay)
                        delay = min(delay * 2, max_delay)
                        retries += 1
                        continue
                    raise
        return wrapper
    return decorator


def _pil_image_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class Supervisor:
    def __init__(self, policy, client, model: str, seed: int, device=None):
        self.client = client
        self.policy = policy
        self.model = model
        self.seed = seed
        self.device = device or ("cuda:1" if torch.cuda.is_available() else "cpu")
        print(f"Supervisor initialized on {self.device}")

        self.system_msg_eval = self._build_system_msg("supervisor_eval") 
        self.system_msg_boxgen = self._build_system_msg("supervisor_box_gen")

    def __del__(self):
        # cleanup method to free GPU memory
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'client'):
            del self.client
        with torch.cuda.device(self.device):
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    def _build_system_msg(self, prompt_type):
        guidelines_text = "\n".join(
            f"- {g['id']}: {g['relevant_text']}" for g in self.policy["guidelines"]
        )
        sup = self.policy["system_instructions"]["instructions"][prompt_type]

        return (
            "ROLE: " + prompt_type + "\n"
            + "\n".join(sup["role_description"]) + "\n\n"
            + "\n".join(sup["system_prompts"])   + "\n\n"
            + "# GUIDELINES:\n"  + guidelines_text + "\n\n"
            + "# EXAMPLES:\n```json\n"
            + json.dumps(sup["json_schema"], indent=2)
            + "\n```"
        )
    
    @retry_on_503()
    def critique(self, worker_boxes, worker_annotated_image, original_image, iter, image_name, cropnum) -> Tuple[bool, bool, str, Dict[str, Any], Image.Image]:
        notes = []
        if iter > 0:
            notes.append(
                "NOTE: Don't make up stuff about them being covered by the yellow mask when they are not. **Remember, boxes and masks are different. An object inside a box need not be covered in a mask.** You should be very careful about this."
            )
        notes.append("Important Hint for false positives- Ask yourself: `is this identified object really covered by yellow mask?'. If the answer is 'no', do not proceed with it.")
        
        worker_json = (
            "WORKER_OUTPUT_JSON:\n```json\n"
            + json.dumps(worker_boxes, indent=2)
            + "\n```"
        )
        
        text_block = "\n\n".join([
            "\n".join(notes),
            "\n".join(worker_json)
        ])
        
        buf_org = io.BytesIO()
        original_image.save(buf_org, format="PNG")
        org_bytes = buf_org.getvalue()

        buf_sup = io.BytesIO()
        worker_annotated_image.save(buf_sup, format="PNG")
        sup_bytes = buf_sup.getvalue()


        contents = [
            types.Part.from_text(text=text_block),
            types.Part.from_text(text="ORIGINAL_IMAGE:"),
            types.Part.from_bytes(data=org_bytes, mime_type="image/png"),
            types.Part.from_text(text="WORKER_ANNOTATED_IMAGE: Even though the boxes might cover an object, it might not be covered by yellow mask. Look carefully at the image and **only** the yellow mask."),
            types.Part.from_bytes(data=sup_bytes, mime_type="image/png"),
        ]

        response_schema = {
            "type": "object",
            "properties": {
                "missing_objects": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "missing_object_id": {
                                "type": "string",
                                "description": "Unique identifier for the missing object"
                            },
                            "label": {
                                "type": "string",
                                "description": "Object class label"
                            },
                            "reason": {
                                "type": "string",
                                "description": "Explanation for why this object is missing"
                            }
                        },
                        "required": ["missing_object_id", "label", "reason"],
                        "propertyOrdering": ["missing_object_id", "label", "reason"]
                    }
                },
                "false_positives": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "false_positive_id": {
                                "type": "string",
                                "description": "Unique identifier for the false positive"
                            },
                            "label": {
                                "type": "string",
                                "description": "Object class label"
                            },
                            "reason": {
                                "type": "string",
                                "description": "Explanation for why this is a false positive"
                            }
                        },
                        "required": ["false_positive_id", "label", "reason"],
                        "propertyOrdering": ["false_positive_id", "label", "reason"]
                    }
                },
                "refinements": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "box_id": {
                                "type": "string",
                                "description": "ID of the box that needs refinement"
                            },
                            "label": {
                                "type": "string",
                                "description": "Object class label"
                            },
                            "critique": {
                                "type": "string",
                                "description": "Description of the issue with current mask coverage"
                            },
                            "suggestion": {
                                "type": "string",
                                "description": "Suggested box adjustment to fix the issue"
                            }
                        },
                        "required": ["box_id", "label", "critique", "suggestion"],
                        "propertyOrdering": ["box_id", "label", "critique", "suggestion"]
                    }
                }
            },
            "required": ["missing_objects", "false_positives", "refinements"],
            "propertyOrdering": ["missing_objects", "false_positives", "refinements"]
        }

        # --- critique ---
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                # thinking_config=types.ThinkingConfig(thinking_budget=0),
                system_instruction=self.system_msg_eval,
                temperature=0.3,
                response_mime_type="application/json",
                response_schema=response_schema,
                # max_output_tokens=2048,
                seed=self.seed,
            ),
        )

        critique_output = response.text
        critique = parse_json(critique_output)

        missing = critique.get("missing_objects", [])
        false_pos = critique.get("false_positives", [])
        refinements = critique.get("refinements", [])

        # --- boxgen ---
        box_input = {"missing_objects": missing, "false_positives": false_pos}      
        contents = [
            types.Part.from_text(text="ORIGINAL_IMAGE:"),
            types.Part.from_bytes(data=org_bytes, mime_type="image/png"),
            types.Part.from_text(text="WORKER_ANNOTATED_IMAGE: with yellow mask"),
            types.Part.from_bytes(data=sup_bytes, mime_type="image/png"),
            types.Part.from_text(text=json.dumps(box_input))
        ]

        boxgen_response_schema = {
            "type": "object",
            "properties": {
                "instances": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "box_id": {
                                "type": "string",
                                "description": "Unique identifier matching missing_object_id or false_positive_id"
                            },
                            "label": {
                                "type": "string",
                                "description": "Object class label"
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
                        "required": ["box_id", "label", "box_2d"],
                        "propertyOrdering": ["box_id", "label", "box_2d"]
                    }
                }
            },
            "required": ["instances"],
            "propertyOrdering": ["instances"]
        }

        response_boxes = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                system_instruction=self.system_msg_boxgen,
                temperature=0.5,
                response_mime_type="application/json",
                response_schema=boxgen_response_schema,
                max_output_tokens=2048,
                seed=self.seed,
            )
        )
        boxes_json = parse_json(response_boxes.text).get("instances", [])

        # map boxes by id
        box_map = {inst["box_id"]: inst["box_2d"] for inst in boxes_json}

        # merge box_2d into each missing/false_positive entry
        merged_missing: List[Dict[str, Any]] = []
        for item in missing:
            obj_id = item.get("missing_object_id")
            bbox = box_map.get(obj_id, [0, 0, 0, 0])
            merged_missing.append({**item, "box_2d": bbox})

        merged_false: List[Dict[str, Any]] = []
        for item in false_pos:
            obj_id = item.get("false_positive_id")
            bbox = box_map.get(obj_id, [0, 0, 0, 0])
            if bbox != [0, 0, 0, 0]:
                merged_false.append({**item, "box_2d": bbox})

        # build final JSON
        final_json = {
            "instances": {
                "missing_objects": merged_missing,
                "false_positives": merged_false
            },
            "refinements": refinements
        }
        critique_json = json.dumps(final_json)
        critique_raw = "```json\n" + critique_json + "\n```"

        # --- annotate image ---
        width, height = original_image.size
        candidate_present, missing_suggested_boxes, extra_existing_boxes = extract_supervisor_boxes_from_json_new(critique_json, width, height)
        missing_objects = get_annotated_boxes(worker_annotated_image, missing_suggested_boxes, color='blue')
        extra_and_missing_objects_image = get_annotated_boxes(missing_objects, extra_existing_boxes, color='deeppink')

        candidate_present = bool(final_json.get("instances", {}).get("missing_objects") or final_json.get("instances", {}).get("false_positives"))
        refinement_present = bool(final_json.get("refinements"))

        return candidate_present, refinement_present, critique_raw, critique_json, extra_and_missing_objects_image
