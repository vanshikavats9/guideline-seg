import torch
from transformers import CLIPProcessor, CLIPModel
from transformers import CLIPVisionModel, CLIPTextModel, CLIPFeatureExtractor, CLIPTokenizer
from PIL import Image
import json
from src.utils.helper import parse_json
import torch.nn as nn
import torch.nn.functional as F
from src.utils.helper import unnormalize_boxes
from transformers import SiglipModel, SiglipProcessor, SiglipVisionModel
from transformers import AutoProcessor, AutoModel


class ObjectEmbedding(nn.Module):
    def __init__(self, clip_emb_dim, bbox_dim=4):
        super().__init__()
        self.bbox_mlp = nn.Linear(bbox_dim, clip_emb_dim)
    
    def forward(self, clip_embedding, bbox_coords):
        if bbox_coords.dim() == 1:
            bbox_coords = bbox_coords.unsqueeze(0)
        bbox_embedding = self.bbox_mlp(bbox_coords)
        unified_embedding = torch.stack([clip_embedding, bbox_embedding], dim=1)
        return unified_embedding
    


class ValidationModule:
    def __init__(self, device=None, attention_threshold=0.3):
        self.device = device or ("cuda:2" if torch.cuda.is_available() else "cpu")
        print(f"ValidationModule initialized on {self.device}")
        self.attention_threshold = attention_threshold

        self.siglip_model = SiglipModel.from_pretrained("google/siglip-base-patch16-224").to(self.device)
        self.siglip_processor = SiglipProcessor.from_pretrained("google/siglip-base-patch16-224")
        self.siglip_proj = torch.nn.Linear(768, 512).to(self.device)
        self.vision_model = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224").to(self.device)
        self.model = AutoModel.from_pretrained("google/siglip-base-patch16-224").to(self.device)
        self.processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

    def __del__(self):
        try:
            for attr in [
                'siglip_model', 'vision_model', 'model',
                'siglip_proj']:
                if hasattr(self, attr):
                    obj = getattr(self, attr)
                    if isinstance(obj, torch.nn.Module):
                        obj.to('cpu')
                    del obj
                    delattr(self, attr)
            torch.cuda.ipc_collect()
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
        except Exception as e:
            print('Validator cleanup error:', e)

    def _crop_image(self, image, bbox, width, height, iternum, flag):
        x1, y1, x2, y2 = bbox
        # calculate buffer (20% of width and height)
        w_buffer = int((x2 - x1) * 0.2)
        h_buffer = int((y2 - y1) * 0.2)

        if flag == "missing" or flag == "extra":
            w_buffer = int((x2 - x1) * 0.3)
            h_buffer = int((y2 - y1) * 0.3)
        
        # apply buffer while respecting image boundaries
        buffered_x1 = max(0, x1 - w_buffer)
        buffered_y1 = max(0, y1 - h_buffer)
        buffered_x2 = min(width, x2 + w_buffer)
        buffered_y2 = min(height, y2 + h_buffer)
   
        crop = image.crop((buffered_x1, buffered_y1, buffered_x2, buffered_y2))
        return crop

    # [SUBJECT] prepare image crops from bounding boxes with buffer area around them
    def prepare_buffer_crops_subject(self, image, boxes):  
        width, height = image.size
        buffer_crops = {}

        bounding_boxes_json = parse_json(boxes)
        bounding_boxes_json = json.dumps(bounding_boxes_json)
        boxes_data_readable = json.loads(bounding_boxes_json)["instances"]
        
        iternum = 0
        for box in boxes_data_readable:
            bbox = unnormalize_boxes(box["box_2d"], width, height)
            label = box["label"]
            box_id = box["id"]

            buffer_crops[box_id] = {
                "crop": self._crop_image(image, bbox, width, height, iternum, "subject"),
                "label": label,
                "box": bbox
            }
            iternum += 1
            
        return buffer_crops
    

    # [CANDIDATE] prepare image crops from bounding boxes with buffer area around them
    def prepare_buffer_crops_candidate(self, image, boxes):
        width, height = image.size
        buffer_crops = {}
        iternum = 0
        bounding_boxes_json = parse_json(boxes)
        bounding_boxes_json = json.dumps(bounding_boxes_json)
        boxes_data_readable = json.loads(bounding_boxes_json)["instances"]
        
        for obj in boxes_data_readable["missing_objects"]:
            box_id = obj["missing_object_id"]
            box = obj["box_2d"]
            label = obj["label"]

            bbox = unnormalize_boxes(box, width, height)
            buffer_crops[box_id] = {
                "crop": self._crop_image(image, bbox, width, height, iternum, "missing"),
                "label": label,
                "box": bbox
            }
            iternum += 1
            
        iternum = 0
        for obj in boxes_data_readable["false_positives"]:
            box_id = obj["false_positive_id"]
            box = obj["box_2d"]
            label = obj["label"]

            bbox = unnormalize_boxes(box, width, height)
            buffer_crops[box_id] = {
                "crop": self._crop_image(image, bbox, width, height, iternum, "extra"),
                "label": label,
                "box": bbox
            }
            iternum += 1
        return buffer_crops
    

     # compute cosine similarity between image and text
    def get_siglip_embeddings(self, image_list, unify_embeddings=False):       
            image_embeddings = {}

            for img_id in image_list:
                image_id = str(img_id)
                image_crop = image_list[image_id]["crop"]
                label = image_list[image_id]["label"]
                self.model = self.model.eval()

                # image embedding
                img_inputs = self.processor(images=image_crop, return_tensors="pt")
                img_inputs = {k: v.to(self.device) for k, v in img_inputs.items()}
                
                with torch.no_grad():
                    img_emb = self.model.get_image_features(**img_inputs)

                if unify_embeddings:
                    box_coords = torch.tensor(image_list[image_id]["box"], dtype=torch.float32, device=self.device)
                    img_emb = self.add_spatial_embedding(img_emb, box_coords)
                
                image_embeddings[image_id] = {
                    "embedding": img_emb,
                    "crop": image_crop,
                    "label": label
                }   
            return image_embeddings
    

    def add_spatial_embedding(self, clip_emb, box_coords):
        # add spatial embedding to the image crop
        embedder = ObjectEmbedding(clip_emb_dim=768).to(self.device)
        unified_emb = embedder(clip_emb, box_coords)
        return unified_emb
    
    
    def cosine_similarity(self, image_embeddings):
        similarity_scores = {}

        for image_id in image_embeddings:
            label_text = "This is a photo of " + image_embeddings[image_id]["label"] + "."
            negative_text = "This is a photo of cat."

            inputs = self.processor(text=[label_text, negative_text], images=image_embeddings[image_id]["crop"], padding="max_length", return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=-1)

            print(f"Image ID: {image_id}, Label: {image_embeddings[image_id]['label']}, Probs: {probs}")

            similarity_scores[image_id] = {
                "probs": probs.cpu().numpy(),
                "decision": 1 if probs.argmax(dim=-1) == 0 else 0
            }

        return similarity_scores
        
