import torch

available_gpus = [0, 1, 2] 
gpu_memory_usage = {i: 0 for i in available_gpus}  

def get_least_used_gpu():
    return min(gpu_memory_usage, key=gpu_memory_usage.get)

def update_gpu_usage(gpu_id, delta):
    gpu_memory_usage[gpu_id] += delta


import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
import random
import io
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageColor
import torch, gc
import glob, os, shutil
from src_new.utils.helper import get_annotated_image
torch.cuda.empty_cache()
gc.collect()
from pathlib import Path
import time

# distribute models across GPUs
SAM_GPU = 0 
GEMMA_GPU = 1 
ST_GPU = 2 

print(f"Using GPUs: SAM on GPU {SAM_GPU}, Gemma on GPU {GEMMA_GPU}, SentenceTransformer on GPU {ST_GPU}")

from sam_model.lang_sam.models.sam import SAM
sam = SAM()
sam.build_model(sam_type="sam2.1_hiera_large", device=f"cuda:{SAM_GPU}")

from google import genai
from google.genai import types
client = genai.Client(api_key='your_api_key')
model = 'gemini-2.5-flash-preview-05-20'
print("model: ", model)


import json
from src.modules.worker import Worker
from src.modules.supervisor import Supervisor
from src.modules.validator import ValidationModule
from src.utils.zoom_crop import get_focused_crop
from src.modules.retriever_enricher_gemma import Enricher, Retriever

from src.utils.run_loggers import RunLogger
from src.utils.iter_controller import RefineAgent, crowd_bucket, STOP, CONTINUE, STEP_COST, FINAL_BONUS, EARLY_STOP_PENALTY

import time
import functools
from google.genai.errors import ServerError

# -----------------------------------------------------------------
def _issue_count(sup_json: dict) -> float:
    inst = sup_json.get("instances", {})
    miss = len(inst.get("missing_objects", []))
    extra = len(inst.get("false_positives", []))
    ref = len(sup_json.get("refinements", []))
    return miss + extra + 0.1 * ref


# -----------------------------------------------------------------
def clear_gpu_memory():
    gc.collect()
    for gpu_id in [SAM_GPU, GEMMA_GPU, ST_GPU]:
        with torch.cuda.device(f"cuda:{gpu_id}"):
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    print(f"Cleared memory on GPUs {SAM_GPU}, {GEMMA_GPU}, {ST_GPU}")


validator_global = ValidationModule(device=f"cuda:{ST_GPU}")

# load the policy for the zoomed image (enricher and retriever)
def load_scheme(zoomed_image):
    """
    Build the policy dict for a zoomed image
    """
    try:
        with open("instructions/system.json", "r") as f:
            system_instructions = json.load(f)

        scheme = {
            "system_instructions": system_instructions,
            "prompt": "detect pedestrians in the image"
        }

        # instantiate enricher and retriever
        enricher  = Enricher(device=f"cuda:{GEMMA_GPU}", st_device=f"cuda:{ST_GPU}")
        query_txt = enricher.build_query(zoomed_image, scheme["prompt"])
        retriever = Retriever(device=f"cuda:{ST_GPU}")
        scheme["guidelines"] = retriever.fetch(query_txt, k=8)

        retriever.cleanup()
        enricher._cleanup_if_needed()
        del retriever, enricher
        torch.cuda.empty_cache()

        return scheme
    except Exception as e:
        clear_gpu_memory()
        raise e

def retry_on_503(max_retries=1, base_delay=1.0, max_delay=60.0):
    """
    Decorator to retry functions on 503 errors with exponential backoff.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except ServerError as e:
                    # Check if 503 error
                    if "503" in str(e) and "overloaded" in str(e).lower():
                        last_exception = e
                        if attempt < max_retries:
                            # Calculate delay with exponential backoff
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

def main():

    print("starting...")
    print(f"GPU Memory Usage at start: {gpu_memory_usage}")
    
    ONLINE_LEARN     = True
    LOG_TRAJECTORY   = True
    MIN_ITERS        = 2
    MAX_ITERS        = 4
    STAGNATE_THRESH  = 0.5
    SEED = 1311

    # Parse CLI arguments
    parser = argparse.ArgumentParser(description='Process a folder for segmentation')
    parser.add_argument('--input_folder', type=str,
                        help='Path to the input folder')
    parser.add_argument('--output_folder', type=str,
                        help='Path to the output folder')
    args = parser.parse_args()

    save_folder_name = os.path.normpath(args.output_folder)
    SRC_ROOT_IMG = os.path.normpath(args.input_folder)
    OUT_ROOT = os.path.join(save_folder_name, "preds")
    LOG_ROOT = os.path.join(save_folder_name, "logs")

    rl_agent = RefineAgent(state_path=os.path.join(LOG_ROOT, "q_table.json"))
    run_logger = RunLogger(log_dir=Path(LOG_ROOT))


    pattern = os.path.join(SRC_ROOT_IMG, "*", 'images', "*", '*', "*.jpeg")
    image_folder = sorted(glob.glob(pattern, recursive=True))

    for i in range(len(image_folder)):
        if i % 5 == 0:
            clear_gpu_memory()
        try:
            image_path = image_folder[i]
            image_name = image_path.split("/")[-1].split(".")[0]
            split_path = image_path.split("/")

            out_dir = Path(os.path.join(OUT_ROOT, split_path[-5], split_path[-3], split_path[-2], image_name))

            print(f"\nProcessing image {i}: {image_folder[i]}")

            image = Image.open(image_path)
            org_image = image.copy()

            # create output save directory
            if os.path.exists(out_dir):
                shutil.rmtree(out_dir)
            os.makedirs(out_dir, exist_ok=True)
            os.makedirs(os.path.join(out_dir, "jsons"), exist_ok=True)
            os.makedirs(os.path.join(out_dir, "masks"), exist_ok=True)

            prompt = "detect pedestrians in the image"

            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            result = get_focused_crop(
                cv_image,
                prompt,
                scale=0.8, 
                buffer_ratio=1,
                threshold=0.15
            )


            if len(result) == 3:  # Single crop case
                crop, cropped_coords, resolved_class = result
                crops_to_process = [(crop, cropped_coords)]
            else:  # Multiple crops case
                crops_to_process, resolved_class = result


            all_masks = []
            all_coords = []
            all_masks_init = []

            # for each crop of the image
            for crop_idx, (crop, coords) in enumerate(crops_to_process):
                if crop is not None:
                    zoom = org_image.crop(coords)
                    zoomed_image = zoom # use zoomed crop
                else:
                    print(f"No crop generated for prompt '{prompt}'. Using full image.")
                    zoomed_image = image
                    coords = (0, 0, image.size[0], image.size[1])

                # load policy for the zoomed image
                policy = load_scheme(zoomed_image)
                policy_path = os.path.join(out_dir, "jsons", f"policy_crop_{crop_idx}.json")
                with open(policy_path, "w") as f:
                    json.dump(policy, f, indent=2)

                # instantiate all modules
                worker = Worker(policy, client, sam, model, SEED, device=f"cuda:{SAM_GPU}")
                supervisor = Supervisor(policy, client, model, SEED, device=f"cuda:{GEMMA_GPU}")
                validator = validator_global  # reuse the global validator

                org_zoomed_image = zoomed_image.copy()

                # --- Worker initial pass ---
                w_boxes, w_masks, w_anno_img, w_json = worker.run_segmentation(
                    zoomed_image, image_name, cropnum=crop_idx
                )
                # save the initial mask
                w_anno_img.save(f"{out_dir}/masks/crop_{crop_idx}_init.png", quality=95, dpi=(300, 300))

                if w_masks is not None:
                    all_masks_init.append(w_masks)
                else:
                    all_masks_init.append(np.zeros((image.size[1], image.size[0]), dtype=np.uint8))

                w_json_instances = json.loads(w_json)["instances"]
                initial_object_count = len(w_json_instances) if w_json_instances else 0
                # print(f"[{image_name} | crop {crop_idx}] Initial objects detected: {initial_object_count}")

                # =====  First supervisor critique  =====
                candidate_present, refinement_present, s_output, s_json, s_anno_img = supervisor.critique( #w_mask_255,
                    w_boxes, w_anno_img, org_zoomed_image, 0, image_name, cropnum=crop_idx
                )
                # open(os.path.join(out_dir, "jsons", f"supervisor_critique_{crop_idx}.jsonl"), "w").close()


                sup_json        = json.loads(s_json)
                issue_cnt_prev  = _issue_count(sup_json) # issue count before refinement
                tiny_steps = 0
                TOTAL_ITERS_USED = 0


                for iter_idx in range(1, MAX_ITERS + 1):
                    # print("iter: ", iter_idx)
                    # with open(os.path.join(out_dir, "jsons", f"supervisor_critique_{crop_idx}.jsonl"), "a") as f:
                    #     f.write(s_json)
                    #     f.write("\n")

                    if candidate_present:
                        action = CONTINUE
                        print(f"[iter{iter_idx}] cand->CONTINUE")
                    elif iter_idx <= MIN_ITERS:
                        action = CONTINUE
                        print(f"[iter{iter_idx}] safety->CONTINUE")
                    else:
                        action = rl_agent.act(issue_cnt_prev, initial_object_count, iter_idx)
                        print(f"[iter{iter_idx}] agent->{'CONTINUE' if action else 'STOP'}")
                    
                    
                    if action == CONTINUE:
                        TOTAL_ITERS_USED += 1

                    # STOP branch
                    if action == STOP and iter_idx > MIN_ITERS:
                        print(f"STOP at iter {iter_idx}")
                        # compute stop reward
                        stop_reward = -EARLY_STOP_PENALTY if issue_cnt_prev>0 else 0.0
                        if ONLINE_LEARN:
                            rl_agent.update(issue_cnt_prev, issue_cnt_prev,
                                            initial_object_count, STOP)
                        if LOG_TRAJECTORY:
                            run_logger.log_p2_inference(os.path.join(split_path[-5], split_path[-3], split_path[-2], image_name), crop_idx, iter_idx, STOP,
                                            issue_cnt_prev, issue_cnt_prev, stop_reward,
                                            initial_objects=initial_object_count)
                        break    
                    
                    
                    img_emb_subject = {}
                    img_emb_candidate = {}
                    # trigger the validator only if the candidate misisng/extra is present
                    if candidate_present:
                        crops_candidate = validator.prepare_buffer_crops_candidate(org_zoomed_image, s_output)                       
                        img_emb_candidate = validator.get_siglip_embeddings(crops_candidate, unify_embeddings=True)
                        similarity_scores_candidate = validator.cosine_similarity(img_emb_candidate)

                        # remove keys with similarity score less than 0.5
                        keys_to_remove = [key for key in similarity_scores_candidate if similarity_scores_candidate[key]['decision'] == 0]
                        for key in keys_to_remove:
                            img_emb_candidate.pop(key, None)

                        if refinement_present:
                            w_mask_255, w_boxes, w_json, w_anno_img, w_masks = worker.apply_supervisor_feedback(org_zoomed_image, 
                                                                                        w_json, s_json, s_anno_img, 
                                                                                        {}, img_emb_candidate, w_masks, 
                                                                                        candidate_present = True,
                                                                                        refinement_present = True,
                                                                                        iter = iter_idx,
                                                                                        image_name = image_name,
                                                                                        cropnum = crop_idx)
                            # worker_annotated_image.save(f"{out_dir}/masks/crop_{j}_iter_{iter}.png", quality=95, dpi=(300, 300))
                        else:
                            w_mask_255, w_boxes, w_json, w_anno_img, w_masks = worker.apply_supervisor_feedback(org_zoomed_image, 
                                                                                        w_json, s_json, s_anno_img, 
                                                                                        {}, img_emb_candidate, w_masks, 
                                                                                        candidate_present = True,
                                                                                        refinement_present = False,
                                                                                        iter = iter_idx,
                                                                                        image_name = image_name,
                                                                                        cropnum = crop_idx)
                            # worker_annotated_image.save(f"{out_dir}/masks/crop_{j}_iter_{iter}.png", quality=95, dpi=(300, 300))

                    elif refinement_present and not candidate_present:
                        w_mask_255, w_boxes, w_json, w_anno_img, w_masks = worker.apply_supervisor_feedback(org_zoomed_image, 
                                                                                        w_json, s_json, s_anno_img, 
                                                                                        {}, {}, w_masks, 
                                                                                        candidate_present = False,
                                                                                        refinement_present = True,
                                                                                        iter = iter_idx,
                                                                                        image_name = image_name,
                                                                                        cropnum = crop_idx)                  
                    w_anno_img.save(f"{out_dir}/masks/crop_{crop_idx}_refined.png", quality=95, dpi=(300, 300))


                    # ---- Supervisor re-evaluates ----
                    candidate_present, refinement_present, s_output, s_json, s_anno_img = supervisor.critique(
                        w_boxes, w_anno_img, org_zoomed_image, iter_idx,
                        image_name, cropnum=crop_idx
                    )
                      
                                    
                    sup_json = json.loads(s_json)
                    issue_cnt_new = _issue_count(sup_json)
                    print(f"issues {issue_cnt_prev}->{issue_cnt_new}")

                    # reward
                    delta = issue_cnt_prev - issue_cnt_new
                    reward = delta - STEP_COST
                    if issue_cnt_new == 0 and delta>0:
                        reward += FINAL_BONUS

                    # update & log
                    if ONLINE_LEARN:
                        rl_agent.update(issue_cnt_prev, issue_cnt_new,
                                        initial_object_count, CONTINUE)
                    if LOG_TRAJECTORY:
                        run_logger.log_p2_inference(os.path.join(split_path[-5], split_path[-3], split_path[-2], image_name), crop_idx, iter_idx,
                                        CONTINUE,
                                        issue_cnt_prev, issue_cnt_new,
                                        reward,
                                        initial_objects=initial_object_count)

                    if issue_cnt_new == 0 and not candidate_present:
                        break
                    tiny = (delta < STAGNATE_THRESH)
                    if tiny and tiny_steps>=1 and iter_idx>=MIN_ITERS and not candidate_present:
                        break
                    tiny_steps = tiny_steps+1 if tiny else 0

                    issue_cnt_prev = issue_cnt_new

                    print()
                    print()
                    print()
                
                run_logger.log_num_iters(os.path.join(split_path[-5], split_path[-3], split_path[-2], image_name), crop_idx, initial_object_count, TOTAL_ITERS_USED)
                # store masks and coordinates for this crop
                all_masks.append(w_masks)
                all_coords.append(coords)           
            final_mask_init = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)

            
            if all_masks_init and len(all_masks_init[0]) != 0:
                for masks, (x1, y1, x2, y2) in zip(all_masks_init, all_coords):
                    if len(masks) > 0:  # check if this specific crop's masks are not empty
                        cropped_mask = np.max(masks, axis=0).squeeze()
                        h, w = cropped_mask.shape
                        final_mask_init[y1:y1+h, x1:x1+w] = np.maximum(final_mask_init[y1:y1+h, x1:x1+w], cropped_mask)

            # save init mask
            mask_01_init = (final_mask_init > 0).astype(np.uint8)  # ensure mask is 0 or 1
            mask_img = Image.fromarray(mask_01_init)    # scale to 0-255 for PNG
            mask_img.save(f"{out_dir}/masks/init_mask.png", format="PNG")
 
            # merge all masks into the original image dimension
            final_mask = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)
            
            if all_masks and len(all_masks[0]) != 0:
                for masks, (x1, y1, x2, y2) in zip(all_masks, all_coords):
                    if len(masks) > 0:  # check if this specific crop's masks are not empty
                        cropped_mask = np.max(masks, axis=0).squeeze()
                        h, w = cropped_mask.shape
                        final_mask[y1:y1+h, x1:x1+w] = np.maximum(final_mask[y1:y1+h, x1:x1+w], cropped_mask)

            # save the final mask
            mask_01 = (final_mask > 0).astype(np.uint8)  # ensure mask is 0 or 1
            mask_img = Image.fromarray(mask_01)    # scale to 0-255 for PNG
            mask_img.save(f"{out_dir}/masks/final_mask.png", format="PNG")

            # create final annotated image
            final_image = get_annotated_image(org_image, [final_mask])
            final_image.save(f"{out_dir}/masks/final_annotated.png", quality=95, dpi=(300, 300))

            # clean memory regularly
            if i > 0 and i % 20 == 0:  # clean up every 20 iterations
                print(f"Performing periodic cleanup at iteration {i}")
                clear_gpu_memory()
                
            # clean up GPU memory
            del worker
            del supervisor
            clear_gpu_memory()
            
        except Exception as e:
            import traceback
            print(f"\nError processing image {i}: {str(e)}")
            # clean up on error too
            if 'worker' in locals(): del worker
            if 'supervisor' in locals(): del supervisor
            if 'validator' in locals(): del validator
            clear_gpu_memory()
            
            # log the failed file
            failed_log_path = f"{LOG_ROOT}/failed_files.txt"
            os.makedirs(os.path.dirname(failed_log_path), exist_ok=True)
            with open(failed_log_path, "a") as fail_log:
                fail_log.write(f"{image_path}\n")
            print("Full traceback:")
            print(traceback.format_exc())
            print("continuing to next image")
            continue


if __name__ == "__main__":
    main()
