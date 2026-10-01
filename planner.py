import openai
from openai import OpenAI
import base64
from io import BytesIO
import time
from typing import Optional
import string
import re
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


# Put your VLM API key and url here (for a local vLLM server, e.g. api_key="EMPTY", base_url="http://127.0.0.1:8000/v1")
client_QwenAPI = OpenAI(
    api_key="xxxxxxxx",
    base_url="xxxxxxx",
    timeout=300.0
)


def format_content_openai(contents):
    formated_content = []
    for c in contents:
        formated_content.append({"type": "text", "text": c[0]})
        if len(c) == 2:
            formated_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{c[1]}", "detail": "high",},})
    return formated_content


def call_openai_api(sys_prompt, contents, model) -> Optional[str]:
    """Raises RuntimeError when the endpoint keeps failing: results computed with fallback decisions are invalid."""
    max_tries, retry_count, last_err = 3, 0, None
    while retry_count < max_tries:
        try:
            formated_content = format_content_openai(contents)
            message_text = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": formated_content},
            ]
            completion = client_QwenAPI.chat.completions.create(
                model=model,
                messages=message_text,
                max_tokens=64,
                temperature=0.0, top_p=0.9,
            )
            return completion.choices[0].message.content
        except openai.RateLimitError as e:
            print("Rate limit error, waiting for 3s")
            last_err = e
        except Exception as e:
            print("Error: ", e)
            last_err = e
        time.sleep(3)
        retry_count += 1

    raise RuntimeError(f"VLM planner unavailable (model={model}): {type(last_err).__name__}: {last_err}") from last_err


def check_planner(model):
    """Query the VLM once before loading the simulator, so a wrong key, URL or model name fails immediately."""
    call_openai_api("You are a helpful assistant.", [("Reply with OK.",)], model=model)
    print(f"VLM planner {model} is reachable.")

def resize_max_side(img, max_side=480):
    img_pil = Image.fromarray(img)
    w, h = img_pil.size
    scale = min(max_side / w, max_side / h, 1.0)
    if scale < 1.0:
        img_pil = img_pil.resize(
            (int(w * scale), int(h * scale)), Image.BILINEAR
        )
    return np.array(img_pil)

def encode_tensor2base64(img):
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    img = resize_max_side(img, 448)
    img = Image.fromarray(img)

    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    return img_base64

def call_planner(instruction, images, model="qwen2.5-vl-7b-instruct", temporal=True):
    if temporal:
        sys_prompt = (
            "Role: You are an autonomous agent navigating an indoor environment.\n"
            "Task: You observe a sequence of images representing past and current views of the scene.\n"
            "Goal: Based only on the provided images and the Instruction, select EXACTLY ONE image that should be used as the NEXT navigation reference to find the target.\n"
            "Constraints:\n"
            "- Use only visible evidence.\n"
            "- Do NOT assume or invent unseen objects or attributes.\n"
            "- The decision must be grounded in visual evidence and spatial reasoning.\n"
        )

        content = []
        text = f"Instruction: {instruction}\n\n"
        content.append((text,))
        text = (
            "The following are candidate images ordered temporally from earlier to later (more recent).\n"
            "Choose exactly ONE option that should be used as the next navigation reference point for finding the target.\n\n"

            "Rules:\n"
            "1) Extract ALL constraints from the Instruction:\n"
            "   - Target object + required attributes (color/pattern/material/size/state).\n"
            "   - Target place: room/area + any disambiguator for that place (e.g., 'the bedroom that has ...').\n"
            "   - Reference objects + spatial relations (e.g., 'below the elk herd painting', 'next to', 'under').\n"
            "   - Uniqueness/exclusions/order (only / nearest / middle).\n"
            
            "2) For each option, decide for each constraint: SUPPORTED / UNKNOWN (not visible) / CONTRADICTED.\n"
            "   - If any constraint is CONTRADICTED, do NOT select that option.\n"
            "     Example: required 'purple floral-quilted bed' but the only visible bed is clearly blue-striped.\n\n"

            "3) If the target is visible, choose the option that best matches ALL constraints.\n"
                "   If the target is not visible, choose the option that best supports the correct place and related cues, "
                "   and does not contradict any constraint.\n"
                "   If all options are weak/uncertain, choose the MOST RECENT option to continue exploration (lower step cost).\n"
        )
        content.append((text,))

    else:
        sys_prompt = (
            "Role: You are an autonomous agent navigating an indoor environment.\n"
            "Task: You are at a fixed location and observe multiple views from different directions.\n"
            "Goal: Based only on the provided images and the Instruction, select EXACTLY ONE direction/image to move NEXT to find the target.\n"
            "Constraints:\n"
            "- Use only visible evidence.\n"
            "- Do NOT assume or invent unseen objects or attributes.\n"
            "- The decision must be grounded in visual evidence and spatial reasoning.\n"
        )

        content = []
        text = f"Instruction: {instruction}\n\n"
        content.append((text,))
        text = (
            "The following images show different viewing directions from the SAME position.\n"
            "Choose exactly ONE option to move NEXT to find the target.\n\n"

            "Rules:\n"
            "1) Extract ALL constraints from the Instruction:\n"
            "   - Target object + required attributes (color/pattern/material/size/state).\n"
            "   - Target place: room/area + any disambiguator for that place.\n"
            "   - Reference objects + spatial relations.\n"
            "   - Uniqueness/exclusions/order if present.\n"

            "2) For each option, decide for each constraint: SUPPORTED / UNKNOWN / CONTRADICTED.\n"
            "   - If any constraint is CONTRADICTED, do NOT select that option.\n"
            "     Example: required 'purple floral-quilted bed' but the only visible bed is blue-striped.\n\n"

            "3) If the target is visible, choose the option that best matches ALL constraints.\n"
            "   If not visible, choose the option most consistent with the correct place and without contradiction.\n"
        )
        content.append((text,))


    labels = list(string.ascii_uppercase)
    assert len(images) <= len(labels)
    for i, rgb in enumerate(images):
        content.append((f"Option {labels[i]}: ", encode_tensor2base64(rgb)))
        content.append((f"End Option {labels[i]}.",))
    text = (
        "Answer Format (must follow exactly):\n"
        "Line 1: Option <LETTER>\n"
        "Line 2: A clear reason for this choice in no more than 30 words.\n\n"
        "Rules:\n"
        "- <LETTER> must be one of the valid option labels shown above (e.g., A, B, C, ...).\n"
        "- The first line must be EXACTLY in the form: Option <LETTER>.\n"
        "- Do NOT add any other text, symbols, or words on the first line.\n"
        "- The explanation must start on the second line.\n"
    )
    content.append((text,))

    # call LLM
    retry_bound = 5
    final_response, final_reason = None, None
    for _ in range(retry_bound):
        response = call_openai_api(sys_prompt, content, model=model)
        if response is None:
            print("call_openai_api returns None, retrying")
            continue

        lines = response.strip().splitlines()
        if not lines:
            continue
        m = re.search(r'options?\b\s*[:\-]?\s*\*{0,2}([A-Za-z])\b', lines[0], re.IGNORECASE)
        if m is None or (ord(m.group(1).upper()) - ord('A')) >= len(images):
            print("Warning: cannot parse option letter from first line: {}".format(lines[0]))
            continue
        final_response = ord(m.group(1).upper()) - ord('A')
        final_reason = "\n".join(lines[1:]).strip() or "No reason given."
        break

    if final_response is None:
        print("Warning: call planner failed. Response: {}".format(response))
        final_response, final_reason = len(images)-1 if temporal else 0, "Call planner failed."
    return final_response, final_reason



class BoundedDiverseMemory:
    def __init__(self, max_memory=10, max_storage=50):
        self.max_memory = max_memory
        self.max_storage = max_storage

        self.steps  = []
        self.glob_steps = []
        self.frames = []
        self.pos    = []
        self.rot    = []
        self.embed  = []
        self.pos_list = []

    def _poses_to_current(self, i):
        return [p for seq in self.pos_list[i:] for p in seq]

    def update(self, step_id, frame, pos=None, rot=None, embed=None, glob_id=None):
        self.steps.append(step_id)
        self.glob_steps.append(glob_id)
        self.frames.append(frame)
        self.pos.append(pos)
        self.pos_list.append([pos])
        self.rot.append(rot)
        self.embed.append(embed)

        # keep sorted by step
        zipped = list(zip(self.steps, self.frames, self.pos, self.pos_list, self.rot, self.embed, self.glob_steps))
        zipped.sort(key=lambda x: x[0])
        self.steps, self.frames, self.pos, self.pos_list, self.rot, self.embed, self.glob_steps = map(list, zip(*zipped))

        if len(self.steps) > self.max_storage:
            min_gap = float("inf")
            remove_idx = None
            for i in range(1, len(self.steps) - 1):
                g = self.steps[i] - self.steps[i - 1]
                if g < min_gap:
                    min_gap = g
                    remove_idx = i

            if remove_idx is None:
                remove_idx = len(self.steps) - 2

            self.pos_list[remove_idx - 1].extend(self.pos_list[remove_idx])
            self.pos_list.pop(remove_idx)
            self.steps.pop(remove_idx)
            self.glob_steps.pop(remove_idx)
            self.frames.pop(remove_idx)
            self.pos.pop(remove_idx)
            self.rot.pop(remove_idx)
            self.embed.pop(remove_idx)

    def _spatially_far_enough(self, idx, selected, th):
        p = np.asarray(self.pos[idx], dtype=float)
        return all(np.linalg.norm(p - np.asarray(self.pos[j], dtype=float)) >= th for j in selected)

    def get_memory(self, spatial_th=0.25):
        if not self.steps:
            return []
        n = len(self.steps)
        if n <= self.max_memory:
            return [
                (self.steps[i], self.frames[i], self.pos[i], self.rot[i], self.embed[i], self._poses_to_current(i), self.glob_steps[i])
                for i in range(n)
            ]

        valid_idx = [i for i, e in enumerate(self.embed) if isinstance(e, torch.Tensor)]
        assert len(valid_idx) >= self.max_memory

        embeds = torch.stack([self.embed[i].reshape(-1) for i in valid_idx], dim=0).float()
        embeds = F.normalize(embeds, dim=-1)
        idx2row = {idx: r for r, idx in enumerate(valid_idx)}

        latest = n - 1
        assert latest in valid_idx, 'latest step does not have embed.'

        selected = [latest]
        selected_rows = [idx2row[latest]]
        while len(selected) < self.max_memory:
            S = embeds.index_select(0, torch.tensor(selected_rows, device=embeds.device))  # (K,C)
            sims = embeds @ S.transpose(0, 1)  # (Nv,K)
            max_sim = sims.max(dim=1).values  # (Nv,)
            score = 1.0 - max_sim  # (Nv,)
            score[selected_rows] = -1e9

            best_row = None
            for r in torch.argsort(score, descending=True).tolist():
                if score[r] <= -1e8:
                    break
                if self._spatially_far_enough(valid_idx[r], selected, spatial_th):
                    best_row = r
                    break
            if best_row is None:
                best_row = int(score.argmax().item())
            best_idx = valid_idx[best_row]
            selected.append(best_idx)
            selected_rows.append(best_row)

        mem = [(self.steps[i], self.frames[i], self.pos[i], self.rot[i], self.embed[i], self._poses_to_current(i), self.glob_steps[i]) for i in selected]
        mem.sort(key=lambda x: x[0])
        return mem

    def clear(self):
        self.steps.clear()
        self.glob_steps.clear()
        self.frames.clear()
        self.pos.clear()
        self.pos_list.clear()
        self.rot.clear()
        self.embed.clear()


def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [convert_numpy_types(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(x) for x in obj]
    return obj