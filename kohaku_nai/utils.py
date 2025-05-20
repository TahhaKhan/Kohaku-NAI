import sys
import asyncio
import random
import re
import io
import zipfile
import json
import piexif
from typing import Any
import base64
from PIL import Image
from httpx import AsyncClient
from curl_cffi.requests import AsyncSession


if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


API_URL = "https://api.novelai.net"
API_IMAGE_URL = "https://image.novelai.net"
HttpClient = AsyncClient | AsyncSession

jwt_token = ""
global_client: HttpClient | None = None

def image_to_base64_str(img: Image.Image) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

async def make_client(
    backend: str = "httpx",
    remote_server: str = "",
    password: str = "",
    token: str = "",
):
    assert backend in ["httpx", "curl_cffi"]
    assert remote_server or token
    if backend == "httpx":
        client_class = AsyncClient
    else:
        client_class = AsyncSession

    if remote_server:
        client = client_class(timeout=3600)
        payload = {"password": password}
        response = await client.post(f"{remote_server}/login", params=payload)
        if response.status_code == 200:
            return client, response.json()["status"]
    else:
        kwargs = {
            "timeout": 3600,
            "headers": {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Origin": "https://novelai.net",
                "Referer": "https://novelai.net/",
            },
        }
        if backend == "curl_cffi":
            kwargs["impersonate"] = "chrome110"
        client = client_class(**kwargs)
        status = await client.get(f"{API_URL}/user/data")
        if status.status_code == 200:
            return client, status.json()
    return None, None


async def set_client(
    backend: str = "httpx",
    remote_server: str = "",
    password: str = "",
    token: str = "",
):
    global global_client
    global_client, status = await make_client(backend, remote_server, password, token)
    return status


QUALITY_TAGS = "best quality, amazing quality, very aesthetic, absurdres"
V45_QUALITY_TAGS = "location, masterpiece, no text, -0.8::feet::, rating:general"
UCPRESET = {
    "Heavy": "lowres, {bad}, error, fewer, extra, missing, worst quality, jpeg artifacts, bad quality, watermark, unfinished, displeasing, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract],",
    "Light": "lowres, jpeg artifacts, worst quality, watermark, blurry, very displeasing,",
    "Human Focus": "lowres, {bad}, error, fewer, extra, missing, worst quality, jpeg artifacts, bad quality, watermark, unfinished, displeasing, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract], bad anatomy, bad hands, @_@, mismatched pupils, heart-shaped pupils, glowing eyes,",
    "None": "",
}
UCPRESETV4 = {
    "Heavy": "blurry, lowres, error, film grain, scan artifacts, worst quality, bad quality, jpeg artifacts, very displeasing, chromatic aberration, logo, dated, signature, multiple views,",
    "Light": "blurry, lowres, error, worst quality, bad quality, jpeg artifacts, very displeasing, logo, dated, signature,",
    "None": "",
}
UCPRESETV45 = {
    "None": "",
    "Light": "blurry, lowres, upscaled, artistic error, scan artifacts, jpeg artifacts, logo, too many watermarks, negative space, blank page",
    "Heavy": "blurry, lowres, upscaled, artistic error, film grain, scan artifacts, worst quality, bad quality, jpeg artifacts, very displeasing, chromatic aberration, halftone, multiple views, logo, too many watermarks, negative space, blank page",
    "Human Focus": "blurry, lowres, upscaled, artistic error, film grain, scan artifacts, bad anatomy, bad hands, worst quality, bad quality, jpeg artifacts, very displeasing, chromatic aberration, halftone, multiple views, logo, too many watermarks, @_@, mismatched pupils, glowing eyes, negative space, blank page",
}
DEFAULT_ARGS = {
    "prompt": "",
    "negative_prompt": "",
    "quality_tags": False,
    "ucpreset": "",
    "seed": -1,
    "scale": 5.0,
    "width": 1024,
    "height": 1024,
    "steps": 28,
    "sampler": "k_euler",
    "schedule": "native",
    "smea": False,
    "dyn": False,
    "dyn_threshold": False,
    "cfg_rescale": 0,
    "images": 1,
}


file_name_cleaner = re.compile(r"[^a-zA-Z0-9_.-]")


def make_file_name(args: dict[str, Any]):
    prompt = args.pop("prompt", "")[:20]
    neg_prompt = args.pop("negative_prompt", "")[:20]
    file_name = f"{prompt}_{neg_prompt}_" + "_".join(
        [f"{k}={v}" for k, v in args.items()]
    )
    return file_name_cleaner.sub("", file_name)


async def remote_gen(
    end_point="http://127.0.0.1:7000",
    prompt="",
    quality_tags=False,
    negative_prompt="",
    ucpreset="",
    seed=-1,
    scale=5.0,
    width=1024,
    height=1024,
    steps=28,
    sampler="k_euler",
    schedule="native",
    smea=False,
    dyn=False,
    dyn_threshold=False,
    cfg_rescale=0,
    model="nai-diffusion-3",
    variety=False,
    reference_image_multiple=None,
    reference_information_extracted_multiple=None,
    reference_strength_multiple=None,
    extra_infos={},
    priority=0,
    character_prompts=[],
    use_ai_char=False,
    **kwargs,
):
    if reference_image_multiple is None:
        reference_image_multiple = []
    if reference_information_extracted_multiple is None:
        reference_information_extracted_multiple = []
    if reference_strength_multiple is None:
        reference_strength_multiple = []

    if seed == -1:
        seed = random.randint(0, 2**32 - 1)

    new_ref_images = []
    for ref in reference_image_multiple:
        if isinstance(ref, Image.Image):
            new_ref_images.append(image_to_base64_str(ref))
        else:
            new_ref_images.append(ref)
    reference_image_multiple = new_ref_images

    print(f"[DEBUG] use_ai_char: {use_ai_char}")
    print(f"[DEBUG] character_prompts: {character_prompts}")
    print(f"[DEBUG] ucpreset: {ucpreset}")
    print(f"[DEBUG] model: {model}")

    payload = {
        "prompt": f"{prompt}, {QUALITY_TAGS}" if quality_tags else prompt,
        "neg_prompt": f"{UCPRESET[ucpreset]}, {negative_prompt}" if ucpreset in UCPRESET else negative_prompt,
        "seed": seed,
        "scale": scale,
        "width": width,
        "height": height,
        "steps": steps,
        "sampler": sampler,
        "schedule": schedule,
        "smea": smea,
        "dyn": dyn,
        "dyn_threshold": dyn_threshold,
        "cfg_rescale": cfg_rescale,
        "extra_infos": (
            extra_infos if isinstance(extra_infos, str) else json.dumps(extra_infos, ensure_ascii=False)
        ),
        "priority": priority,
        "model": model,
        "variety": variety,
        "reference_image_multiple": reference_image_multiple,
        "reference_information_extracted_multiple": reference_information_extracted_multiple,
        "reference_strength_multiple": reference_strength_multiple,
        "characterPrompts": character_prompts,
    }

    if model.strip() in ("nai-diffusion-4-curated-preview", "nai-diffusion-4-full", "nai-diffusion-4-5-curated"):
        # Handle preset mapping based on model
        if ucpreset not in ["Heavy", "Light", "Human Focus", "None"]:
            if model.strip() == "nai-diffusion-4-5-curated":
                preset = 1  # Default to Light for V4.5
            else:
                preset = 1  # Default to Light for V4
        else:
            # Handle preset mapping based on model
            if model.strip() == "nai-diffusion-4-5-curated":
                preset = {"Heavy": 0, "Light": 1, "Human Focus": 2, "None": 3}[ucpreset]
            else:
                preset = {"Heavy": 0, "Light": 1, "None": 2}[ucpreset]
        
        # Handle V4.5 separately for UC presets and quality tags
        if model.strip() == "nai-diffusion-4-5-curated":
            neg = f"{UCPRESETV45[ucpreset]}, {negative_prompt}" if ucpreset in UCPRESETV45 else negative_prompt
            if quality_tags:
                prompt = f"{prompt}, {V45_QUALITY_TAGS}"
        else:
            neg = f"{UCPRESETV4[ucpreset]}, {negative_prompt}" if ucpreset in UCPRESETV4 else negative_prompt
            # V4 models don't use quality tags by default

        # Use the explicit use_ai_char parameter
        using_ai_choice = use_ai_char and len(character_prompts) > 0
        use_coords_value = not using_ai_choice  # True if manual, False if AI positioning
        
        print(f"[DEBUG DETAIL] use_ai_char: {use_ai_char}, character_prompts length: {len(character_prompts)}")
        print(f"[DEBUG DETAIL] using_ai_choice: {using_ai_choice}, use_coords_value: {use_coords_value}")
        print(f"[DEBUG DETAIL] Character prompts:")
        for i, cp in enumerate(character_prompts):
            print(f"  Char {i+1}: prompt='{cp.get('prompt', 'None')}', center={cp.get('center', 'None')}")
        
        # CRITICAL FIX: Check if any character prompt has use_ai_position flag
        ai_position_requested = any(cp.get('use_ai_position', False) for cp in character_prompts if cp)
        if ai_position_requested:
            print("[DEBUG CRITICAL] Character prompt has use_ai_position flag - forcing AI positioning")
            using_ai_choice = True
            use_coords_value = False
        
        v4_char_captions = [{"char_caption": cp["prompt"], "centers": [cp["center"]]} for cp in character_prompts]
        v4_neg_char_captions = [{"char_caption": cp["uc"], "centers": [cp["center"]]} for cp in character_prompts]
        payload["parameters"] = {
            "params_version": 3,
            "width": width,
            "height": height,
            "scale": scale,
            "sampler": sampler,
            "steps": steps,
            "n_samples": 1,
            "ucPreset": preset,  
            "qualityToggle": True,
            "autoSmea": smea,
            "dynamic_thresholding": dyn_threshold,
            "controlnet_strength": 1,
            "legacy": False,
            "add_original_image": True,
            "cfg_rescale": cfg_rescale,
            "noise_schedule": schedule,
            "legacy_v3_extend": False,
            "skip_cfg_above_sigma": 19 if variety else None,
            "use_coords": use_coords_value,
            "normalize_reference_strength_multiple": True,
            "v4_prompt": {
                "caption": {"base_caption": prompt, "char_captions": v4_char_captions},
                "use_coords": use_coords_value,
                "use_order": True
            },
            "v4_negative_prompt": {
                "caption": {"base_caption": neg, "char_captions": v4_neg_char_captions},
                "use_coords": False,  # Always false for negative prompt
                "use_order": False
            },
            "seed": seed,
            "characterPrompts": character_prompts,
            "negative_prompt": neg,
            "reference_image_multiple": [],  
            "reference_information_extracted_multiple": [],
            "reference_strength_multiple": [],
            "deliberate_euler_ancestral_bug": False,  # Match official client
            "prefer_brownian": True  # Match official client
        }

    # Final debug check of use_coords settings
    print(f"[DEBUG FINAL] remote_gen payload use_coords: {payload['parameters']['use_coords']}")
    print(f"[DEBUG FINAL] remote_gen v4_prompt use_coords: {payload['parameters']['v4_prompt']['use_coords']}")
    print(f"[DEBUG FINAL] remote_gen character_prompts count: {len(character_prompts)}")

    # Update the payload prompt for remote_gen
    if model.strip() == "nai-diffusion-4-5-curated" and quality_tags:
        payload["prompt"] = prompt  # Use the already modified prompt
        
    # CRITICAL FIX: Force correct use_coords settings right before sending
    if use_ai_char and len(character_prompts) > 0:
        print("[DEBUG OVERRIDE] Forcing use_coords to False for AI positioning")
        payload["parameters"]["use_coords"] = False
        payload["parameters"]["v4_prompt"]["use_coords"] = False

    response = await global_client.post(f"{end_point}/gen", json=payload)
    if response.status_code == 200:
        mem_file = io.BytesIO(response.content)
        mem_file.seek(0)
        return Image.open(mem_file), response.content
    else:
        try:
            data = response.json()
        except json.JSONDecodeError:
            data = response.content
        return None, data



async def generate_novelai_image(
    prompt="",
    quality_tags=False,
    negative_prompt="",
    ucpreset="",
    seed=-1,
    scale=5.0,
    width=1024,
    height=1024,
    steps=28,
    sampler="k_euler",
    schedule="native",
    smea=False,
    dyn=False,
    dyn_threshold=False,
    cfg_rescale=0,
    client: HttpClient | None = None,
    model="nai-diffusion-3",
    variety=False,
    reference_image_multiple=None,
    reference_information_extracted_multiple=None,
    reference_strength_multiple=None,
    character_prompts=[],
    use_ai_char=False,
    **kwargs,
):
    if reference_image_multiple is None:
        reference_image_multiple = []
    if reference_information_extracted_multiple is None:
        reference_information_extracted_multiple = []
    if reference_strength_multiple is None:
        reference_strength_multiple = []
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)

    new_ref_images = []
    for ref in reference_image_multiple:
        if isinstance(ref, Image.Image):
            new_ref_images.append(image_to_base64_str(ref))
        else:
            new_ref_images.append(ref)
    reference_image_multiple = new_ref_images

    if model.strip() in ("nai-diffusion-4-curated-preview", "nai-diffusion-4-full", "nai-diffusion-4-5-curated"):
        # Handle preset mapping based on model
        if ucpreset not in ["Heavy", "Light", "Human Focus", "None"]:
            if model.strip() == "nai-diffusion-4-5-curated":
                preset = 1  # Default to Light for V4.5
            else:
                preset = 1  # Default to Light for V4
        else:
            # Handle preset mapping based on model
            if model.strip() == "nai-diffusion-4-5-curated":
                preset = {"Heavy": 0, "Light": 1, "Human Focus": 2, "None": 3}[ucpreset]
            else:
                preset = {"Heavy": 0, "Light": 1, "None": 2}[ucpreset]
        
        # Handle V4.5 separately for UC presets and quality tags
        if model.strip() == "nai-diffusion-4-5-curated":
            neg = f"{UCPRESETV45[ucpreset]}, {negative_prompt}" if ucpreset in UCPRESETV45 else negative_prompt
            if quality_tags:
                prompt = f"{prompt}, {V45_QUALITY_TAGS}"
        else:
            neg = f"{UCPRESETV4[ucpreset]}, {negative_prompt}" if ucpreset in UCPRESETV4 else negative_prompt
            # V4 models don't use quality tags by default
        
        # Use the explicit use_ai_char parameter
        using_ai_choice = use_ai_char and len(character_prompts) > 0
        use_coords_value = not using_ai_choice  # True if manual, False if AI positioning
        
        print(f"[DEBUG DETAIL] use_ai_char: {use_ai_char}, character_prompts length: {len(character_prompts)}")
        print(f"[DEBUG DETAIL] using_ai_choice: {using_ai_choice}, use_coords_value: {use_coords_value}")
        print(f"[DEBUG DETAIL] Character prompts:")
        for i, cp in enumerate(character_prompts):
            print(f"  Char {i+1}: prompt='{cp.get('prompt', 'None')}', center={cp.get('center', 'None')}")
        
        # CRITICAL FIX: Check if any character prompt has use_ai_position flag
        ai_position_requested = any(cp.get('use_ai_position', False) for cp in character_prompts if cp)
        if ai_position_requested:
            print("[DEBUG CRITICAL] Character prompt has use_ai_position flag - forcing AI positioning")
            using_ai_choice = True
            use_coords_value = False
        
        v4_char_captions = [{"char_caption": cp["prompt"], "centers": [cp["center"]]} for cp in character_prompts]
        v4_neg_char_captions = [{"char_caption": cp["uc"], "centers": [cp["center"]]} for cp in character_prompts]
        payload = {
            "input": prompt,
            "model": model,
            "action": "generate",
            "parameters": {
                "params_version": 3,
                "width": width,
                "height": height,
                "scale": scale,
                "sampler": sampler,
                "steps": steps,
                "n_samples": 1,
                "ucPreset": preset,
                "qualityToggle": True,
                "autoSmea": smea,
                "dynamic_thresholding": dyn_threshold,
                "controlnet_strength": 1,
                "legacy": False,
                "add_original_image": True,
                "cfg_rescale": cfg_rescale,
                "noise_schedule": schedule,
                "legacy_v3_extend": False,
                "skip_cfg_above_sigma": 19 if variety else None,
                "use_coords": use_coords_value,
                "normalize_reference_strength_multiple": True,
                "v4_prompt": {
                    "caption": {"base_caption": prompt, "char_captions": v4_char_captions},
                    "use_coords": use_coords_value,
                    "use_order": True
                },
                "v4_negative_prompt": {
                    "caption": {"base_caption": neg, "char_captions": v4_neg_char_captions},
                    "use_coords": False,  # Always false for negative prompt
                    "use_order": False
                },
                "seed": seed,
                "characterPrompts": [],
                "negative_prompt": neg,
                "reference_image_multiple": [], 
                "reference_information_extracted_multiple": [],
                "reference_strength_multiple": [],
                "deliberate_euler_ancestral_bug": False,  # Match official client
                "prefer_brownian": True  # Match official client
            },
        }
        if character_prompts:
            payload["parameters"]["characterPrompts"] = character_prompts
        
        # Final debug check of use_coords settings
        print(f"[DEBUG FINAL] generate_novelai_image payload use_coords: {payload['parameters']['use_coords']}")
        print(f"[DEBUG FINAL] generate_novelai_image v4_prompt use_coords: {payload['parameters']['v4_prompt']['use_coords']}")
        print(f"[DEBUG FINAL] generate_novelai_image character_prompts count: {len(character_prompts)}")
        
        # CRITICAL FIX: Force correct use_coords settings right before sending
        if use_ai_char and len(character_prompts) > 0:
            print("[DEBUG OVERRIDE] Forcing use_coords to False for AI positioning")
            payload["parameters"]["use_coords"] = False
            payload["parameters"]["v4_prompt"]["use_coords"] = False
        
        print (payload)
    else:
        if ucpreset not in ["Heavy", "Light", "Human Focus", "None"]:
            preset = 0  
        else:
            preset = {"Heavy": 0, "Light": 1, "Human Focus": 2, "None": 3}[ucpreset]
        neg = f"{UCPRESET[ucpreset]}, {negative_prompt}" if ucpreset in UCPRESET else negative_prompt
        payload = {
            "action": "generate",
            "input": f"{prompt}, {QUALITY_TAGS}" if quality_tags else prompt,
            "model": model,
            "parameters": {
                "width": width,
                "height": height,
                "scale": scale,
                "sampler": sampler,
                "steps": steps,
                "n_samples": 1,
                "ucPreset": preset,
                "qualityToggle": True,
                "sm": smea,
                "sm_dyn": dyn,
                "dynamic_thresholding": dyn_threshold,
                "controlnet_strength": 1,
                "legacy": False,
                "add_original_image": True,
                "cfg_rescale": cfg_rescale,
                "noise_schedule": schedule,
                "legacy_v3_extend": False,
                "skip_cfg_above_sigma": 19 if variety else None,
                "seed": seed,
                "characterPrompts": [],
                "negative_prompt": neg,
                "reference_image_multiple": reference_image_multiple,
                "reference_information_extracted_multiple": reference_information_extracted_multiple,
                "reference_strength_multiple": reference_strength_multiple,
            },
        }
    response = await client.post(f"{API_IMAGE_URL}/ai/generate-image", json=payload)


    if response.headers.get("Content-Type") == "binary/octet-stream":
        zipfile_in_memory = io.BytesIO(response.content)
        with zipfile.ZipFile(zipfile_in_memory, "r") as zip_ref:
            file_names = zip_ref.namelist()
            if file_names:
                with zip_ref.open(file_names[0]) as file:
                    return file.read(), json.dumps(
                        payload, ensure_ascii=False, indent=2
                    )
            else:
                return "NAI doesn't return any images", response
    else:
        return "Generation failed", response


def free_check(width: int, height: int, steps: int):
    return width * height <= 1024 * 1024 and steps <= 28


def image_from_bytes(data: bytes):
    img_file = io.BytesIO(data)
    img_file.seek(0)
    return Image.open(img_file)


def process_image_as_webp(
    image: Image,
    quality: int = 75,
    method: int = 4,
    metadata: dict[str, Any] = None,
) -> bytes:
    """
    encode image as webp.

    See:
        https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#webp

    Args:
        image (Image): a PIL image, assumed to be generated by NovelAI and with metadata in pnginfo
        quality (int, optional): webp compression quality. Defaults to 75.
        method (int, optional): webp compression method. Defaults to 4.
        metadata (dict[str, Any], optional): `metadata` be directly encoded as Exif, if provided. Defaults to None.

    Returns:
        bytes: the encoded image
    """
    metadata_bytes: bytes | None = None
    if metadata:
        metadata_bytes = piexif.dump(metadata)
    else:
        # Try to read metadata from image.  Note that `image.info` will return a
        # dict that NOT include `exif` field if the image is generated by
        # NovelAI.  NovelAI embeds metadata in pnginfo, which is different from
        # exif.  (exif just a field in pnginfo in this case)
        items = (image.info or {}).copy()
        if len(items) > 0:
            # WebP only support save exif in the metadata. So we put everything
            # in UserComment field
            #
            # The bad news is that the code from AUTOMATIC1111 still can't read
            # from it directly. To address this a custom schema mapping is
            # required (which leads to replace the whole Exif.UserComment field
            # and loss the original metadata) or modification on
            # `read_info_from_image` function in AUTOMATIC1111
            if "Comment" in items:
                try:
                    comment_str = items["Comment"]
                    # Let's unmarsal then marshal it for aesthetic
                    json_info = json.loads(comment_str)
                    items["Comment"] = json_info
                except json.JSONDecodeError:
                    pass
            # https://exiftool.org/TagNames/EXIF.html
            # 0x9286 UserComment
            metadata_bytes = piexif.dump(
                {"Exif": {0x9286: bytes(json.dumps(items, indent=4), "utf-8")}}
            )

    ret = io.BytesIO()
    if metadata_bytes:
        image.save(
            ret,
            format="webp",
            quality=quality,
            method=method,
            lossless=False,
            exact=False,
            exif=metadata_bytes,
        )
    else:
        image.save(
            ret,
            format="webp",
            quality=quality,
            method=method,
            lossless=False,
            exact=False,
        )
    ret.seek(0)
    return ret.read()


class GenerationError(Exception):
    pass