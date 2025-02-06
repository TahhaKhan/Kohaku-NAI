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

    # Convert any PIL images in the reference_image_multiple list into base64 strings.
    new_ref_images = []
    for ref in reference_image_multiple:
        if isinstance(ref, Image.Image):
            new_ref_images.append(image_to_base64_str(ref))
        else:
            new_ref_images.append(ref)
    reference_image_multiple = new_ref_images


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
        # NEW fields:
        "model": model,
        "variety": variety,
        "reference_image_multiple": reference_image_multiple,
        "reference_information_extracted_multiple": reference_information_extracted_multiple,
        "reference_strength_multiple": reference_strength_multiple,
    }
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
    **kwargs,
):
    if reference_image_multiple is None:
        reference_image_multiple = []
    if reference_information_extracted_multiple is None:
        reference_information_extracted_multiple = []
    if reference_strength_multiple is None:
        reference_strength_multiple = []
    # If seed is -1, generate a random seed.
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)

    # Convert any PIL images in the reference_image_multiple list into base64 strings.
    new_ref_images = []
    for ref in reference_image_multiple:
        if isinstance(ref, Image.Image):
            new_ref_images.append(image_to_base64_str(ref))
        else:
            new_ref_images.append(ref)
    reference_image_multiple = new_ref_images

    # Build payload differently depending on the model
    if model == "nai-diffusion-4-curated-preview":
        if ucpreset not in ["Heavy", "Light", "None"]:
            preset = 1  # default to Light
        else:
            preset = {"Heavy": 0, "Light": 1, "None": 2}[ucpreset]
        neg = f"{UCPRESETV4[ucpreset]}, {negative_prompt}" if ucpreset in UCPRESET else negative_prompt
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
                "dynamic_thresholding": dyn_threshold,
                "controlnet_strength": 1,
                "legacy": False,
                "add_original_image": True,
                "cfg_rescale": cfg_rescale,
                "noise_schedule": schedule,
                "legacy_v3_extend": False,
                "skip_cfg_above_sigma": 19 if variety else None,
                "use_coords": False,
                "v4_prompt": {
                    "caption": {"base_caption": prompt, "char_captions": []},
                    "use_coords": False,
                    "use_order": True,
                },
                "v4_negative_prompt": {
                    "caption": {"base_caption": neg, "char_captions": []}
                },
                "seed": seed,
                "characterPrompts": [],
                "negative_prompt": neg,
                "reference_image_multiple": [],  # v4 currently disables reference images
                "reference_information_extracted_multiple": [],
                "reference_strength_multiple": [],
            },
        }
    else:
        # v3 model payload
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


    # Process the response
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