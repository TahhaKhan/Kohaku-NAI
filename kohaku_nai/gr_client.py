import os
import time
from threading import Thread
from hashlib import sha3_256

import toml
import json
import gradio as gr
import webview

from kohaku_nai.utils import (
    remote_gen,
    set_client,
    generate_novelai_image,
    image_from_bytes,
)
from kohaku_nai.client_modules import extension


client_config: dict = toml.load("config.toml")["client"]
extra_infos = client_config.get("remote_extra_infos", {})

# Mapping from UI model names to API model names
MODEL_NAME_MAPPING = {
    "NAI V3": "nai-diffusion-3",
    "NAI V4 Curated": "nai-diffusion-4-curated-preview",
    "NAI V4 Full": "nai-diffusion-4-full",
    "NAI V4.5 Curated": "nai-diffusion-4-5-curated"
}

def control_ui():
    gr.Markdown("### Main Prompts")
    prompt = gr.TextArea(
        label="Prompt",
        lines=3,
        value=client_config["default_prompt"],
        placeholder="Enter your main prompt here...",
    )
    neg_prompt = gr.TextArea(
        label="Negative Prompt",
        lines=2,
        value=client_config["default_neg"],
        placeholder="Enter terms to avoid in the generation...",
    )
    with gr.Row():
        with gr.Column(scale=3, min_width=160):
            enable_quality_tags = gr.Checkbox(label="Enable Quality Tags", value=True, info="Add quality-enhancing tags automatically")
        with gr.Column(scale=5, min_width=360):
            neg_preset = gr.Radio(
                choices=["Heavy", "Light", "Human Focus", "None"],
                value="Light",
                label="UC Preset",
                info="Predefined negative prompt templates",
            )
    
    gr.Markdown("### Generation Parameters")
    with gr.Row():
        with gr.Column(scale=1):
            seed = gr.Number(
                label="Seed", 
                value=-1, 
                step=1, 
                maximum=2**32 - 1, 
                minimum=-1,
                info="Random seed if -1"
            )
        with gr.Column(scale=1):
            sampler = gr.Dropdown(
                choices=[
                    "k_euler",
                    "k_euler_ancestral",
                    "k_dpmpp_2s_ancestral",
                    "k_dpmpp_2m",
                    "k_dpmpp_2m_sde",
                    "k_dpmpp_sde",
                    "ddim_v3",
                ],
                value="k_euler",
                label="Sampler",
                interactive=True,
            )
        with gr.Column(scale=1):
            scale = gr.Slider(
                label="CFG Scale", 
                value=5.0, 
                minimum=1, 
                maximum=10, 
                step=0.1,
                info="How closely to follow the prompt",
            )
        with gr.Column(scale=1):
            steps = gr.Slider(
                label="Steps", 
                value=28, 
                minimum=1, 
                maximum=50, 
                step=1,
                info="More steps = better quality but slower",
            )
    with gr.Row():
        with gr.Column(scale=1):
            width = gr.Slider(
                label="Width", 
                value=832, 
                minimum=64, 
                maximum=2048, 
                step=64,
                info="Image width in pixels",
            )
        with gr.Column(scale=1):
            height = gr.Slider(
                label="Height", 
                value=1216, 
                minimum=64, 
                maximum=2048, 
                step=64,
                info="Image height in pixels",
            )

    return [width, height], [
        prompt,
        enable_quality_tags,
        neg_prompt,
        neg_preset,
        seed,
        scale,
        width,
        height,
        steps,
        sampler,
    ]


def settings_ui():
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Model Settings")
            with gr.Row():
                model_selector = gr.Radio(
                    choices=[
                        "NAI V3",
                        "NAI V4 Curated",
                        "NAI V4 Full",
                        "NAI V4.5 Curated"
                    ],
                    value="NAI V3",
                    label="Model",
                    interactive=True,
                )
                variety_chk = gr.Checkbox(False, label="Variety+")
            
            gr.Markdown("### Advanced Generation Settings")
            scheduler = gr.Dropdown(
                choices=["native", "karras", "exponential", "polyexponential"],
                value="native",
                label="Scheduler",
                interactive=True,
            )
            with gr.Row():
                with gr.Column(scale=1):
                    smea = gr.Checkbox(False, label="SMEA")
                with gr.Column(scale=1):
                    dyn = gr.Checkbox(False, label="SMEA DYN")
                with gr.Column(scale=1):
                    dyn_threshold = gr.Checkbox(False, label="Dynamic Thresholding")
            with gr.Row():
                cfg_rescale = gr.Slider(0, 1, 0, step=0.01, label="CFG rescale")

        with gr.Column():
            gr.Markdown("### Client Settings")
            mode = gr.Radio(
                ["remote", "local"], value=client_config["mode"], label="Mode"
            )
            backend = gr.Radio(
                ["curl_cffi", "httpx"],
                value=client_config.get("backend", "curl_cffi"),
                label="Http Backend",
                info='use "httpx" if you meet issues with "curl_cffi"',
            )
            with gr.Column(visible=client_config["mode"] == "remote") as remote_blk:
                end_point = gr.Textbox(client_config["end_point"], label="End Point")
                end_point_pswd = gr.Textbox(
                    client_config["end_point_pswd"],
                    label="End Point Password",
                    type="password",
                )
                extra_info_json = gr.Code(
                    value=json.dumps(extra_infos, ensure_ascii=False, indent=2),
                    label="Remote Extra Infos",
                    language="json",
                )
            with gr.Column(visible=client_config["mode"] == "local") as local_blk:
                token = gr.Textbox(client_config["token"], label="Token", type="password")

            mode.change(lambda m: gr.update(visible=m == "remote"), mode, remote_blk)
            mode.change(lambda m: gr.update(visible=m == "local"), mode, local_blk)

    return [scheduler, smea, dyn, dyn_threshold, cfg_rescale, extra_info_json, model_selector, variety_chk], [mode, backend, end_point, end_point_pswd, token]



async def generate(
    mode, backend, end_point, end_point_pswd, token,
    prompt, enable_quality_tags, neg_prompt, neg_preset,
    seed, scale, width, height, steps, sampler, scheduler,
    smea, dyn, dyn_threshold, cfg_rescale, extra_info_json,
    model, variety,  # from the new model_selector and variety checkbox
    ref_img1, ref_info1, ref_strength1,
    ref_img2, ref_info2, ref_strength2,
    ref_img3, ref_info3, ref_strength3,
    ref_img4, ref_info4, ref_strength4,
    ref_img5, ref_info5, ref_strength5,
    # NEW character controls:
    use_ai_char,
    char_prompt1, char_neg1, char_x1, char_y1,
    char_prompt2, char_neg2, char_x2, char_y2,
    char_prompt3, char_neg3, char_x3, char_y3,
    char_prompt4, char_neg4, char_x4, char_y4,
    char_prompt5, char_neg5, char_x5, char_y5,
    char_prompt6, char_neg6, char_x6, char_y6,
):

    prompt = extension.process_prompt(prompt)
    neg_prompt = extension.process_prompt(neg_prompt)

    # Map UI model name to API model name
    api_model = MODEL_NAME_MAPPING.get(model, model)  # Fallback to original name if not in mapping

    reference_images = []
    reference_info = []
    reference_strength = []

    for r_img, r_info, r_str in [
            (ref_img1, ref_info1, ref_strength1),
            (ref_img2, ref_info2, ref_strength2),
            (ref_img3, ref_info3, ref_strength3),
            (ref_img4, ref_info4, ref_strength4),
            (ref_img5, ref_info5, ref_strength5)]:
        if r_img is not None:
            reference_images.append(r_img)
            reference_info.append(r_info)
            reference_strength.append(r_str)

    # Build character_prompts if model = v4
    character_prompts = []
    if model in ("NAI V4 Curated", "NAI V4 Full", "NAI V4.5 Curated"):
        def make_char(cp, cn, x, y):
            if not cp or not cp.strip():
                return None

            cp = extension.process_prompt(cp)  
            cn = extension.process_prompt(cn)
            
            if use_ai_char:
                center = {"x": 0.5, "y": 0.5}
                print(f"[DEBUG CHAR] Using AI choice: center set to fixed position (0.5, 0.5)")
            else:
                center = {"x": float(x), "y": float(y)}
                print(f"[DEBUG CHAR] Manual positioning: center set to ({x}, {y})")
            
            return {
                "prompt": cp.strip(),
                "uc": (cn.strip() if cn else ""),
                "center": center,
                "use_ai_position": use_ai_char
            }

        char_list = [
            make_char(char_prompt1, char_neg1, char_x1, char_y1),
            make_char(char_prompt2, char_neg2, char_x2, char_y2),
            make_char(char_prompt3, char_neg3, char_x3, char_y3),
            make_char(char_prompt4, char_neg4, char_x4, char_y4),
            make_char(char_prompt5, char_neg5, char_x5, char_y5),
            make_char(char_prompt6, char_neg6, char_x6, char_y6),
        ]
        character_prompts = [c for c in char_list if c is not None]
        print(f"[DEBUG CHAR] Created {len(character_prompts)} character prompts")

    if mode == "remote":
        if (pswd := end_point_pswd) or (pswd := client_config["end_point_pswd"]):
            await set_client(backend, end_point, pswd)
        print(f"[DEBUG] Sending to remote_gen with use_ai_char={use_ai_char}")
        print(f"[DEBUG FINAL CHECK] use_ai_char={use_ai_char} with {len(character_prompts)} character prompts")
        img, img_data = await remote_gen(
            end_point,
            prompt,
            enable_quality_tags,
            neg_prompt,
            neg_preset,
            seed,
            scale,
            width,
            height,
            steps,
            sampler,
            scheduler,
            smea,
            dyn,
            dyn_threshold,
            cfg_rescale,
            model=api_model,
            variety=variety,
            reference_image_multiple=reference_images,
            reference_information_extracted_multiple=reference_info,
            reference_strength_multiple=reference_strength,
            extra_infos=extra_info_json,
            character_prompts=character_prompts,
            use_ai_char=use_ai_char
        )
        if not isinstance(img_data, bytes):
            print(f"Error Message: {img_data}")
            return None
    elif mode == "local":
        await set_client(backend, token=token)
        print(f"[DEBUG] Sending to generate_novelai_image with use_ai_char={use_ai_char}")
        print(f"[DEBUG FINAL CHECK] use_ai_char={use_ai_char} with {len(character_prompts)} character prompts")
        img_data, _ = await generate_novelai_image(
            prompt,
            enable_quality_tags,
            neg_prompt,
            neg_preset,
            seed,
            scale,
            width,
            height,
            steps,
            sampler,
            scheduler,
            smea,
            dyn,
            dyn_threshold,
            cfg_rescale,
            model=api_model,
            variety=variety,
            reference_image_multiple=reference_images,
            reference_information_extracted_multiple=reference_info,
            reference_strength_multiple=reference_strength,
            character_prompts=character_prompts,
            extra_infos=extra_info_json,
            use_ai_char=use_ai_char
        )
        if not isinstance(img_data, bytes):
            print(f"Error Message: {img_data}")
            return None
        img = image_from_bytes(img_data)
    else:
        return None

    if img is None:
        return None

    if client_config["autosave"]:
        save_path = client_config["save_path"]
        os.makedirs(name=save_path, exist_ok=True)
        img_hash = sha3_256(img_data).hexdigest()
        with open(os.path.join(save_path, f"{img_hash}.png"), "wb") as f:
            f.write(img_data)

    return [img]


def preview_ui():
    with gr.Blocks() as page:
        image = gr.Gallery(elem_id="preview_image")
    return image


def main_ui():
    with gr.Blocks() as page:
        with gr.Row(variant="panel"):
            with gr.Column():
                with gr.Tabs():
                    with gr.TabItem("Gen"):
                        (width, height), controls = control_ui()
                    with gr.TabItem("Settings"):
                        adv_controls, modes = settings_ui()
                        (scheduler, smea, dyn, dyn_threshold, cfg_rescale,
                         extra_info_json, model_selector, variety_chk) = adv_controls
            with gr.Column():
                gen_btn = gr.Button(value="Generate", variant="primary")
                image = preview_ui()
            # --- New Character Controls for V4 (only visible when model == "NAI V4 Curated") ---
        with gr.Accordion("Characters (V4 and V4.5 models)", open=False, elem_id="char_container") as char_container:
            gr.Markdown("You may add up to 6 characters. Characters are only available for V4.0, V4.5 models. Leave the prompt empty to ignore a character.")
            use_ai_char = gr.Checkbox(
                label="Use AI's Choice for Character Positions",
                value=True,
                info="If checked, all positions become (0.5,0.5) ignoring your X/Y fields",
            )

            with gr.Tabs():
                with gr.TabItem("Character 1"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            char_prompt1 = gr.Textbox(
                                label="Character 1 Prompt", 
                                placeholder="e.g. 1girl, red hair...",
                                lines=2
                            )
                        with gr.Column(scale=2):
                            char_neg1 = gr.Textbox(
                                label="Character 1 Negative", 
                                placeholder="Optional undesired tags",
                                lines=2
                            )
                    with gr.Row():
                        char_x1 = gr.Number(label="X Position", value=0.5, precision=3, info="0.0 is left, 1.0 is right")
                        char_y1 = gr.Number(label="Y Position", value=0.5, precision=3, info="0.0 is top, 1.0 is bottom")

                with gr.TabItem("Character 2"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            char_prompt2 = gr.Textbox(
                                label="Character 2 Prompt", 
                                placeholder="e.g. 1girl, red hair...",
                                lines=2
                            )
                        with gr.Column(scale=2):
                            char_neg2 = gr.Textbox(
                                label="Character 2 Negative", 
                                placeholder="Optional undesired tags",
                                lines=2
                            )
                    with gr.Row():
                        char_x2 = gr.Number(label="X Position", value=0.5, precision=3, info="0.0 is left, 1.0 is right")
                        char_y2 = gr.Number(label="Y Position", value=0.5, precision=3, info="0.0 is top, 1.0 is bottom")

                with gr.TabItem("Character 3"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            char_prompt3 = gr.Textbox(
                                label="Character 3 Prompt", 
                                placeholder="e.g. 1girl, red hair...",
                                lines=2
                            )
                        with gr.Column(scale=2):
                            char_neg3 = gr.Textbox(
                                label="Character 3 Negative", 
                                placeholder="Optional undesired tags",
                                lines=2
                            )
                    with gr.Row():
                        char_x3 = gr.Number(label="X Position", value=0.5, precision=3, info="0.0 is left, 1.0 is right")
                        char_y3 = gr.Number(label="Y Position", value=0.5, precision=3, info="0.0 is top, 1.0 is bottom")

                with gr.TabItem("Character 4"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            char_prompt4 = gr.Textbox(
                                label="Character 4 Prompt", 
                                placeholder="e.g. 1girl, red hair...",
                                lines=2
                            )
                        with gr.Column(scale=2):
                            char_neg4 = gr.Textbox(
                                label="Character 4 Negative", 
                                placeholder="Optional undesired tags",
                                lines=2
                            )
                    with gr.Row():
                        char_x4 = gr.Number(label="X Position", value=0.5, precision=3, info="0.0 is left, 1.0 is right")
                        char_y4 = gr.Number(label="Y Position", value=0.5, precision=3, info="0.0 is top, 1.0 is bottom")

                with gr.TabItem("Character 5"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            char_prompt5 = gr.Textbox(
                                label="Character 5 Prompt", 
                                placeholder="e.g. 1girl, red hair...",
                                lines=2
                            )
                        with gr.Column(scale=2):
                            char_neg5 = gr.Textbox(
                                label="Character 5 Negative", 
                                placeholder="Optional undesired tags",
                                lines=2
                            )
                    with gr.Row():
                        char_x5 = gr.Number(label="X Position", value=0.5, precision=3, info="0.0 is left, 1.0 is right")
                        char_y5 = gr.Number(label="Y Position", value=0.5, precision=3, info="0.0 is top, 1.0 is bottom")

                with gr.TabItem("Character 6"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            char_prompt6 = gr.Textbox(
                                label="Character 6 Prompt", 
                                placeholder="e.g. 1girl, red hair...",
                                lines=2
                            )
                        with gr.Column(scale=2):
                            char_neg6 = gr.Textbox(
                                label="Character 6 Negative", 
                                placeholder="Optional undesired tags",
                                lines=2
                            )
                    with gr.Row():
                        char_x6 = gr.Number(label="X Position", value=0.5, precision=3, info="0.0 is left, 1.0 is right")
                        char_y6 = gr.Number(label="Y Position", value=0.5, precision=3, info="0.0 is top, 1.0 is bottom")

        with gr.Accordion("Reference Images", open=False, elem_id="ref_images_container") as ref_container:
            gr.Markdown("You may add up to 5 reference images. These are used for image-to-image generation and style transfer.")
            with gr.Column():
                with gr.Row():
                    with gr.Column(scale=2):
                        ref_img1 = gr.Image(label="Reference Image 1", type="pil", visible=True, elem_id="ref_img1")
                    with gr.Column(scale=1):
                        with gr.Group():
                            ref_info1 = gr.Slider(0, 1, 0.0, step=0.01, label="Info Extracted", visible=True, info="How much semantic information to extract")
                            ref_strength1 = gr.Slider(0, 1, 0.0, step=0.01, label="Reference Strength", visible=True, info="How strongly to apply the reference")

                with gr.Row(visible=False) as ref_row2:
                    with gr.Column(scale=2):
                        ref_img2 = gr.Image(label="Reference Image 2", type="pil", visible=True, elem_id="ref_img2")
                    with gr.Column(scale=1):
                        with gr.Group():
                            ref_info2 = gr.Slider(0, 1, 0.0, step=0.01, label="Info Extracted", visible=True)
                            ref_strength2 = gr.Slider(0, 1, 0.0, step=0.01, label="Reference Strength", visible=True)

                with gr.Row(visible=False) as ref_row3:
                    with gr.Column(scale=2):
                        ref_img3 = gr.Image(label="Reference Image 3", type="pil", visible=True, elem_id="ref_img3")
                    with gr.Column(scale=1):
                        with gr.Group():
                            ref_info3 = gr.Slider(0, 1, 0.0, step=0.01, label="Info Extracted", visible=True)
                            ref_strength3 = gr.Slider(0, 1, 0.0, step=0.01, label="Reference Strength", visible=True)

                with gr.Row(visible=False) as ref_row4:
                    with gr.Column(scale=2):
                        ref_img4 = gr.Image(label="Reference Image 4", type="pil", visible=True, elem_id="ref_img4")
                    with gr.Column(scale=1):
                        with gr.Group():
                            ref_info4 = gr.Slider(0, 1, 0.0, step=0.01, label="Info Extracted", visible=True)
                            ref_strength4 = gr.Slider(0, 1, 0.0, step=0.01, label="Reference Strength", visible=True)

                with gr.Row(visible=False) as ref_row5:
                    with gr.Column(scale=2):
                        ref_img5 = gr.Image(label="Reference Image 5", type="pil", visible=True, elem_id="ref_img5")
                    with gr.Column(scale=1):
                        with gr.Group():
                            ref_info5 = gr.Slider(0, 1, 0.0, step=0.01, label="Info Extracted", visible=True)
                            ref_strength5 = gr.Slider(0, 1, 0.0, step=0.01, label="Reference Strength", visible=True)

                add_ref = gr.Button("Add Another Reference")
        
        def update_uc_choices(selected_model):
            if selected_model == "NAI V4.5 Curated":
                return gr.update(choices=["Heavy", "Light", "Human Focus", "None"], value="Light")
            elif selected_model in ("NAI V4 Curated", "NAI V4 Full"):
                return gr.update(choices=["Heavy", "Light", "None"], value="Light")
            else:  # NAI V3
                return gr.update(choices=["Heavy", "Light", "Human Focus", "None"], value="Light")
        
        # Function to update scheduler based on model
        def update_scheduler(selected_model):
            if selected_model in ("NAI V4 Curated", "NAI V4 Full", "NAI V4.5 Curated"):
                return gr.update(value="karras")
            else:
                return gr.update(value="native")

        # Function to update sampler, width, and height based on model
        def update_gen_defaults(selected_model):
            if selected_model in ("NAI V4 Curated", "NAI V4 Full", "NAI V4.5 Curated"):
                return gr.update(value="k_euler_ancestral"), gr.update(value=832), gr.update(value=1216)
            else:
                return gr.update(value="k_euler"), gr.update(value=832), gr.update(value=1216)

        model_selector.change(update_uc_choices, inputs=model_selector, outputs=controls[3])
        model_selector.change(update_scheduler, inputs=model_selector, outputs=scheduler)
        model_selector.change(
            update_gen_defaults,
            inputs=model_selector,
            outputs=[controls[9], controls[6], controls[7]]  # sampler, width, height
        )

        model_selector.change(
            lambda m: gr.update(visible=(m == "NAI V3")),
            inputs=model_selector,
            outputs=ref_container
        )
        model_selector.change(
            lambda m: gr.update(visible=(m in ("NAI V4 Curated", "NAI V4 Full", "NAI V4.5 Curated"))),
            inputs=model_selector,
            outputs=char_container
        )

        ref_counter = gr.State(value=1)

        def reveal_next(counter):
            new_counter = counter + 1 if counter < 5 else counter
            out_updates = []
            for group in range(2, 6): 
                visible = new_counter >= group
                if group == 2:
                    out_updates.append(gr.update(visible=visible))  # ref_row2
                elif group == 3:
                    out_updates.append(gr.update(visible=visible))  # ref_row3
                elif group == 4:
                    out_updates.append(gr.update(visible=visible))  # ref_row4
                elif group == 5:
                    out_updates.append(gr.update(visible=visible))  # ref_row5
            return [new_counter] + out_updates

        add_ref.click(
            fn=reveal_next,
            inputs=ref_counter,
            outputs=[ref_counter, ref_row2, ref_row3, ref_row4, ref_row5]
        )
        gen_btn.click(
            generate,
            modes + controls + adv_controls + [
                ref_img1, ref_info1, ref_strength1,
                ref_img2, ref_info2, ref_strength2,
                ref_img3, ref_info3, ref_strength3,
                ref_img4, ref_info4, ref_strength4,
                ref_img5, ref_info5, ref_strength5,
                use_ai_char,
                char_prompt1, char_neg1, char_x1, char_y1,
                char_prompt2, char_neg2, char_x2, char_y2,
                char_prompt3, char_neg3, char_x3, char_y3,
                char_prompt4, char_neg4, char_x4, char_y4,
                char_prompt5, char_neg5, char_x5, char_y5,
                char_prompt6, char_neg6, char_x6, char_y6,
            ],
            image
        )
    return page



def util_ui():
    with gr.Blocks() as page:
        gr.Markdown("# WIP")
    return page


def ui():
    with gr.Blocks(
        title="NAI Client by Kohaku",
        theme=gr.themes.Soft(),
        css=open("client.css", "r", encoding="utf-8").read(),
    ) as website:
        with gr.Tabs(elem_id="main-tabs"):
            with gr.Tab("Main", elem_classes="page-tab"):
                main_ui()
            with gr.Tab("Util", elem_classes="page-tab"):
                util_ui()
    return website


if __name__ == "__main__":
    extension.load_extensions()
    website = ui()
    gr_thread = Thread(
        target=website.launch,
        daemon=True,
        kwargs={"inbrowser": not client_config.get("use_standalone_window", False)},
    )
    gr_thread.start()
    while not website.local_url:
        time.sleep(0.01)
    if client_config.get("use_standalone_window", False):
        webview.create_window(
            "NAI Client",
            website.local_url + "?__theme=dark",
            resizable=True,
            zoomable=True,
        )
        webview.start()
    else:
        input("Press Enter to close gradio")
