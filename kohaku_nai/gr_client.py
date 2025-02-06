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


def control_ui():
    prompt = gr.TextArea(
        label="Prompt",
        lines=3,
        value=client_config["default_prompt"],
    )
    neg_prompt = gr.TextArea(
        label="Negative Prompt",
        lines=1,
        value=client_config["default_neg"],
    )
    with gr.Row():
        with gr.Column(scale=3, min_width=160):
            enable_quality_tags = gr.Checkbox(label="Enable Quality Tags", value=True)
        with gr.Column(scale=5, min_width=360):
            neg_preset = gr.Radio(
                choices=["Heavy", "Light", "Human Focus", "None"],
                value="Light",
                label="UC Preset",
            )
    with gr.Row():
        seed = gr.Number(label="Seed", value=-1, step=1, maximum=2**32 - 1, minimum=-1)
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
        scale = gr.Slider(label="Scale", value=5.0, minimum=1, maximum=10, step=0.1)
        steps = gr.Slider(label="Steps", value=28, minimum=1, maximum=50, step=1)
    with gr.Row():
        width = gr.Slider(label="Width", value=1024, minimum=64, maximum=2048, step=64)
        height = gr.Slider(
            label="Height", value=1024, minimum=64, maximum=2048, step=64
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
            gr.Markdown("### Advance Generation settings")
            with gr.Row():
                model_selector = gr.Dropdown(
                    choices=["nai-diffusion-3", "nai-diffusion-4-curated-preview"],
                    value="nai-diffusion-3",
                    label="Model",
                    interactive=True,
                )
                variety_chk = gr.Checkbox(False, label="Variety+")
                

            scheduler = gr.Dropdown(
                choices=["native", "karras", "exponential", "polyexponential"],
                value="native",
                label="Scheduler",
                interactive=True,
            )
            with gr.Row():
                smea = gr.Checkbox(False, label="SMEA")
                dyn = gr.Checkbox(False, label="SMEA DYN")
                dyn_threshold = gr.Checkbox(False, label="Dynamic Thresholding")
            with gr.Row():
                cfg_rescale = gr.Slider(0, 1, 0, step=0.01, label="CFG rescale")

        with gr.Column():
            gr.Markdown("### Client settings")
            mode = gr.Radio(
                ["remote", "local"], value=client_config["mode"], label="Mode"
            )
            backend = gr.Radio(
                ["curl_cffi", "httpx"],
                value=client_config.get("backend", "curl_cffi"),
                label="Http Backend",
                info='use "httpx" if you met issues with "curl_cffi"',
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
                token = gr.Textbox(client_config["token"], label="Token")

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
    ref_img5, ref_info5, ref_strength5
):

    prompt = extension.process_prompt(prompt)
    neg_prompt = extension.process_prompt(neg_prompt)

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
            # (Assume r_img is already a base64 string or you encode it as needed)
            reference_images.append(r_img)
            reference_info.append(r_info)
            reference_strength.append(r_str)


    if mode == "remote":
        if (pswd := end_point_pswd) or (pswd := client_config["end_point_pswd"]):
            await set_client(backend, end_point, pswd)
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
            model=model,
            variety=variety,
            reference_image_multiple=reference_images,
            reference_information_extracted_multiple=reference_info,
            reference_strength_multiple=reference_strength,
            extra_infos=extra_info_json,
        )
        if not isinstance(img_data, bytes):
            print(f"Error Message: {img_data}")
            return None
    elif mode == "local":
        await set_client(backend, token=token)
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
            model=model,
            variety=variety,
            reference_image_multiple=reference_images,
            reference_information_extracted_multiple=reference_info,
            reference_strength_multiple=reference_strength,
            extra_infos=extra_info_json,
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
                        # Capture the advanced controls; note that model_selector is returned as index 6
                        adv_controls, modes = settings_ui()
                        # Unpack for clarity:
                        (scheduler, smea, dyn, dyn_threshold, cfg_rescale,
                         extra_info_json, model_selector, variety_chk) = adv_controls
            with gr.Column():
                gen_btn = gr.Button(value="Generate", variant="primary")
                image = preview_ui()
        
        # Define the Reference Images accordion (placed outside the tabs)
        with gr.Accordion("Reference Images (v3 only)", open=False, elem_id="ref_images_container") as ref_container:
            gr.Markdown("You may add up to 5 reference images:")
            with gr.Column():
                # Create five groups; initially only the first group is visible.
                ref_img1 = gr.Image(label="Reference Image 1", type="pil", visible=True)
                ref_info1 = gr.Slider(0, 1, 0.0, step=0.01, label="Info Extracted", visible=True)
                ref_strength1 = gr.Slider(0, 1, 0.0, step=0.01, label="Reference Strength", visible=True)

                ref_img2 = gr.Image(label="Reference Image 2", type="pil", visible=False)
                ref_info2 = gr.Slider(0, 1, 0.0, step=0.01, label="Info Extracted", visible=False)
                ref_strength2 = gr.Slider(0, 1, 0.0, step=0.01, label="Reference Strength", visible=False)

                ref_img3 = gr.Image(label="Reference Image 3", type="pil", visible=False)
                ref_info3 = gr.Slider(0, 1, 0.0, step=0.01, label="Info Extracted", visible=False)
                ref_strength3 = gr.Slider(0, 1, 0.0, step=0.01, label="Reference Strength", visible=False)

                ref_img4 = gr.Image(label="Reference Image 4", type="pil", visible=False)
                ref_info4 = gr.Slider(0, 1, 0.0, step=0.01, label="Info Extracted", visible=False)
                ref_strength4 = gr.Slider(0, 1, 0.0, step=0.01, label="Reference Strength", visible=False)

                ref_img5 = gr.Image(label="Reference Image 5", type="pil", visible=False)
                ref_info5 = gr.Slider(0, 1, 0.0, step=0.01, label="Info Extracted", visible=False)
                ref_strength5 = gr.Slider(0, 1, 0.0, step=0.01, label="Reference Strength", visible=False)

                add_ref = gr.Button("Add Another Reference")
        
        # ----- CALLBACKS ADDED HERE -----
        # controls is the list returned by control_ui(); index 3 is neg_preset.
        def update_uc_choices(selected_model):
            if selected_model == "nai-diffusion-4-curated-preview":
                # For v4, only these three choices should be available.
                return gr.update(choices=["Heavy", "Light", "None"], value="Light")
            else:
                # For v3, include the additional "Human Focus" option.
                return gr.update(choices=["Heavy", "Light", "Human Focus", "None"], value="Light")

        # Assume controls[3] is neg_preset and model_selector comes from settings_ui().
        model_selector.change(update_uc_choices, inputs=model_selector, outputs=controls[3])

        # (1) Update the reference accordion visibility based on the model selection:
        model_selector.change(
            lambda m: gr.update(visible=(m == "nai-diffusion-3")),
            inputs=model_selector,
            outputs=ref_container
        )

        # (2) Simple example: clicking the "Add Another Reference" button reveals the second reference group.
        # (You can replicate or extend this logic for further groups.)
        # def reveal_next():
        #     # This callback simply returns updates to make group 2 visible.
        #     return (gr.update(visible=True),  # for ref_img2
        #             gr.update(visible=True),  # for ref_info2
        #             gr.update(visible=True))  # for ref_strength2

        # add_ref.click(
        #     fn=reveal_next,
        #     inputs=None,
        #     outputs=[ref_img2, ref_info2, ref_strength2]
        # )
        ref_counter = gr.State(value=1)

        def reveal_next(counter):
            # If there are fewer than 5 groups visible, increment counter
            new_counter = counter + 1 if counter < 5 else counter
            # For groups 2 to 5, set visibility based on whether new_counter is at least that number.
            out_updates = []
            for group in range(2, 6):  # Groups 2, 3, 4, 5
                visible = new_counter >= group
                # For each group, update its image, info, and strength.
                out_updates.extend([gr.update(visible=visible)] * 3)
            return new_counter, *out_updates

        # The callback will update the state and all groups for groups 2–5.
        add_ref.click(
            fn=reveal_next,
            inputs=ref_counter,
            outputs=[ref_counter,
                     ref_img2, ref_info2, ref_strength2,
                     ref_img3, ref_info3, ref_strength3,
                     ref_img4, ref_info4, ref_strength4,
                     ref_img5, ref_info5, ref_strength5]
        )
        # ----- END CALLBACKS -----

        # Now wire up the generate button click (make sure to include the new reference inputs
        # in the parameter list if you plan to pass them to generate())
        gen_btn.click(
            generate,
            modes + controls + adv_controls + [
                ref_img1, ref_info1, ref_strength1,
                ref_img2, ref_info2, ref_strength2,
                ref_img3, ref_info3, ref_strength3,
                ref_img4, ref_info4, ref_strength4,
                ref_img5, ref_info5, ref_strength5,
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
