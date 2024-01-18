import random
import numpy as np
# from datasets import load_dataset, Dataset
import os
import torch
import time
from torch import autocast
from diffusers.models import AutoencoderKL
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, DiffusionPipeline, StableDiffusionXLPipeline, \
    EulerAncestralDiscreteScheduler, PixArtAlphaPipeline, AutoPipelineForText2Image, StableDiffusionImg2ImgPipeline, \
    MotionAdapter, AnimateDiffPipeline, DDIMScheduler
from termcolor import colored
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
from PIL import Image               # to load images
from IPython.display import display # to display images
import gc
auth_token = "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"


def set_seed(seed: int):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def generate_video(pipe, idevice, iprompt, num_frames, seed):
    set_seed(seed)  # Set the seed before generating video
    generator = torch.Generator(device=idevice).manual_seed(seed)

    # Generate video frames
    with autocast(idevice):
        video_frames = pipe(iprompt, num_frames=num_frames, generator=generator).frames[0]

    # Export frames to video
    video_path = f"video_{time.strftime('%Y%m%d-%H%M%S')}.mp4"
    export_to_video(video_frames, video_path)

    return video_path


# Load the text-to-video model
def load_video_model():
    adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)

    # model_id = "damo-vilab/text-to-video-ms-1.7b"
    model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
    device = "cuda"
    # pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
    pipe = AnimateDiffPipeline.from_pretrained(model_id, motion_adapter=adapter, torch_dtype=torch.float16)
    scheduler = DDIMScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
        clip_sample=False,
        timestep_spacing="linspace",
        beta_schedule="linear",
        steps_offset=1,
    )
    pipe.scheduler = scheduler

    # memory optimization
    pipe.enable_vae_slicing()
    pipe.to(device)
    return pipe


def load_model(model_name, model_id):
    print(model_name, "********************************model_id")
    if model_name == "Anime":
        vae_id = "madebyollin/sdxl-vae-fp16-fix"
        vae = AutoencoderKL.from_pretrained(vae_id)
        device = "cuda"
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_auth_token=auth_token,
            safety_checker=None,
            feature_extractor=None,
            use_safetensors=True,
            requires_safety_checker=True,
            vae=vae,
        )
        # You may need to customize this part based on your requirements for Anime model
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.to(device)
        return pipe
    # elif model_id == "Pixart":
    #     vae_id = "madebyollin/sdxl-vae-fp16-fix"
    #     vae = AutoencoderKL.from_pretrained(vae_id)
    #     device = "cuda"
    #     pipe = PixArtAlphaPipeline.from_pretrained(
    #         model_id,
    #         torch_dtype=torch.float16,
    #         use_auth_token=auth_token,
    #         safety_checker=None,
    #         feature_extractor=None,
    #         use_safetensors=True,
    #         requires_safety_checker=True,
    #         vae=vae,
    #     )
    #     # You may need to customize this part based on your requirements for Anime model
    #     pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    # elif model_id == "OpenDalle":
    #     vae_id = "madebyollin/sdxl-vae-fp16-fix"
    #     vae = AutoencoderKL.from_pretrained(vae_id)
    #     device = "cuda"
    #     pipe = AutoPipelineForText2Image.from_pretrained(
    #         model_id,
    #         torch_dtype=torch.float16,
    #         use_auth_token=auth_token,
    #         safety_checker=None,
    #         feature_extractor=None,
    #         use_safetensors=True,
    #         requires_safety_checker=True,
    #         vae=vae,
    #     )
    #     # You may need to customize this part based on your requirements for Anime model
    #     pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    elif model_name == "sdxl":
        # load both base & refiner
        cache_dir = "hugging_model(sanjay)"
        pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16,
                                                 use_safetensors=True, variant="fp16")

        # base = DiffusionPipeline.from_pretrained(
        #     "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, cache_dir=cache_dir, variant="fp16", use_safetensors=True
        # )
        # base.to("cuda")
        device = "cuda"
        # pipe = DiffusionPipeline.from_pretrained(
        #     "stabilityai/stable-diffusion-xl-refiner-1.0",
        #     text_encoder_2=base.text_encoder_2,
        #     vae=base.vae,
        #     torch_dtype=torch.float16,
        #     cache_dir=cache_dir,
        #     use_safetensors=True,
        #     variant="fp16",
        # )
        # gc.collect()
        # torch.cuda.empty_cache()
        pipe.to(device)
        return pipe





    else:
        print(model_id, "_____________________________^^^^^^^^^^^^model_id")
        # For other models, use the original implementation
        vae_id = "stabilityai/sd-vae-ft-mse"
        vae = AutoencoderKL.from_pretrained(vae_id)
        device = "cuda"
        euler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_auth_token=auth_token,
            safety_checker=None,
            feature_extractor=None,
            use_safetensors=False,
            requires_safety_checker=False,
            scheduler=euler,
            vae=vae,
        )

        pipe.to(device)
        return pipe


def img_to_img_transformation(image, prompt, generate):
    device = "cuda"
    # model_id_or_path = "stabilityai/stable-diffusion-2-1"
    model_id_or_path = "digiplay/Photon_v1"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
    pipe = pipe.to(device)

    # Load the previously generated image
    # print(type(image),',,,,,,,',image,'------------------------------------------------image')

    # init_image = Image.open(image).convert("RGB")
    init_image = generate.resize((768, 512))  # Resize if necessary

    # Perform img2img transformation
    transformed_images = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images

    # Save the transformed image
    transformed_image_path = "transformed_" + image
    transformed_images[0].save(transformed_image_path)
    return transformed_image_path


#
def generate_image(pipe, idevice, iprompt, inegative, isteps, iguidance, seed):
    print("karma_______-------------++++++++++++++++")
    set_seed(seed)  # Set seed before generating images
    if not iprompt:
        iprompt = "face, very expressive, portrait photography, world photography, soft studio lighting, intricate, beautiful, award winning, stunning, stock film, 8k, centered, amazing, impressive, awesome, highly detailed, fantastic, overwhelming, masterpiece, subject in frame"
    if inegative == '!no':
        inegative = None
    elif not inegative:
        inegative = "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face, artifacts, jpeg noise, bad eyes, text"
    if not isteps:
        isteps = 50
    if not iguidance:
        iguidance = 8.5
    # with autocast(idevice):
    print("hateeeeee???????????????????????????????????/")
    image = pipe(prompt=iprompt).images[0]
    print(image, type(image), '--------------new------------------')
    inow = time.strftime("%Y%m%d-%H%M%S")
    ifilename = inow + '.png'
    image.save(ifilename)
    print(image, type(image), '--------------------------------')

    with open(inow + '.txt', 'w', encoding='utf-8') as f:
        f.write("Prompt   : " + str(iprompt) + "\n")
        f.write("Negative : " + str(inegative) + "\n")
        f.write("Steps    : " + str(isteps) + "\n")
        f.write("Guidance : " + str(iguidance) + "\n")
        f.write("Seed     : " + str(seed) + "\n")
        f.write("Output   : " + str(ifilename) + "\n")
    f.close()
    return ifilename, image


if __name__ == "__main__":
    print(colored("\n\n\n############### SD NANO ###############", "cyan", attrs=["reverse", "bold"]))

    model_options = {
        "Photon_v1": "digiplay/Photon_v1",
        "stable-diffusion-2-1": "stabilityai/stable-diffusion-2-1",
        "Anime": "cagliostrolab/animagine-xl-3.0",
        "sdxl": "stabilityai/stable-diffusion-xl-base-1.0"
        # "Pixart": "PixArt-alpha/PixArt-XL-2-1024-MS",
        # "OpenDalle": "dataautogpt3/OpenDalleV1.1",

        # Add more models as needed
    }

    while True:
        try:
            choice = input(colored("Choose mode (1 for text-to-image, 2 for text-to-video): ", "green")).strip()
            if choice == '1':
                print("Available models:")
                for i, (key, value) in enumerate(model_options.items(), 1):
                    print(f"{i}. {key}")
                # Text-to-Image Generation
                model_choice = int(input(colored("\nEnter the number of the model you want to use: ", "green")))
                selected_model_name = list(model_options.keys())[model_choice - 1]
                selected_model_id = list(model_options.values())[model_choice - 1]
                print(f"\nSelected Model: {selected_model_name}")

                selected_model = load_model(selected_model_name, selected_model_id)
                selected_device = "cuda"  # or "cpu"
                print(colored("\nDevice: " + selected_device, "cyan"))
                # prompt = input(colored("\nPrompt (!exit to quit, Enter for example): ", "green"))
                prompt = input(colored("\nPrompt (!exit to quit, Enter for example): ", "green")).strip()
                if prompt == '!exit':
                    exit()

                negative = input(colored("Negative prompt (Enter for default, !no for none): ", "green"))
                steps = int(input(colored("Sampling Steps (Enter for 50): ", "green")) or 50)
                guidance = float(input(colored("Guidance scale (Enter for 8.5): ", "green")) or 8.5)
                seed_input = input(colored("Enter seed (Leave blank for random seed): ", "green")).strip()
                seed = int(seed_input) if seed_input else random.randint(1, 1_000_000)
                print(prompt, '8888888888888888888888888888888888888888')
                print(selected_model, '<<<<<<<<<selected_model<<<<<<<<', selected_device,
                      '----------selected_device------', negative, '>>>>>>>>>>>.negative>', guidance,
                      'guidance.............e.', seed, 'seed888888')
                filename, generateImage = generate_image(pipe=selected_model, idevice=selected_device, iprompt=prompt,
                                                         inegative=negative, isteps=steps, iguidance=guidance,
                                                         seed=seed)

                pil_im = Image.open(filename)
                pil_im.show()
                print('---------------------------------------------------------------------')
                display(pil_im)
                print("Done creating ", filename, type(filename))

                # Prompt for img2img transformation
                img2img_prompt = input("Enter the prompt for img2img transformation: ")

                # Run img2img transformation
                transformed_image_path = img_to_img_transformation(filename, img2img_prompt, generateImage)
                print("Transformed image saved as " + transformed_image_path)

            elif choice == '2':
                # Text-to-Video Generation
                video_model = load_video_model()
                video_prompt = input("Enter prompt for video generation: ")
                num_frames = int(input("Enter number of frames for video: "))
                seed = random.randint(1, 1_000_000)  # Or get user input for seed
                video_path = generate_video(video_model, "cuda", video_prompt, num_frames, seed)
                print(f"Video created: {video_path}")

            else:
                print("Invalid choice. Please enter 1 or 2.")
        except (ValueError, IndexError):
            print("Invalid input. Please enter a valid number.")



