import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import os
from utils import load_config, ensure_output_directory
import subprocess
import argparse

# Load configuration
config = load_config()
ensure_output_directory(config["output_dir"])

# ✅ Load Stable Diffusion model before modifying scheduler
pipe = StableDiffusionPipeline.from_pretrained(config["model_name"])

# ✅ Apply DPM++ solver (faster & better quality)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# ✅ Move model to CPU (or CUDA if available)
pipe.to("cpu")  # Use "cuda" if running on GPU
def generate_image(prompt, output_path):
    print(f"🔹 Generating image for prompt: {prompt}")

    try:
        # Generate image
        output = pipe(prompt, num_inference_steps=config["num_inference_steps"], guidance_scale=config["guidance_scale"])
        image = output.images[0] if output.images else None

        if image is None:
            print("❌ ERROR: The pipeline did NOT generate an image.")
            return
        
        print(f"✅ Image generated successfully! Type: {type(image)}")

        # Verify image type
        if not isinstance(image, torch.Tensor):
            print(f"✅ Saving image to: {output_path}")
            image.save(output_path)

            # Confirm that the image is saved
            if os.path.exists(output_path):
                print(f"✅ Image saved successfully at {output_path}")
            else:
                print("❌ ERROR: Image was NOT saved!")

        else:
            print("❌ ERROR: Unexpected image format!")

        # Upscale if enabled
        if config["upscale"]:
            output_path = upscale_image(output_path)

    except Exception as e:
        print(f"❌ ERROR: {e}")

    return output_path
def upscale_image(image_path):
    output_path = image_path.replace(".png", "_upscaled.png")
    realesrgan_path = os.path.join("real-esrgan", "realesrgan-ncnn-vulkan.exe")

    if not os.path.exists(realesrgan_path):
        print("❌ ERROR: Real-ESRGAN executable not found! Skipping upscaling.")
        return image_path  # Return original image if upscaling is unavailable

    command = f'"{realesrgan_path}" -i "{image_path}" -o "{output_path}"'
    print(f"🔹 Running upscaling command: {command}")
    subprocess.run(command, shell=True)

    if os.path.exists(output_path):
        print(f"✅ Upscaled image saved at: {output_path}")
        return output_path
    else:
        print("❌ ERROR: Upscaling failed!")
        return image_path  # Return original image if upscaling fails
# ✅ Ensure the script actually runs when called from the terminal
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images from text using Stable Diffusion.")
    parser.add_argument("prompt", type=str, help="Text prompt for the image")
    parser.add_argument("--output", type=str, default="output.png", help="Output file path")
    args = parser.parse_args()

    generate_image(args.prompt, os.path.join(config["output_dir"], args.output))
