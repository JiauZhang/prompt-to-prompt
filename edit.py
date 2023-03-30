import torch, argparse
from diffusers import StableDiffusionPipeline, DDIMScheduler
from utils import image_grid, latent_to_image

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--model_id', type=str, default='runwayml/stable-diffusion-v1-5')
parser.add_argument('--prompt', type=str, default='A photo of a butterfly on a flower.')
parser.add_argument('--num_inference_steps', type=int, default=50)

args = parser.parse_args()
device = args.device
model_id = args.model_id
prompt = args.prompt
steps = args.num_inference_steps

ddim_scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
SD = StableDiffusionPipeline.from_pretrained(
    model_id, scheduler=ddim_scheduler, torch_dtype=torch.float16,
    cache_dir='.',
).to(device)
generator = torch.Generator(device).manual_seed(20230330)

images = SD(
    prompt, generator=generator, num_inference_steps=steps,
).images
image = image_grid(images, rows=1, cols=1)
image.save('woman.png')
