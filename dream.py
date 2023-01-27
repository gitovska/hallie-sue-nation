import jax
import numpy as np
import jax.numpy as jnp
from flax.jax_utils import replicate
from flax.training.common_utils import shard
import requests
from io import BytesIO
from PIL import Image
from diffusers import FlaxStableDiffusionImg2ImgPipeline

def create_key(seed=0):
    return jax.random.PRNGKey(seed)
rng = create_key(0)

image_path = "../zihang.jpg"
with open(image_path, "rb") as ib:
    init_img = Image.open(BytesIO(ib.read())).convert("RGB")
init_img = init_img.resize((768, 512))

prompts = "superheros are people who build things in the style of a dream"

pipeline, params = FlaxStableDiffusionImg2ImgPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", revision="bf16", dtype=jax.numpy.bfloat16
)

num_samples = jax.device_count()
rng = jax.random.split(rng, jax.device_count())
prompt_ids, processed_image = pipeline.prepare_inputs(prompt=[prompts]*num_samples, image = [init_img]*num_samples)
p_params = replicate(params)
prompt_ids = shard(prompt_ids)
processed_image = shard(processed_image)

output = pipeline(
    prompt_ids=prompt_ids, 
    image=processed_image, 
    params=p_params, 
    prng_seed=rng, 
    strength=0.5, 
    num_inference_steps=150, 
    jit=True, 
    height=512,
    width=768).images

output_images = pipeline.numpy_to_pil(np.asarray(output.reshape((num_samples,) + output.shape[-3:])))
for i in range(len(output_images)):
    output_images[i].save(f"../output/{i}.bmp")
