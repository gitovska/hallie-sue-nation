import jax
import numpy as np
from flax.jax_utils import replicate
from flax.training.common_utils import shard
import requests
import os
from io import BytesIO
from PIL import Image
from diffusers import FlaxStableDiffusionImg2ImgPipeline


# The following guide to running Stable Diffusion on a TPU used as the basis for the Dreamer.
# https://huggingface.co/blog/stable_diffusion_jax

# The stable_diffusion_jax colab notebook was modified with the help of the FlaxStableDiffusionImg2ImgPipeline
# sample code in the readme of the huggingface/diffusers repo
# github.com/huggingface/diffusers

class Dreamer:

    def __init__(self):
        self.__device_count = jax.device_count()
        rng = self._create_key(0)
        self.__rng = jax.random.split(rng, self.__device_count)
        self.__pipeline, self.__params = FlaxStableDiffusionImg2ImgPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", revision="bf16", dtype=jax.numpy.bfloat16
        )
        self.__strength = 0.5
        self.__num_inference_steps = 150
        self.__jit = True
        self.__height = 512
        self.__width = 768

    def _create_key(self, seed=0) -> jax.random.PRNGKeyArray:
        return jax.random.PRNGKey(seed)

    def _get_image(self, url) -> Image:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img = img.resize((768, 512))
        return img

    def _image_grid(self, imgs: list[Image], rows: int, cols: int):
        w, h = imgs[0].size
        grid = Image.new('RGB', size=(cols * w, rows * h))
        for i, img in enumerate(imgs): grid.paste(img, box=(i % cols * w, i // cols * h))
        return grid

    def dream(self, prompts: list[str], tweet_id: int, image_url: str):
        init_image = self._get_image(image_url)
        prompt_ids, processed_image = self.__pipeline.prepare_inputs(prompt=prompts,
                                                                     image=[init_image] * self.__device_count)
        p_params = replicate(self.__params)
        prompt_ids = shard(prompt_ids)
        processed_image = shard(processed_image)

        output = self.__pipeline(
            prompt_ids=prompt_ids,
            image=processed_image,
            params=p_params,
            prng_seed=self.__rng,
            strength=self.__strength,
            num_inference_steps=self.__num_inference_steps,
            jit=self.__jit,
            height=self.__height,
            width=self.__width).images

        output_images = self.__pipeline.numpy_to_pil(np.asarray(output.reshape((self.__device_count,) + output.shape[-3:])))
        os.makedirs(f"data/output/{tweet_id}")
        for i in range(len(output_images)):
            output_images[i].save(f"./data/output/{tweet_id}/{tweet_id}_dream_{i+1}.bmp")

        grid = self._image_grid(output_images, 4, 2)
        grid.save(f"./data/output/{tweet_id}/{tweet_id}_dream_grid.bmp")
