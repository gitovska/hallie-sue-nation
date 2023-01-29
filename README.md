# hallie-sue-nation

Hallie Sue Nation is a Twitter bot that replies to people's tweets of dreams and accompanying images with a pictorial dream sequence.

https://twitter.com/HallieSueNation

She is deployed on a Google Cloud TPU, and uses its eight cores simultaneously with Stable Diffusion's (v1.4) Image2Image pipeline.
This produces eight generated images concurrently using eight prompts based on the original tweet, generated by OpenAI's GPT3 Curie Text Generation Model.

This repository contains a Dockerfile, which will allow you to run Hallie Sue Nation in a Docker Container on a TPU equipped system.

## Set-up

Clone this repository and generate a docker image with the Docker file.

Run the docker image. Hallie Sue will periodically download new tweets in which she is mentioned and begin dreaming.