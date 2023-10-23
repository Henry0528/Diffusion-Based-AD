# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Tuple, Union

import torch

from diffusers.utils import randn_tensor, numpy_to_pil
from diffusers.pipeline_utils import DiffusionPipeline, ImagePipelineOutput

from noise.simplex import simplexGenerator
from utils.pipeline_utils import DBADPipelineOutput


class DDIMReconstructionPipeline(DiffusionPipeline):
    r"""
    Pipeline for image generation.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        unet ([`UNet2DModel`]):
            A `UNet2DModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    def __init__(self, unet, scheduler, noise_kind='gaussian'):
        super().__init__()

        self.noise_kind = noise_kind
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
            self,
            batch_size: int = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            original_images: torch.Tensor = None,
            eta: float = 0.0,
            num_inference_steps: int = 50,
            start_at_timestep: int = 200,
            use_clipped_model_output: Optional[bool] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers. A value of `0` corresponds to
                DDIM and `1` corresponds to DDPM.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            use_clipped_model_output (`bool`, *optional*, defaults to `None`):
                If `True` or `False`, see documentation for [`DDIMScheduler.step`]. If `None`, nothing is passed
                downstream to the scheduler (use `None` for schedulers which don't support this argument).
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Example:

        ```py
        >>> from diffusers import DDIMPipeline
        >>> import PIL.Image
        >>> import numpy as np

        >>> # load model and scheduler
        >>> pipe = DDIMReconstructionPipeline.from_pretrained("fusing/ddim-lsun-bedroom")

        >>> # run pipeline in inference (sample random noise and denoise)
        >>> image = pipe(eta=0.0, num_inference_steps=50)

        >>> # process image to PIL
        >>> image_processed = image.cpu().permute(0, 2, 3, 1)
        >>> image_processed = (image_processed + 1.0) * 127.5
        >>> image_processed = image_processed.numpy().astype(np.uint8)
        >>> image_pil = PIL.Image.fromarray(image_processed[0])

        >>> # save image
        >>> image_pil.save("test.png")
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """

        # Sample gaussian noise to begin loop
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                original_images.shape[2],
                original_images.shape[3],
            )
        else:
            image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if self.noise_kind == 'gaussian':
            noise = randn_tensor(image_shape, generator=generator, device=self._execution_device, dtype=self.unet.dtype)
        elif self.noise_kind == 'simplex':
            noise = simplexGenerator.batch_3d_octaves(image_shape, 6, 0.6).to(self._execution_device)
        else:
            raise ValueError(f"Unknown noise kind '{self.noise_kind}', please use either 'gaussian' or 'simplex'")
        starting_step = torch.ones((image_shape[0],), dtype=torch.int64, device=self.device) * start_at_timestep
        image = self.scheduler.add_noise(original_images, noise, starting_step)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps, start_at_timestep)
        history = {'images': [], 'timesteps': []}

        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            model_output = self.unet(image, t).sample

            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # eta corresponds to η in paper and should be between [0, 1]
            # do x_t -> x_t-1
            image = self.scheduler.step(
                model_output, t, image, original_images, eta=eta, use_clipped_model_output=use_clipped_model_output,
                generator=generator
            ).prev_sample

            image_cp = post_process_img(image, output_type, np_ordering=False)
            history["images"].append(image_cp)
            history["timesteps"].append(t)

        history["images"].reverse()
        history["timesteps"].reverse()
        if not return_dict:
            return image_cp, history

        return DBADPipelineOutput(images=post_process_img(image, np_ordering=False), history=history)


def post_process_img(image, output_type="numpy", np_ordering=True):
    image_cp = (image / 2 + 0.5).clamp(0, 1)

    if np_ordering:
        image_cp = image_cp.permute(0, 2, 3, 1)

    if output_type== "numpy":
        image_cp = image_cp.cpu().numpy()

    if output_type == "pil":
        image_cp = numpy_to_pil(image_cp)

    return image_cp