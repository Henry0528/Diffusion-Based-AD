# imports
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
from diffusers import UNet2DModel
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid

from loader.loader import MVTecDataset
from pipeline_reconstruction_ddim import DDIMReconstructionPipeline
from schedulers.scheduling_ddim import DDIMScheduler

# dataset
TARGET_RESOLUTION = 128
STEPS_TO_REGENERATE = 300       # 200
RECON_WEIGHT = 1               # 1
DATASET_NAME = "cable"
STATES = ["cable_swap"]
CHECKPOINT_PATH = "cable_1_1692802490/epoch_300.pt"    # hazelnut_1_1692278910/epoch_300.pt
NUM_TRAIN_STEPS, BETA_SCHEDULE = 1000, "linear"
RANDOM_FLIP = False
ETA = 0  # 1 = DDPM, 0 = DDIM (no noise added during denoising process)

augmentations = transforms.Compose(
    [
        transforms.Resize(TARGET_RESOLUTION, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip() if RANDOM_FLIP else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


def transform_images(imgs):
    return [augmentations(image.convert("RGB")) for image in imgs]


# data loader
data_train = MVTecDataset("C:/Users/nilsb/Documents/mvtec_anomaly_detection.tar", True, f"{DATASET_NAME}", ["good"],
                          transform_images)
train_loader = DataLoader(data_train, batch_size=8, shuffle=True)
test_data = MVTecDataset("C:/Users/nilsb/Documents/mvtec_anomaly_detection.tar", False, f"{DATASET_NAME}", STATES,
                         transform_images)
test_loader = DataLoader(test_data, batch_size=8, shuffle=False)

# set model, optimizer, scheduler
model_config = ""
model = UNet2DModel(
    sample_size=TARGET_RESOLUTION,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(128, 128, 256, 256, 512, 512),
    down_block_types=(
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    )
)

model.load_state_dict(torch.load(f"checkpoints/{CHECKPOINT_PATH}"))
model.eval()
model.to("cuda")
# noise_scheduler = DBADScheduler(NUM_TRAIN_STEPS, beta_schedule=BETA_SCHEDULE)
noise_scheduler = DDIMScheduler()

def generate_samples(model, noise_scheduler, plt_title, original_images):
    pipeline = DDIMReconstructionPipeline(
        unet=model,
        scheduler=noise_scheduler,
    )

    generator = torch.Generator(device=pipeline.device).manual_seed(0)
    # run pipeline in inference (sample random noise and denoise)
    images = pipeline(
        batch_size=8,
        generator=generator,
        num_inference_steps=100,
        original_images=original_images.to(model.device),
        eta=ETA,
        start_at_timestep=STEPS_TO_REGENERATE,
        output_type="numpy",
    ).images

    images_processed = (images * 255).round().astype("int")
    images = torch.from_numpy(images_processed)
    images = torch.permute(images, (0, 3, 1, 2))

    original_images = transforms.Normalize([-0.5 * 2], [2])(original_images)
    originals = (original_images * 255).round().type(torch.int32)

    diff_map = (originals - images) ** 2
    diff_map = diff_map / torch.amax(diff_map, dim=(2, 3)).reshape(-1, 3, 1, 1)   # per channel and image
    # diff_map = diff_map / torch.amax(diff_map, dim=(1, 2, 3)).reshape(-1, 1, 1, 1)  # per image
    diff_map = (diff_map * 255).round()
    diff_map = transforms.functional.rgb_to_grayscale(diff_map).to(torch.uint8)

    grid = make_grid(torch.cat((images.to(torch.uint8), originals.to(torch.uint8)), 0), 4)
    show(grid, plt_title)
    grid = make_grid(diff_map, 4)
    show(grid, plt_title + ' mask')


def show(imgs, title):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.title(title)
    plt.show()


def main():
    # train loop
    print("**** starting inference *****")

    with torch.no_grad():
        # validate and generate images
        noise_scheduler_inference = DDIMScheduler(NUM_TRAIN_STEPS, beta_schedule=BETA_SCHEDULE, reconstruction_weight=RECON_WEIGHT)
        # noise_scheduler_inference.set_timesteps(timesteps=list(range(0, 200, 1)).reverse())
        generate_samples(model, noise_scheduler_inference, f"Test samples ", next(iter(test_loader))[0])


if __name__ == '__main__':
    main()
