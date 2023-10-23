# imports
import json
import os.path
import time
import argparse
from collections import Counter

import wandb
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

import inference_ddim
from pipe.train import train_step
from pipe.validate import validate_step
from diffusers import DDPMScheduler, UNet2DModel, get_scheduler
import diffusers
from tqdm import tqdm
from loader.loader import MVTecDataset
from utils.anomalies import diff_map_to_anomaly_map
from utils.files import save_args
from utils.visualize import generate_samples, plot_single_channel_imgs, plot_rgb_imgs, gray_to_rgb
from dataclasses import dataclass
from schedulers.scheduling_ddim import DDIMScheduler
from schedulers.scheduling_ddpm import DBADScheduler
from efficientnet_pytorch import EfficientNet
from torch.nn import functional as F

@dataclass
class TrainArgs:
    checkpoint_dir: str
    log_dir: str
    run_name: str
    mvtec_item: str
    resolution: int
    epochs: int
    save_n_epochs: int
    dataset_path: str
    train_steps: int
    beta_schedule: str
    device: str
    reconstruction_weight: float
    eta: float
    batch_size: int
    noise_kind: str
    plt_imgs: bool
    img_dir: str
    calc_val_loss: bool
    crop: bool


def parse_args() -> TrainArgs:
    parser = argparse.ArgumentParser(description='Add config for the training')
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints",
                        help='directory path to store the checkpoints')
    parser.add_argument('--log_dir', type=str, default="logs",
                        help='directory path to store logs')
    parser.add_argument('--run_name', type=str, required=True,
                        help='name of the run and corresponding checkpoints/logs that are created')
    parser.add_argument('--mvtec_item', type=str, required=True,
                        choices=["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
                                 "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper", "normal"],
                        help='name of the item within the MVTec Dataset to train on')
    parser.add_argument('--resolution', type=int, default=224,
                        help='resolution of the images to generate (dataset will be resized to this resolution during training)')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='epochs to train for')
    parser.add_argument('--save_n_epochs', type=int, default=50,
                        help='write a checkpoint every n-th epoch')
    parser.add_argument('--train_steps', type=int, default=1000,
                        help='number of steps for the full diffusion process')
    parser.add_argument('--beta_schedule', type=str, default="linear",
                        help='Type of schedule for the beta/variance values')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='directory path to the (mvtec) dataset')
    parser.add_argument('--device', type=str, default="cuda",
                        help='device to train on')
    parser.add_argument('--recon_weight', type=float, default=1, dest="reconstruction_weight",
                        help='Influence of the original sample during inference (doesnt affect training)')
    parser.add_argument('--eta', type=float, default=0,
                        help='Stochasticity parameter of DDIM, with eta=1 being DDPM and eta=0 meaning no randomness. Only used during inference, not training.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size during training')
    parser.add_argument('--noise_kind', type=str, default="gaussian",
                        choices=["simplex", "gaussian"],
                        help='Kind of noise to use for the noising steps.')
    parser.add_argument('--crop', action='store_true',
                        help='If set: the image will be cropped to the resolution instead of resized.')
    parser.add_argument('--plt_imgs', action='store_true',
                        help='If set: plot the images with matplotlib')
    parser.add_argument('--calc_val_loss', action='store_true',
                        help='If set: calculate not only the train loss, but also the validation loss during each epoch')
    parser.add_argument('--img_dir', type=str, default=None,
                        help='Directory to store the images created during the run. A new directory with the run-id will be created in this directory. If not used images wont be stored except for tensorboard.')

    return TrainArgs(**vars(parser.parse_args()))


def transform_imgs_test(imgs):
    augmentations = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    return [augmentations(image.convert("RGB")) for image in imgs]


def transform_imgs_train(imgs):
    augmentations = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    return [augmentations(image.convert("RGB")) for image in imgs]


def main(args: TrainArgs):
    # -------------      load data      ------------
    data_train = MVTecDataset(args.dataset_path, True, args.mvtec_item, ["good"],
                              transform_imgs_train)
    train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers = 24)
    test_data = MVTecDataset(args.dataset_path, False, 'bottle', ["all"],
                             transform_imgs_test)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers = 24)

    # ----------- set model, optimizer, scheduler -----------------
    channel_multiplier = {
        128: (128, 128, 256, 384, 512),
        256: (128, 128, 256, 256, 512, 512),
        224: (256, 512, 768, 1024)
    }
    down_blocks = ["DownBlock2D" for _ in channel_multiplier[args.resolution]]
    down_blocks[-2] = "AttnDownBlock2D"
    up_blocks = ["UpBlock2D" for _ in channel_multiplier[args.resolution]]
    up_blocks[1] = "AttnUpBlock2D"

    model_args = {
        "sample_size": args.resolution,
        "in_channels": 272,
        "out_channels": 272,
        "layers_per_block": 2,
        "block_out_channels": channel_multiplier[args.resolution],
        "down_block_types": down_blocks,
        "up_block_types": up_blocks
    }
    model = UNet2DModel(
        **model_args
    )
    feature_extractor = EfficientNet.from_pretrained('efficientnet-b4')
    noise_scheduler = DDPMScheduler(args.train_steps, beta_schedule=args.beta_schedule)
    inf_noise_scheduler = DDIMScheduler(args.train_steps, 150,
                                        beta_schedule=args.beta_schedule, timestep_spacing="leading",
                                        reconstruction_weight=args.reconstruction_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        weight_decay=1e-4,
        lr=1e-4,
        betas=(0.95, 0.999),
        eps=1e-08,
    )

    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=1500,
        num_training_steps=(len(train_loader) * args.epochs),
    )
    loss_fn = torch.nn.MSELoss()

    # additional info/util
    timestamp = str(time.time())[:11]
    writer = SummaryWriter(f'{args.log_dir}/{args.run_name}_{timestamp}')
    diffmap_blur = transforms.GaussianBlur(2 * int(4 * 4 + 0.5) + 1, 4)
    run_id = f"{args.run_name}_{timestamp}"
    print(diffusers.utils.logging.is_progress_bar_enabled())
    diffusers.utils.logging.disable_progress_bar()
    # use wandb to record the training loss
    wandb.init(
        # set the wandb project where this run will be logged
        project="diffusionAD",
        name="mvtec",
        # track hyperparameters and run metadata
        config={
            "dataset": "mvtec",
            "batch_size": 64
        }
    )
    # -----------------     train loop   -----------------
    print("**** starting training *****")
    print(f"run_id: {run_id}")
    save_args(args, f"{args.checkpoint_dir}/{args.run_name}_{timestamp}", "train_arg_config")
    save_args(model_args, f"{args.checkpoint_dir}/{args.run_name}_{timestamp}", "model_config")

    for epoch in range(args.epochs):
        model.train()
        model.to(args.device)
        feature_extractor.to(args.device)
        feature_extractor.eval()
        progress_bar = tqdm(total=len(train_loader) + len(test_loader))
        progress_bar.set_description(f"Epoch {epoch}")

        running_loss_train = 0

        for btc_num, (batch, _) in enumerate(train_loader):
            with torch.no_grad():
                image_features = list(feature_extractor.extract_endpoints(batch.to(args.device)).values())[:-2]
            for i in range(4):
                image_features[i] = F.interpolate(image_features[i], size=(32, 32), mode='bilinear')
            fea_cat = torch.cat(image_features, dim=1)
            loss = train_step(model, fea_cat, noise_scheduler, lr_scheduler, loss_fn, optimizer, args.train_steps,
                              args.noise_kind)

            running_loss_train += loss
            progress_bar.update(1)

        running_loss_test = 0
        with torch.no_grad():
            scores = Counter()
            for _btc_num, (_batch, _labels, gts) in enumerate(test_loader):
                loss = validate_step(model, _batch, noise_scheduler, args.train_steps,
                                     loss_fn) if args.calc_val_loss else 0

                running_loss_test += loss

                writer.add_scalars(main_tag='scores', tag_scalar_dict=dict(scores), global_step=epoch)
                progress_bar.update(1)

            # if epoch % 100 == 0:
            #     # runs it only for the last batch
            #     inference_ddim.run_inference_step(diffmap_blur, scores, gts, _btc_num, _batch, model,
            #                                       args.noise_kind, inf_noise_scheduler, _labels, writer, args.eta,
            #                                       15, 150, args.crop, args.plt_imgs,
            #                                       os.path.join(args.img_dir, run_id))

            for key in scores:
                scores[key] /= len(test_loader)

            progress_bar.set_postfix_str(
                f"Train Loss: {running_loss_train / len(train_loader)}, Test Loss: {running_loss_test / len(test_loader)}, {dict(scores)}")
            progress_bar.close()

            if epoch % args.save_n_epochs == 0 and epoch > 0:
                torch.save(model.state_dict(), f"{args.checkpoint_dir}/{args.run_name}_{timestamp}/epoch_{epoch}.pt")

        writer.add_scalar('Loss/train', running_loss_train, epoch)
        writer.add_scalar('Loss/test', running_loss_test, epoch)
        wandb.log({"loss_train": running_loss_train})
        wandb.log({"loss_test": running_loss_test})

    writer.add_hparams({'category': args.mvtec_item, 'res': args.resolution, 'eta': args.eta,
                        'recon_weight': args.reconstruction_weight}, {'MSE': running_loss_test},
                       run_name='hp')

    writer.flush()
    writer.close()

    torch.save(model.state_dict(), f"{args.checkpoint_dir}/{args.run_name}_{timestamp}/epoch_{args.epochs}.pt")


if __name__ == '__main__':
    args: TrainArgs = parse_args()
    main(args)
