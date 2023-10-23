# imports
import argparse
import json
import os
from dataclasses import dataclass

import torch
from diffusers import UNet2DModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import utils.anomalies
from loader.loader import MVTecDataset
from schedulers.scheduling_ddim import DDIMScheduler
from utils.files import save_args
from utils.metrics import scores, scores_batch
from utils.visualize import generate_samples, plot_single_channel_imgs, plot_rgb_imgs, gray_to_rgb, \
    split_into_patches, add_overlay, add_batch_overlay
from collections import Counter
from efficientnet_pytorch import EfficientNet
from torch.nn import functional as F
@dataclass
class InferenceArgs:
    num_inference_steps: int
    start_at_timestep: int
    reconstruction_weight: float
    mvtec_item: str
    mvtec_item_states: list
    checkpoint_dir: str
    checkpoint_name: str
    log_dir: str
    train_steps: int
    beta_schedule: str
    eta: float
    device: str
    dataset_path: str
    shuffle: bool
    img_dir: str
    plt_imgs: bool
    patch_imgs: bool
    run_id: str
    batch_size: int


def parse_args() -> InferenceArgs:
    parser = argparse.ArgumentParser(description='Add config for the training')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='directory path to store the checkpoints')
    parser.add_argument('--log_dir', type=str, default="logs",
                        help='directory path to store logs')
    parser.add_argument('--img_dir', type=str, default="generated_imgs",
                        help='directory path to store generated imgs')
    parser.add_argument('--checkpoint_name', type=str, required=True,
                        help='name of the run and corresponding checkpoints/logs that are created')
    parser.add_argument('--mvtec_item', type=str, required=True,
                        choices=["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
                                 "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"],
                        help='name of the item within the MVTec Dataset to train on')
    parser.add_argument('--mvtec_item_states', type=str, nargs="+", default=["all"],
                        help="States of the mvtec items that should be used. Available options depend on the selected item. Set to 'all' to include all states")
    parser.add_argument('--num_inference_steps', type=int, default=50,
                        help='At which timestep/how many timesteps should be regenerated')
    parser.add_argument('--start_at_timestep', type=int, default=300,
                        help='At which timestep/how many timesteps should be regenerated')
    parser.add_argument('--train_steps', type=int, default=1000,
                        help='number of steps for the full diffusion process')
    parser.add_argument('--beta_schedule', type=str, default="linear",
                        help='Type of schedule for the beta/variance values')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='directory path to the (mvtec) dataset')
    parser.add_argument('--run_id', type=str, default='inference',
                        help='id of the run, required for the logging')
    parser.add_argument('--device', type=str, default="cuda",
                        help='device to train on')
    parser.add_argument('--recon_weight', type=float, default=1, dest="reconstruction_weight",
                        help='Influence of the original sample during generation')
    parser.add_argument('--eta', type=float, default=0,
                        help='Stochasticity parameter of DDIM, with eta=1 being DDPM and eta=0 meaning no randomness')
    parser.add_argument('--shuffle', action='store_true',
                        help='Shuffle the items in the dataset')
    parser.add_argument('--plt_imgs', action='store_true',
                        help='Plot the images with matplot lib. I.e. call plt.show()')
    parser.add_argument('--patch_imgs', action='store_true',
                        help='If the image size is larger than the models input, split input into multiple patches and stitch it together afterwards.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Number of images to process per batch')

    return InferenceArgs(**vars(parser.parse_args()))

def main(args: InferenceArgs, writer: SummaryWriter):
    # train loop
    print("**** starting inference *****")
    config_file = open(f"{args.checkpoint_dir}/model_config.json", "r")
    model_config = json.loads(config_file.read())
    train_arg_file = open(f"{args.checkpoint_dir}/train_arg_config.json", "r")
    train_arg_config: dict = json.loads(train_arg_file.read())
    save_args(args, args.img_dir, "inference_args")

    augmentations = transforms.Compose(
        [
            transforms.Resize(model_config["sample_size"],
                              interpolation=transforms.InterpolationMode.BILINEAR) if not args.patch_imgs else transforms.Lambda(
                lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    def transform_images(imgs):
        return [augmentations(image.convert("RGB")) for image in imgs]

    # data loader
    test_data = MVTecDataset(args.dataset_path, False, args.mvtec_item, args.mvtec_item_states,
                             transform_images)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=args.shuffle)

    # set model, optimizer, scheduler
    model = UNet2DModel(
        **model_config
    )

    model.load_state_dict(torch.load(f"{args.checkpoint_dir}/{args.checkpoint_name}"))
    model.eval()
    model.to(args.device)
    feature_extractor = EfficientNet.from_pretrained('efficientnet-b4')
    feature_extractor.to(args.device)
    feature_extractor.eval()
    diffmap_blur = transforms.GaussianBlur(2 * int(4 * 4 + 0.5) +1, 4)

    with torch.no_grad():
        # validate and generate images
        noise_scheduler_inference = DDIMScheduler(args.train_steps, args.start_at_timestep,
                                                  beta_schedule=args.beta_schedule, timestep_spacing="leading",
                                                  reconstruction_weight=args.reconstruction_weight)
        noise_kind = train_arg_config.get("noise_kind", "gaussian")
        eval_scores = Counter()

        for i, (imgs, states, gts) in enumerate(test_loader):
            imgs = imgs.to(args.device)
            image_features = list(feature_extractor.extract_endpoints(imgs).values())[:-2]
            for i in range(4):
                image_features[i] = F.interpolate(image_features[i], size=(32, 32), mode='bilinear')
            fea_cat = torch.cat(image_features, dim=1)
            gts = gts.to(args.device)
            run_inference_step(diffmap_blur, eval_scores, gts, i, fea_cat, model, noise_kind,
                               noise_scheduler_inference, states, writer, args.eta, args.num_inference_steps,
                               args.start_at_timestep, args.patch_imgs, args.plt_imgs, args.img_dir)

        for key in eval_scores:
            eval_scores[key] /= len(test_loader)
        writer.add_hparams({'category': args.mvtec_item, 'eta': args.eta,
                            'recon_weight': args.reconstruction_weight, 'states': ','.join(args.mvtec_item_states),
                            't': args.start_at_timestep, 'num_steps': args.num_inference_steps,
                            'input_size': model_config["sample_size"], 'patching': args.patch_imgs}, dict(eval_scores),
                           run_name=f'hp')
        print(eval_scores)


def run_inference_step(diffmap_blur, eval_scores, gts, btc_idx, imgs, model, noise_kind, noise_scheduler_inference,
                       states, writer, eta, num_inference_steps, start_at_timestep, patch_imgs, plt_imgs, img_dir):
    originals, reconstructions, diffmaps, history = generate_samples(model, noise_scheduler_inference,
                                                                     imgs,
                                                                     eta, num_inference_steps,
                                                                     start_at_timestep,
                                                                     patch_imgs,
                                                                     noise_kind)
    anomaly_maps = utils.anomalies.diff_map_to_anomaly_map(diffmaps, .3, diffmap_blur)
    # overlays = add_batch_overlay(originals, anomaly_maps)
    eval_scores.update(scores_batch(gts, anomaly_maps))
    for idx in range(len(gts)):
        if not os.path.exists(f"{img_dir}"):
            os.makedirs(f"{img_dir}")

        # plot_single_channel_imgs([gts[idx], diffmaps[idx], anomaly_maps[idx]],
        #                          ["ground truth", "diff-map", "anomaly-map"],
        #                          cmaps=['gray', 'viridis', 'gray'],
        #                          save_to=f"{img_dir}/{btc_idx}_{states[idx]}_heatmap.png", show_img=plt_imgs)
        # plot_rgb_imgs([originals[idx], reconstructions[idx], overlays[idx]], ["original", "reconstructed", "overlay"],
        #               save_to=f"{img_dir}/{btc_idx}_{states[idx]}.png", show_img=plt_imgs)

        # if writer is not None:
        #     for t, im in zip(history["timesteps"], history["images"]):
        #         writer.add_images(f"{btc_idx}_{states[0]}_process", im[idx].unsqueeze(0), t)
        #
        #     writer.add_images(f"{btc_idx}_{states[0]}_results (ori, rec, diff, pred, gt)", torch.stack(
        #         [originals[idx], reconstructions[idx], gray_to_rgb(diffmaps[idx])[0], gray_to_rgb(anomaly_maps[idx])[0],
        #          gray_to_rgb(gts[idx])[0]]))


if __name__ == '__main__':
    args: InferenceArgs = parse_args()
    writer = SummaryWriter(f'{args.log_dir}/{args.run_id}') if args.log_dir else None
    main(args, writer)
    writer.flush()
    writer.close()
