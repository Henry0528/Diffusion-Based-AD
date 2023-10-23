# Load model and dataset and run evaluation metrics on those.
from collections import Counter

import torch

from utils.metrics import scores_batch
from utils.visualize import generate_samples

from utils.anomalies import diff_map_to_anomaly_map


def evaluate(model: torch.nn.Module, dataloader, noise_scheduler, noise_kind, eta, num_inference_steps,
             start_at_timestep, patch_imgs):
    model.eval()

    with torch.no_grad():
        # validate and generate images
        eval_scores = Counter()

        for i, (imgs, states, gts) in enumerate(dataloader):
            originals, reconstructions, diffmaps, history = generate_samples(model, noise_scheduler,
                                                                             imgs, eta, num_inference_steps,
                                                                             start_at_timestep,
                                                                             patch_imgs,
                                                                             noise_kind)

            anomaly_maps = diff_map_to_anomaly_map(diffmaps, .3)
            eval_scores.update(scores_batch(gts, anomaly_maps))

        return eval_scores
