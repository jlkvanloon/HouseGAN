import argparse

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from floorplan_dataset_maps import FloorplanGraphDataset, floorplan_collate_fn
from models.generator import Generator


def parse_input_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_cpu", type=int, default=16, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--num_variations", type=int, default=10, help="number of variations")
    parser.add_argument("--exp_folder", type=str, default='exp', help="destination folder")

    opt = parser.parse_args()
    print(opt)

    return opt


def get_generator_from_checkpoint(checkpoint='./data/exp_demo_D_500000.pth'):
    generator = Generator()
    generator.load_state_dict(torch.load(checkpoint, map_location=torch.device('cpu')))

    cuda = True if torch.cuda.is_available() else False
    if cuda:
        generator.cuda()

    return generator


def get_floorplan_dataset_loader_eval(opt, rooms_path='./data/dataset_paper'):
    fp_dataset_test = FloorplanGraphDataset(
        rooms_path,
        transforms.Normalize(mean=[0.5], std=[0.5]),
        target_set='D',
        split='eval'
    )

    return torch.utils.data.DataLoader(
        fp_dataset_test,
        batch_size=opt.batch_size,
        shuffle=False,
        collate_fn=floorplan_collate_fn
    )
