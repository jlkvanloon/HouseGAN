import argparse
import os
import numpy as np

import torchvision.transforms as transforms

from floorplan_dataset_maps import FloorplanGraphDataset, floorplan_collate_fn
from torch.autograd import Variable


import torch
from PIL import Image, ImageDraw

from models.generator import Generator
from reconstruct import reconstructFloorplan
import svgwrite
from utils import bb_to_vec, bb_to_seg, mask_to_bb, ID_COLOR, bb_to_im_fid
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import common_functions

parser = argparse.ArgumentParser()
parser.add_argument("--n_cpu", type=int, default=16, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--num_variations", type=int, default=1, help="number of variations")
parser.add_argument("--exp_folder", type=str, default='exp', help="destination folder")

opt = parser.parse_args()
print(opt)

numb_iters = 200000
exp_name = 'exp_with_graph_global_new'
target_set = 'E'
phase='eval'
checkpoint = './checkpoints/{}_{}_{}.pth'.format(exp_name, target_set, numb_iters)

# Create folder
os.makedirs(opt.exp_folder, exist_ok=True)

# Initialize generator and discriminator
generator = Generator()
generator.load_state_dict(torch.load(checkpoint))

# Initialize variables
cuda = True if torch.cuda.is_available() else False
if cuda:
    generator.cuda()
rooms_path = '/local-scratch4/nnauata/autodesk/FloorplanDataset/'

# Initialize dataset iterator
fp_dataset_test = FloorplanGraphDataset(rooms_path, transforms.Normalize(mean=[0.5], std=[0.5]), target_set=target_set, split=phase)
fp_loader = torch.utils.data.DataLoader(fp_dataset_test, 
                                        batch_size=opt.batch_size, 
                                        shuffle=False, collate_fn=floorplan_collate_fn)
# Optimizers
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ------------
#  Vectorize
# ------------
globalIndex = 0
final_images = []
target_graph = list(range(100))
for i, batch in enumerate(fp_loader):
    print(i)
    if i not in target_graph:
        continue
        
    # Unpack batch
    mks, nds, eds, nd_to_sample, ed_to_sample = batch
    
    # Configure input
    real_mks = Variable(mks.type(Tensor))
    given_nds = Variable(nds.type(Tensor))
    given_eds = eds
    for k in range(opt.num_variations):
        print('var num {}'.format(k))
        # plot images
        z = Variable(Tensor(np.random.normal(0, 1, (real_mks.shape[0], opt.latent_dim))))
        with torch.no_grad():
            gen_mks = generator(z, given_nds, given_eds)
            gen_bbs = np.array([np.array(mask_to_bb(mk)) for mk in gen_mks.detach().cpu()])
            real_bbs = np.array([np.array(mask_to_bb(mk)) for mk in real_mks.detach().cpu()])
            real_nodes = np.where(given_nds.detach().cpu()==1)[-1]
            print(real_nodes)
        gen_bbs = gen_bbs[np.newaxis, :, :]/32.0
        junctions = np.array(bb_to_vec(gen_bbs))[0, :, :]
        regions = np.array(bb_to_seg(gen_bbs))[0, :, :, :].transpose((1, 2, 0))
        graph = [real_nodes, None]
        
        if k == 0:
            graph_arr = common_functions.draw_graph([real_nodes, eds.detach().cpu().numpy()])
            final_images.append(graph_arr)
            
            # place real 
            real_bbs = real_bbs[np.newaxis, :, :]/32.0
            real_im = bb_to_im_fid(real_bbs, real_nodes)
            final_images.append(real_im)
            
            
        # reconstruct        
        fake_im = bb_to_im_fid(gen_bbs, real_nodes)
        final_images.append(fake_im)
    
  
row = 0
for k, im in enumerate(final_images):
    path = './out/var_{}/'.format(row)
    os.makedirs(path, exist_ok=True)
    im.save('{}/{}.jpg'.format(path, k))
    if (k+1) % 12 == 0:
        row+=1
