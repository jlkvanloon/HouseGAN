import argparse
import os
import numpy as np
import math
import sys
import random

import torchvision.transforms as transforms
from torchvision.utils import save_image

from floorplan_dataset_maps import FloorplanGraphDataset, floorplan_collate_fn
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
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
parser.add_argument("--num_variations", type=int, default=4, help="number of variations")
parser.add_argument("--exp_folder", type=str, default='exp', help="destination folder")

opt = parser.parse_args()
print(opt)

numb_iters = 500000
exp_name = 'exp_demo'
target_set = 'D'
phase='eval'
checkpoint = './data/{}_{}_{}.pth'.format(exp_name, target_set, numb_iters)
os.makedirs("./dump/", exist_ok=True)
os.makedirs("./output/", exist_ok=True)

def pad_im(cr_im, final_size=256, bkg_color='white'):    
    new_size = int(np.max([np.max(list(cr_im.size)), final_size]))
    padded_im = Image.new('RGB', (new_size, new_size), 'white')
    padded_im.paste(cr_im, ((new_size-cr_im.size[0])//2, (new_size-cr_im.size[1])//2))
    padded_im = padded_im.resize((final_size, final_size), Image.ANTIALIAS)
    return padded_im
        
import cv2
import webcolors
def draw_masks(masks, real_nodes):
    
    bg_img = Image.new("RGBA", (256, 256), (255, 255, 255, 0))  # Semitransparent background.
    for m, nd in zip(masks, real_nodes):
        
        # draw region
        reg = Image.new('RGBA', (32, 32), (0,0,0,0))
        dr_reg = ImageDraw.Draw(reg)
        m[m>0] = 255
        m[m<0] = 0
        m = m.detach().cpu().numpy()
        m = Image.fromarray(m)
        color = ID_COLOR[nd+1]
        r, g, b = webcolors.name_to_rgb(color)
        dr_reg.bitmap((0, 0), m.convert('L'), fill=(r, g, b, 32))
        reg = reg.resize((256, 256))
        
        bg_img.paste(Image.alpha_composite(bg_img, reg))

  
    for m, nd in zip(masks, real_nodes):
        cnt = Image.new('RGBA', (256, 256), (0,0,0,0))
        dr_cnt = ImageDraw.Draw(cnt)
        
        mask = np.zeros((256,256,3)).astype('uint8')
        m[m>0] = 255
        m[m<0] = 0
        m = m.detach().cpu().numpy()[:, :, np.newaxis].astype('uint8')
        m = cv2.resize(m, (256, 256), interpolation = cv2.INTER_AREA) 
        ret,thresh = cv2.threshold(m,127,255,0)
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:  
            contours = [c for c in contours]
        color = ID_COLOR[nd+1]
        r, g, b = webcolors.name_to_rgb(color)
        cv2.drawContours(mask, contours, -1, (255, 255, 255), 2)
        
        mask = Image.fromarray(mask)
        dr_cnt.bitmap((0, 0), mask.convert('L'), fill=(r, g, b, 256))
        
        bg_img.paste(Image.alpha_composite(bg_img, cnt))


    return bg_img

# Create folder
os.makedirs(opt.exp_folder, exist_ok=True)

# Initialize generator and discriminator
generator = Generator()
generator.load_state_dict(torch.load(checkpoint, map_location=torch.device('cpu')))

# Initialize variables
cuda = True if torch.cuda.is_available() else False
if cuda:
    generator.cuda()
rooms_path = './data/dataset_paper'

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
target_graph = list(range(500))
page_count = 0
n_rows = 0
for i, batch in enumerate(fp_loader):
    if i not in target_graph:
        continue
        
    # Unpack batch
    mks, nds, eds, nd_to_sample, ed_to_sample = batch
    
    # Configure input
    real_mks = Variable(mks.type(Tensor))
    given_nds = Variable(nds.type(Tensor))
    given_eds = eds
    for k in range(opt.num_variations):
#         print('var num {}'.format(k))
        # plot images
        z = Variable(Tensor(np.random.normal(0, 1, (real_mks.shape[0], opt.latent_dim))))
        with torch.no_grad():
            gen_mks = generator(z, given_nds, given_eds)
            gen_bbs = np.array([np.array(mask_to_bb(mk)) for mk in gen_mks.detach().cpu()])
            real_bbs = np.array([np.array(mask_to_bb(mk)) for mk in real_mks.detach().cpu()])
            real_nodes = np.where(given_nds.detach().cpu()==1)[-1]
        gen_bbs = gen_bbs[np.newaxis, :, :]/32.0
        junctions = np.array(bb_to_vec(gen_bbs))[0, :, :]
        regions = np.array(bb_to_seg(gen_bbs))[0, :, :, :].transpose((1, 2, 0))
        graph = [real_nodes, None]
        
        if k == 0:
            #original file has 'convert('RGBA')' in the draw_graph function
            graph_arr = common_functions.draw_graph([real_nodes, eds.detach().cpu().numpy()]).convert('RGBA')
            final_images.append(graph_arr)
            
            
        # reconstruct
        fake_im_seg = draw_masks(gen_mks, real_nodes)
        final_images.append(fake_im_seg)
        fake_im_bb = bb_to_im_fid(gen_bbs, real_nodes, im_size=256).convert('RGBA')
        final_images.append(fake_im_bb)
    n_rows += 1
    if (n_rows+1)%12 == 0: 
        final_images_new = []
        for im in final_images:
            final_images_new.append(torch.tensor(np.array(im).transpose((2, 0, 1)))/255.0)

        # print('final: ', final_images_new[0].shape)
        final_images = final_images_new
        final_images = torch.stack(final_images)
        save_image(final_images, "./output/results_page_{}_{}.png".format(target_set, page_count), nrow=2*opt.num_variations+1, padding=2, range=(0, 1), pad_value=0.5, normalize=False)
        page_count += 1
        n_rows = 0
        final_images = []
