import os

import numpy as np
import torch
from torch.autograd import Variable

from run_initialization import parse_input_options, get_generator_from_checkpoint, get_floorplan_dataset_loader_eval
from utils import mask_to_bb, bb_to_im_fid

# Create folder
os.makedirs('./FID/real', exist_ok=True)
os.makedirs('./FID/fake', exist_ok=True)

opt = parse_input_options()
generator = get_generator_from_checkpoint()
fp_loader = get_floorplan_dataset_loader_eval(opt)

# Optimizers
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ------------
#  Vectorize
# ------------
globalIndexReal = 0
globalIndexFake = 0
final_images = []
for i, batch in enumerate(fp_loader):
    print(i)
    if i >= 100:
        break

    # Unpack batch
    mks, nds, eds, nd_to_sample, ed_to_sample = batch

    # Configure input
    real_mks = Variable(mks.type(Tensor))
    given_nds = Variable(nds.type(Tensor))
    given_eds = eds

    for k in range(opt.num_variations):
        # Plot images
        z = Variable(Tensor(np.random.normal(0, 1, (real_mks.shape[0], opt.latent_dim))))
        with torch.no_grad():
            gen_mks = generator(z, given_nds, given_eds)
            gen_bbs = np.array([np.array(mask_to_bb(mk)) for mk in gen_mks.detach().cpu()])
            real_bbs = np.array([np.array(mask_to_bb(mk)) for mk in real_mks.detach().cpu()])
            real_nodes = np.where(given_nds.detach().cpu() == 1)[-1]

        if k == 0:
            real_bbs = real_bbs[np.newaxis, :, :] / 32.0
            real_im = bb_to_im_fid(real_bbs, real_nodes)
            real_im.save('{}/{}.jpg'.format('./FID/real', globalIndexReal))
            globalIndexReal += 1

        # draw vector
        gen_bbs = gen_bbs[np.newaxis, :, :] / 32.0
        fake_im = bb_to_im_fid(gen_bbs, real_nodes)
        fake_im.save('{}/{}.jpg'.format('./FID/fake', globalIndexFake))
        globalIndexFake += 1
