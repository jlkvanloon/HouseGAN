import os
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.utils import save_image

from run_initialization_utils import get_floorplan_dataset_loader_eval, get_generator_from_checkpoint, \
    parse_input_options
from utils import bb_to_vec, bb_to_seg, mask_to_bb, ID_COLOR, bb_to_im_fid


def pad_im(cr_im, final_size=299):
    new_size = int(np.max([np.max(list(cr_im.size)), final_size]))
    padded_im = Image.new('RGB', (new_size, new_size), 'white')
    padded_im.paste(cr_im, ((new_size-cr_im.size[0])//2, (new_size-cr_im.size[1])//2))
    padded_im = padded_im.resize((final_size, final_size), Image.ANTIALIAS)
    return padded_im

def draw_graph(g_true):
    # build true graph 
    G_true = nx.Graph()
    colors_H = []
    for k, label in enumerate(g_true[0]):
        _type = label+1 
        if _type >= 0:
            G_true.add_nodes_from([(k, {'label':_type})])
            colors_H.append(ID_COLOR[_type])
    for k, m, l in g_true[1]:
        if m > 0:
            G_true.add_edges_from([(k, l)], color='b',weight=4)    
    plt.figure()
#     pos = nx.spring_layout(G_true, scale=2)
    pos = nx.nx_agraph.graphviz_layout(G_true, prog='dot')
    nx.draw(G_true, pos, node_size=1000, node_color=colors_H, font_size=0, font_weight='bold')
    plt.tight_layout()
    plt.savefig('./dump/_true_graph.jpg', format="jpg")
    rgb_im = Image.open('./dump/_true_graph.jpg')
    rgb_arr = np.array(pad_im(rgb_im))/255.0
    return rgb_arr

def draw_floorplan(dwg, junctions, juncs_on, lines_on):
    # draw edges
    for k, l in lines_on:
        x1, y1 = np.array(junctions[k])
        x2, y2 = np.array(junctions[l])
        #fill='rgb({},{},{})'.format(*(np.random.rand(3)*255).astype('int'))
        dwg.add(dwg.line((float(x1), float(y1)), (float(x2), float(y2)), stroke='black', stroke_width=4, opacity=1.0))

    # draw corners
    for j in juncs_on:
        x, y = np.array(junctions[j])
        dwg.add(dwg.circle(center=(float(x), float(y)), r=3, stroke='red', fill='white', stroke_width=2, opacity=1.0))
    return 

opt = parse_input_options()
generator = get_generator_from_checkpoint()
fp_loader = get_floorplan_dataset_loader_eval(opt)
os.makedirs(opt.exp_folder, exist_ok=True)

# Optimizers
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ------------
#  Vectorize
# ------------
globalIndex = 0
final_images = []
for i, batch in enumerate(fp_loader):
    print(i)
    if i > 10:
        break
        
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
        gen_bbs = gen_bbs[np.newaxis, :, :]/32.0
        junctions = np.array(bb_to_vec(gen_bbs))[0, :, :]
        regions = np.array(bb_to_seg(gen_bbs))[0, :, :, :].transpose((1, 2, 0))
        graph = [real_nodes, None]
        
        if k == 0:
            graph_arr = draw_graph([real_nodes, eds.detach().cpu().numpy()])
            final_images.append(torch.tensor(graph_arr))

        # reconstruct
        fake_im = bb_to_im_fid(gen_bbs, real_nodes)
        rgb_arr = np.array(fake_im)
        final_images.append(torch.tensor(rgb_arr/255.0))
        
final_images = torch.stack(final_images).transpose(1, 3)
save_image(final_images, "./output/rendered_D.png", nrow=opt.num_variations+1)
