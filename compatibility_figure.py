import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable

from utils.run_initialization_utils import get_generator_from_checkpoint, parse_input_options, \
    get_floorplan_dataset_loader_eval
from utils.utils import mask_to_bb, ID_COLOR, bb_to_im_fid, pad_im


def make_sequence(given_nds, given_eds, noise):
    n_nodes = given_nds.shape[0]
    seq = []
    for k in range(n_nodes):
        curr_nds = given_nds[:k + 1]
        curr_noise = noise[:k + 1]
        curr_eds = []
        for i in range(k + 1):
            for j in range(k + 1):
                if j > i:
                    for e in given_eds:
                        if (e[0] == i and e[2] == j) or (e[2] == i and e[0] == j):
                            curr_eds.append([i, e[1], j])
        curr_eds = torch.tensor(curr_eds)
        seq.append([curr_nds, curr_noise, curr_eds])
    return seq

def draw_graph(g_true):
    # build true graph 
    G_true = nx.Graph()
    colors_H = []
    for k, label in enumerate(g_true[0]):
        _type = label + 1
        if _type >= 0:
            G_true.add_nodes_from([(k, {'label': _type})])
            colors_H.append(ID_COLOR[_type])
    for k, m, l in g_true[1]:
        if m > 0:
            G_true.add_edges_from([(k, l)], color='b', weight=4)
    plt.figure()
    pos = nx.nx_agraph.graphviz_layout(G_true, prog='dot')

    edges = G_true.edges()
    colors = ['black' for u, v in edges]
    weights = [4 for u, v in edges]

    nx.draw(G_true, pos, node_size=1000, node_color=colors_H, font_size=0, font_weight='bold', edges=edges,
            edge_color=colors, width=weights)
    plt.tight_layout()
    plt.savefig('./dump/_true_graph.jpg', format="jpg")
    rgb_im = Image.open('./dump/_true_graph.jpg')
    rgb_arr = pad_im(rgb_im)
    return rgb_arr


def draw_floorplan(dwg, junctions, juncs_on, lines_on):
    # draw edges
    for k, l in lines_on:
        x1, y1 = np.array(junctions[k])
        x2, y2 = np.array(junctions[l])
        # fill='rgb({},{},{})'.format(*(np.random.rand(3)*255).astype('int'))
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
target_graph = [47]
for i, batch in enumerate(fp_loader):
    if i not in target_graph:
        continue

    # Unpack batch
    mks, nds, eds, nd_to_sample, ed_to_sample = batch

    # Configure input
    real_mks = Variable(mks.type(Tensor))
    given_nds = Variable(nds.type(Tensor))
    given_eds = eds
    noise = Variable(Tensor(np.random.normal(0, 1, (real_mks.shape[0], opt.latent_dim))))
    samples = make_sequence(given_nds, given_eds, noise)

    for k, el in enumerate(samples):
        print('var num {}'.format(k))
        given_nds, z, given_eds = el
        # plot images
        with torch.no_grad():
            gen_mks = generator(z, given_nds, given_eds)
            gen_bbs = np.array([np.array(mask_to_bb(mk)) for mk in gen_mks.detach().cpu()])
            real_nodes = np.where(given_nds.detach().cpu() == 1)[-1]
            print(real_nodes)
        gen_bbs = gen_bbs[np.newaxis, :, :] / 32.0
        graph = [real_nodes, None]
        graph_arr = draw_graph([real_nodes, given_eds.detach().cpu().numpy()])
        final_images.append(graph_arr)

        # reconstruct        
        fake_im = bb_to_im_fid(gen_bbs, real_nodes)
        final_images.append(fake_im)

row = 0
for k, im in enumerate(final_images):
    path = './figure_seq/var_{}/'.format(row)
    os.makedirs(path, exist_ok=True)
    im.save('{}/{}.jpg'.format(path, k))
    if (k + 1) % 20 == 0:
        row += 1
# final_images = torch.stack(final_images).transpose(1, 3)
# save_image(final_images, "./output/rendered_{}.png".format(target_set), nrow=opt.num_variations+1)
