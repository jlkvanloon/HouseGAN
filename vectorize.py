import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import svgwrite
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from floorplan_dataset_maps import FloorplanGraphDataset, floorplan_collate_fn
from models.generator import Generator
from reconstruct import reconstructFloorplan
from utils.run_initialization_utils import parse_input_options
from utils.utils import bb_to_img, bb_to_vec, bb_to_seg, mask_to_bb, ID_COLOR, pad_im

opt = parse_input_options()
IM_SIZE = 256
print(opt)

numb_iters = 200000
exp_name = 'exp_with_graph_global_new'
target_set = 'E'
phase = 'eval'
checkpoint = './checkpoints/{}_{}_{}.pth'.format(exp_name, target_set, numb_iters)


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
    #     pos = nx.spring_layout(G_true, scale=2)
    pos = nx.nx_agraph.graphviz_layout(G_true, prog='dot')
    nx.draw(G_true, pos, node_size=1000, node_color=colors_H, font_size=0, font_weight='bold')
    plt.tight_layout()
    plt.savefig('./dump/_true_graph.jpg', format="jpg")
    rgb_im = Image.open('./dump/_true_graph.jpg')
    rgb_arr = np.array(pad_im(rgb_im)) / 255.0
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


# Create folder
os.makedirs(opt.exp_folder, exist_ok=True)

# Initialize generator and discriminator
generator = Generator()
generator.load_state_dict(torch.load(checkpoint))

# Initialize variables
cuda = True if torch.cuda.is_available() else False
if cuda:
    generator.cuda()
rooms_path = '/local-scratch2/nnauata/autodesk/FloorplanDataset/'

# Initialize dataset iterator
fp_dataset_test = FloorplanGraphDataset(rooms_path, transforms.Normalize(mean=[0.5], std=[0.5]), target_set=target_set,
                                        split=phase)
fp_loader = torch.utils.data.DataLoader(fp_dataset_test,
                                        batch_size=opt.batch_size,
                                        shuffle=True, collate_fn=floorplan_collate_fn)
# Optimizers
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ------------
#  Vectorize
# ------------
globalIndex = 0
final_images = []
for i, batch in enumerate(fp_loader):
    print(i)
    if i > 16:
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
            real_nodes = np.where(given_nds.detach().cpu() == 1)[-1]
        gen_bbs = gen_bbs[np.newaxis, :, :] / 32.0
        junctions = np.array(bb_to_vec(gen_bbs))[0, :, :]
        regions = np.array(bb_to_seg(gen_bbs))[0, :, :, :].transpose((1, 2, 0))
        fake_imgs_tensor = bb_to_img(gen_bbs, [given_nds, given_eds], nd_to_sample, ed_to_sample, im_size=IM_SIZE)
        graph = [real_nodes, None]

        if k == 0:
            graph_arr = draw_graph([real_nodes, eds.detach().cpu().numpy()])
            print(graph_arr.shape)
            final_images.append(torch.tensor(graph_arr))

            # place real 
            real_bbs = real_bbs[np.newaxis, :, :] / 32.0
            real_junctions = np.array(bb_to_vec(real_bbs))[0, :, :]
            real_regions = np.array(bb_to_seg(real_bbs))[0, :, :, :].transpose((1, 2, 0))
            real_imgs_tensor = bb_to_img(real_bbs, [given_nds, given_eds], nd_to_sample, ed_to_sample, im_size=IM_SIZE)
            real_junctions, real_juncs_on, real_lines_on = reconstructFloorplan(real_regions, graph, globalIndex)

            # draw vector
            dwg = svgwrite.Drawing('./svg/floorplan_vec_{}.svg'.format(globalIndex), (256, 256))
            dwg.add(svgwrite.image.Image(os.path.abspath('./rooms/{}_rooms_updated.png'.format(globalIndex)),
                                         size=(256, 256)))
            draw_floorplan(dwg, real_junctions, real_juncs_on, real_lines_on)
            dwg.save()

            print('running inkscape ...')
            os.system('inkscape ./svg/floorplan_vec_{}.svg --export-png=_temp.png -w {}'.format(globalIndex, IM_SIZE))
            png_im = Image.open("_temp.png")

            rgb_img = Image.new('RGB', (256, 256), 'white')
            rgb_img.paste(png_im, (0, 0), mask=png_im)
            rgb_arr = np.array(rgb_img)
            final_images.append(torch.tensor(rgb_arr / 255.0))

        # reconstruct
        junctions, juncs_on, lines_on = reconstructFloorplan(regions, graph, globalIndex)

        # draw vector
        dwg = svgwrite.Drawing('./svg/floorplan_vec_{}.svg'.format(globalIndex), (256, 256))
        dwg.add(
            svgwrite.image.Image(os.path.abspath('./rooms/{}_rooms_updated.png'.format(globalIndex)), size=(256, 256)))
        draw_floorplan(dwg, junctions, juncs_on, lines_on)
        dwg.save()

        print('running inkscape ...')
        os.system('inkscape ./svg/floorplan_vec_{}.svg --export-png=_temp.png -w {}'.format(globalIndex, IM_SIZE))
        png_im = Image.open("_temp.png")

        rgb_img = Image.new('RGB', (256, 256), 'white')
        rgb_img.paste(png_im, (0, 0), mask=png_im)

        rgb_arr = np.array(rgb_img)
        final_images.append(torch.tensor(rgb_arr / 255.0))
        globalIndex += 1

final_images = torch.stack(final_images).transpose(1, 3)
print(final_images.shape)
save_image(final_images, "./output/rendered_all_images.png", nrow=opt.num_variations + 2)
