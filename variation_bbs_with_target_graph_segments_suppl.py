import os

import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import webcolors
from PIL import Image, ImageDraw
from torch.autograd import Variable
from torchvision.utils import save_image

from utils.run_initialization_utils import parse_input_options, get_generator_from_checkpoint, \
    get_floorplan_dataset_loader_eval
from utils.utils import bb_to_vec, bb_to_seg, mask_to_bb, ID_COLOR, bb_to_im_fid, pad_im


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
    pos = nx.nx_agraph.graphviz_layout(G_true, prog='neato')

    edges = G_true.edges()
    colors = ['black' for u, v in edges]
    weights = [4 for u, v in edges]

    nx.draw(G_true, pos, node_size=1000, node_color=colors_H, font_size=0, font_weight='bold', edges=edges,
            edge_color=colors, width=weights)
    plt.tight_layout()
    plt.savefig('./dump/_true_graph.jpg', format="jpg")
    rgb_im = Image.open('./dump/_true_graph.jpg')
    rgb_arr = pad_im(rgb_im).convert('RGBA')
    return rgb_arr


def draw_masks(masks, real_nodes):
    bg_img = Image.new("RGBA", (256, 256), (255, 255, 255, 0))  # Semitransparent background.
    for m, nd in zip(masks, real_nodes):
        # draw region
        reg = Image.new('RGBA', (32, 32), (0, 0, 0, 0))
        dr_reg = ImageDraw.Draw(reg)
        m[m > 0] = 255
        m[m < 0] = 0
        m = m.detach().cpu().numpy()
        m = Image.fromarray(m)
        color = ID_COLOR[nd + 1]
        r, g, b = webcolors.name_to_rgb(color)
        dr_reg.bitmap((0, 0), m.convert('L'), fill=(r, g, b, 32))
        reg = reg.resize((256, 256))

        bg_img.paste(Image.alpha_composite(bg_img, reg))

    for m, nd in zip(masks, real_nodes):
        cnt = Image.new('RGBA', (256, 256), (0, 0, 0, 0))
        dr_cnt = ImageDraw.Draw(cnt)

        mask = np.zeros((256, 256, 3)).astype('uint8')
        m[m > 0] = 255
        m[m < 0] = 0
        m = m.detach().cpu().numpy()[:, :, np.newaxis].astype('uint8')
        m = cv2.resize(m, (256, 256), interpolation=cv2.INTER_AREA)
        ret, thresh = cv2.threshold(m, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            contours = [c for c in contours]
        color = ID_COLOR[nd + 1]
        r, g, b = webcolors.name_to_rgb(color)
        cv2.drawContours(mask, contours, -1, (255, 255, 255), 2)

        mask = Image.fromarray(mask)
        dr_cnt.bitmap((0, 0), mask.convert('L'), fill=(r, g, b, 256))

        bg_img.paste(Image.alpha_composite(bg_img, cnt))

    #     im2 = np.zeros((256,256,3)).astype('uint8') + 255
    #     for m, nd in zip(masks, real_nodes):
    #         m[m>0] = 255
    #         m[m<0] = 0
    #         m = m.detach().cpu().numpy()[:, :, np.newaxis].astype('uint8')
    #         m = cv2.resize(m, (256, 256), interpolation = cv2.INTER_AREA)
    #         ret,thresh = cv2.threshold(m,127,255,0)
    #         contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #         if len(contours) > 0:
    #             contours = [c for c in contours]
    #         color = ID_COLOR[nd+1]
    #         r, g, b = webcolors.name_to_rgb(color)

    #         cv2.drawContours(im2, contours, -1, (r, g, b), 2)
    #     im2 = Image.fromarray(im2).convert('RGBA')

    #     im.paste(im2)
    #     out.save('./test.png')
    #     im.save('./test_reg.png')

    return bg_img


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
os.makedirs("./dump/", exist_ok=True)
os.makedirs("./output/", exist_ok=True)

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
target_graph = list(range(500))
page_count = 0
n_rows = 0
for i, batch in enumerate(fp_loader):
    print(i, "/", len(fp_loader))
    if i not in target_graph:
        continue

    # Unpack batch
    mks, nds, eds, nd_to_sample, ed_to_sample = batch

    # Configure input
    real_mks = Variable(mks.type(Tensor))
    given_nds = Variable(nds.type(Tensor))
    given_eds = eds
    for k in range(opt.num_variations):
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
        graph = [real_nodes, None]

        if k == 0:
            graph_arr = draw_graph([real_nodes, eds.detach().cpu().numpy()])
            final_images.append(graph_arr)

        # reconstruct
        fake_im_seg = draw_masks(gen_mks, real_nodes)
        final_images.append(fake_im_seg)
        fake_im_bb = bb_to_im_fid(gen_bbs, real_nodes, im_size=256).convert('RGBA')
        final_images.append(fake_im_bb)
    n_rows += 1
    if (n_rows + 1) % 12 == 0:
        final_images_new = []
        for im in final_images:
            final_images_new.append(torch.tensor(np.array(im).transpose((2, 0, 1))) / 255.0)

        # print('final: ', final_images_new[0].shape)
        final_images = final_images_new
        final_images = torch.stack(final_images)
        save_image(final_images, "./output/results_page_D_{}.png".format(page_count),
                   nrow=2 * opt.num_variations + 1, padding=2, range=(0, 1), pad_value=0.5, normalize=False)
        page_count += 1
        n_rows = 0
        final_images = []
