from collections import defaultdict

import numpy as np
import torch

from floorplan_dataset_maps import is_adjacent
from run_initialization_utils import parse_input_options, get_floorplan_dataset_loader_eval


def return_eq(node1, node2):
    return node1['label'] == node2['label']


def compute_dist(bb1, bb2):
    x0, y0, x1, y1 = bb1
    x2, y2, x3, y3 = bb2

    h1, h2 = x1 - x0, x3 - x2
    w1, w2 = y1 - y0, y3 - y2

    xc1, xc2 = (x0 + x1) / 2.0, (x2 + x3) / 2.0
    yc1, yc2 = (y0 + y1) / 2.0, (y2 + y3) / 2.0

    delta_x = abs(xc2 - xc1) - (h1 + h2) / 2.0
    delta_y = abs(yc2 - yc1) - (w1 + w2) / 2.0

    return delta_x, delta_y


def retrieve_connections(nodes, room_bb):
    edges = []
    nodes = [x for x in nodes if x >= 0]
    room_bb = room_bb.reshape((-1, 4))
    for k, bb1 in enumerate(room_bb):
        for l, bb2 in enumerate(room_bb):
            if k > l:
                if is_adjacent(bb1, bb2):
                    edges.append((k, l))
    return nodes, edges


def draw_floorplan(dwg, junctions, juncs_on, lines_on):
    # draw edges
    for k, l in lines_on:
        x1, y1 = np.array(junctions[k]) / 2.0
        x2, y2 = np.array(junctions[l]) / 2.0
        # fill='rgb({},{},{})'.format(*(np.random.rand(3)*255).astype('int'))
        dwg.add(dwg.line((float(x1), float(y1)), (float(x2), float(y2)), stroke='black', stroke_width=4, opacity=0.5))

    # draw corners
    for j in juncs_on:
        x, y = np.array(junctions[j]) / 2.0
        dwg.add(dwg.circle(center=(x, y), r=2, stroke='red', fill='white', stroke_width=1, opacity=0.75))
    return


opt = parse_input_options()
fp_loader = get_floorplan_dataset_loader_eval(opt)

# Generate samples
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

graphs = []
for i, batch in enumerate(fp_loader):
    # Unpack batch
    mks, nds, eds, nd_to_sample, ed_to_sample = batch
    real_nodes = np.where(nds.detach().cpu() == 1)[-1]
    graphs.append(len(real_nodes))

samples_per_len = defaultdict(int)
for g_len in graphs:
    samples_per_len[g_len] += 1

print(samples_per_len)
