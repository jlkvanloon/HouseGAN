import numpy as np

#this file contains functions common to multiple files in the repository


#Comment Author: Varnika
#the draw_floorplan function is common among the following files:
    # vectorize.py
    # variation_bbs_with_target_graph_segments_suppl.py - I removed this function in this file because it wasn't being called anywhere
    # variation_bbs_with_target_graph_segments.py - I removed this function in this file because it wasn't being called anywhere
    # variation_bbs_with_target graph.py - I removed this function in this file because it wasn't being called anywhere
    # variation_bbs.py - I removed this function in this file because it wasn't being called anywhere

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

#Comment Author: Varnika
# The following files have the same draw_graph function:
    # variation_bbs_with_target_graph_segments.py
    # variation_bbs_with_target_graph_segments_suppl.py
    # variation_bbs_with_target_graph.py

#In variation_bbs.py - draw_graph is quite different from the above function so the function is kept as is.
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
    pos = nx.nx_agraph.graphviz_layout(G_true, prog='neato')

    edges = G_true.edges()
    colors = ['black' for u,v in edges]
    weights = [4 for u,v in edges]

    nx.draw(G_true, pos, node_size=1000, node_color=colors_H, font_size=0, font_weight='bold', edges=edges, edge_color=colors, width=weights)
    plt.tight_layout()
    plt.savefig('./dump/_true_graph.jpg', format="jpg")
    rgb_im = Image.open('./dump/_true_graph.jpg')
    rgb_arr = pad_im(rgb_im)
    return rgb_arr


