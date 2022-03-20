import torch
import torch.nn as nn

from models.conv_block import conv_block


def pool_edges(edges, feats, positive):
    dtype, device = feats.dtype, feats.device
    indices = torch.where(edges[:, 1] > 0 if positive else edges[:, 1] < 0)
    num_vertices, num_edges = feats.size(0), edges.size(0)
    pooled_v = torch.zeros(num_vertices, feats.shape[-3], feats.shape[-1], feats.shape[-1], dtype=dtype, device=device)

    v_src = torch.cat([edges[indices[0], 0], edges[indices[0], 2]]).long()
    v_dst = torch.cat([edges[indices[0], 2], edges[indices[0], 0]]).long()
    vecs_src = feats[v_src.contiguous()]
    v_dst = v_dst.view(-1, 1, 1, 1).expand_as(vecs_src).to(device)
    return pooled_v.scatter_add(0, v_dst, vecs_src)


class CMP(nn.Module):
    def __init__(self, in_channels):
        super(CMP, self).__init__()
        self.in_channels = in_channels
        self.encoder = nn.Sequential(
            *conv_block(3 * in_channels, 2 * in_channels, 3, 1, 1, act="leaky"),
            *conv_block(2 * in_channels, 2 * in_channels, 3, 1, 1, act="leaky"),
            *conv_block(2 * in_channels, in_channels, 3, 1, 1, act="leaky"))

    def forward(self, feats, edges=None):
        # pool positive edges
        pooled_v_pos = pool_edges(edges, feats, positive=True)

        # pool negative edges
        pooled_v_neg = pool_edges(edges, feats, positive=False)

        # update nodes features
        enc_in = torch.cat([feats, pooled_v_pos, pooled_v_neg], 1)
        out = self.encoder(enc_in)
        return out

