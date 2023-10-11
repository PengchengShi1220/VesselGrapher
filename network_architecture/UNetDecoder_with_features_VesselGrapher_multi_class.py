import numpy as np
import shutil
import os
import torch
import random
import nibabel as nib
import networkx as nx
#import distmap
import cupy as cp
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
from cucim.core.operations.morphology import distance_transform_edt as distance_transform_edt_cupy
from itertools import combinations
# import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from torch import nn
from time import time
from typing import Union, List, Tuple
from scipy import ndimage
from sklearn.neighbors import NearestNeighbors
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder

from nnunetv2.training.nnUNetTrainer.variants.network_architecture.torch_nn import BasicConv, batched_index_select, act_layer
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.torch_edge import DenseDilatedKnnGraph
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.pos_embed import get_2d_relative_pos_embed, get_3d_relative_pos_embed
import torch.nn.functional as F
from timm.models.layers import DropPath
from einops import rearrange
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p
from scipy.ndimage import distance_transform_edt, morphology, generate_binary_structure
from skimage.morphology import cube
from scipy.spatial import cKDTree

class OptInit:
    def __init__(self, drop_path_rate=0., pool_op_kernel_sizes_len=4):
        # self.k = [4, 8, 16] + [32] * (pool_op_kernel_sizes_len - 3) 
        self.k = [8, 8, 8] + [8] * (pool_op_kernel_sizes_len - 3) 
        self.conv = 'mr'  
        self.act = 'leakyrelu'
        self.norm = 'instance'
        self.bias = True
        self.dropout = 0.0  # dropout rate
        self.use_dilation = True  # use dilated knn or not
        self.epsilon = 0.2  # stochastic epsilon for gcn
        self.use_stochastic = True 
        self.drop_path = drop_path_rate
        # number of basic blocks in the backbone
        self.blocks = [1] * (pool_op_kernel_sizes_len - 2) + [1, 1] 
        # number of reduce ratios in the backbone
        self.reduce_ratios = [4, 2, 1, 1] + [1] * (pool_op_kernel_sizes_len - 4) #[4, 2, 1, 1] + [1] * (pool_op_kernel_sizes_len - 4) 

class UNetDecoder_with_features(nn.Module):
    def __init__(self,
                 encoder: Union[PlainConvEncoder, ResidualEncoder],
                 patch_size: List[int],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision, nonlin_first: bool = False):
        """
        This class needs the skips of the encoder as input in its forward.

        the encoder goes all the way to the bottleneck, so that's where the decoder picks up. stages in the decoder
        are sorted by order of computation, so the first stage has the lowest resolution and takes the bottleneck
        features and the lowest skip as inputs
        the decoder has two (three) parts in each stage:
        1) conv transpose to upsample the feature maps of the stage below it (or the bottleneck in case of the first stage)
        2) n_conv_per_stage conv blocks to let the two inputs get to know each other and merge
        3) (optional if deep_supervision=True) a segmentation output Todo: enable upsample logits?
        :param encoder:
        :param num_classes:
        :param n_conv_per_stage:
        :param deep_supervision:
        """
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                          "resolution stages - 1 (n_stages in encoder - 1), " \
                                                          "here: %d" % n_stages_encoder

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)
        conv_op=encoder.conv_op
        self.conv_op = conv_op
        self.norm_op = encoder.norm_op
        self.norm_op_kwargs = encoder.norm_op_kwargs
        self.dropout_op = encoder.dropout_op

        img_shape_list = []
        n_size_list = []
        conv_layer_d_num = 0 #2
        pool_op_kernel_sizes = strides[1:]
        if conv_op == nn.Conv2d:
            h, w = patch_size[0], patch_size[1]
            img_shape_list.append((h, w))
            n_size_list.append(h * w)

            for i in range(len(pool_op_kernel_sizes)):
                h_k, w_k = pool_op_kernel_sizes[i]
                h //= h_k
                w //= w_k
                img_shape_list.append((h, w))
                n_size_list.append(h * w)

        elif conv_op == nn.Conv3d:
            h, w, d = patch_size[0], patch_size[1], patch_size[2]
            img_shape_list.append((h, w, d))
            n_size_list.append(h * w * d)

            for i in range(len(pool_op_kernel_sizes)):
                h_k, w_k, d_k = pool_op_kernel_sizes[i]
                h //= h_k
                w //= w_k
                d //= d_k
                img_shape_list.append((h, w, d))
                n_size_list.append(h * w * d)
        else:
            raise ValueError("unknown convolution dimensionality, conv op: %s" % str(conv_op))

        img_min_shape = img_shape_list[-1]

        opt = OptInit(pool_op_kernel_sizes_len=len(strides))
        self.opt = opt
        self.opt.img_min_shape = img_min_shape

        self.conv_layer_d_num = conv_layer_d_num
                
        self.opt.n_size_list = n_size_list

        # we start with the bottleneck and work out way up
        stages = []
        evig_stages = []
        transpconvs = []
        stages_center = []
        seg_layers_for_evig = []
        seg_layers = []
        
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            transpconvs.append(transpconv_op(
                input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                bias=encoder.conv_bias
            ))

            # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
            if s == (n_stages_encoder-1): #< (n_stages_encoder-conv_layer_d_num):
                stages.append(StackedConvBlocks(
                    n_conv_per_stage[s-1] - 1, encoder.conv_op, 2 * input_features_skip, input_features_skip,
                    # n_conv_per_stage[s-1], encoder.conv_op, 2 * input_features_skip, input_features_skip,
                    encoder.kernel_sizes[-(s + 1)], 1, encoder.conv_bias, encoder.norm_op, encoder.norm_op_kwargs,
                    encoder.dropout_op, encoder.dropout_op_kwargs, encoder.nonlin, encoder.nonlin_kwargs, nonlin_first
                ))
                evig_stages.append(Efficient_ViG_blocks(input_features_skip, img_shape_list[n_stages_encoder-(s + 1)], n_stages_encoder-conv_layer_d_num-(s + 1), conv_layer_d_num, opt=self.opt, conv_op=self.conv_op,
                    norm_op=self.norm_op, norm_op_kwargs=self.norm_op_kwargs, dropout_op=self.dropout_op))

            else:
                stages.append(StackedConvBlocks(
                n_conv_per_stage[s-1], encoder.conv_op, 2 * input_features_skip, input_features_skip,
                encoder.kernel_sizes[-(s + 1)], 1, encoder.conv_bias, encoder.norm_op, encoder.norm_op_kwargs,
                encoder.dropout_op, encoder.dropout_op_kwargs, encoder.nonlin, encoder.nonlin_kwargs, nonlin_first
            ))
                
            # we always build the deep supervision outputs so that we can always load parameters. If we don't do this
            # then a model trained with deep_supervision=True could not easily be loaded at inference time where
            # deep supervision is not needed. It's just a convenience thing
            seg_layers_for_evig.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.stages_center = nn.ModuleList(stages_center)
        self.evig_stages = nn.ModuleList(evig_stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)
        self.seg_layers_for_evig = nn.ModuleList(seg_layers_for_evig)

    def forward(self, skips, windows_seg_targets_dict, windows_graph_edges_target_dict, windows_graph_nodes_dict, current_epoch, val_flag, test_flag):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """
        lres_input = skips[-1]
        seg_outputs = []
        recurrent_num = 1

        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s+2)]), 1)
            x = self.stages[s](x)

            if s in [4]:
               
                loss_graph_mean_list = []
                for r in range(0, recurrent_num):
                    tmp = x

                    if test_flag:
                        x, loss_graph_mean, skel_pred = self.evig_stages[0](x, None, None, None, current_epoch, m2g2m=True, val_flag=False, test_flag=test_flag)
                    else:
                        x, loss_graph_mean = self.evig_stages[0](x, windows_seg_targets_dict[len(self.stages)-s-1], None, windows_graph_nodes_dict[len(self.stages)-s-1], current_epoch, m2g2m=True, val_flag=val_flag, test_flag=test_flag)
                    
                    x = x + tmp
                    loss_graph_mean_list.append(loss_graph_mean)

                if test_flag:
                    seg_output = self.seg_layers[-1](x)
                    seg_output = torch.cat((seg_output, skel_pred), 1)
                    seg_outputs.append(seg_output)
                else:
                    if self.deep_supervision:
                        seg_output = self.seg_layers[s](x)
                        seg_outputs.append(seg_output)
                    elif s == (len(self.stages) - 1):
                        seg_output = self.seg_layers[-1](x)
                        seg_outputs.append(seg_output)
            else:

                if self.deep_supervision:
                    seg_output = self.seg_layers[s](x)
                    seg_outputs.append(seg_output)
                elif s == (len(self.stages) - 1):
                    seg_output = self.seg_layers[-1](x)
                    seg_outputs.append(seg_output)
                
            lres_input = x
            
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            s = seg_outputs[0]
        else:
            s = seg_outputs
        return s, loss_graph_mean_list

    def compute_conv_feature_map_size(self, input_size):
        """
        IMPORTANT: input_size is the input_size of the encoder!
        :param input_size:
        :return:
        """
        # first we need to compute the skip sizes. Skip bottleneck because all output feature maps of our ops will at
        # least have the size of the skip above that (therefore -1)
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]
        # print(skip_sizes)

        assert len(skip_sizes) == len(self.stages)

        # our ops are the other way around, so let's match things up
        output = np.int64(0)
        for s in range(len(self.stages)):
            # print(skip_sizes[-(s+1)], self.encoder.output_channels[-(s+2)])
            # conv blocks
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s+1)])
            # trans conv
            output += np.prod([self.encoder.output_channels[-(s+2)], *skip_sizes[-(s+1)]], dtype=np.int64)
            # segmentation
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s+1)]], dtype=np.int64)
        return output

class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0, conv_op=nn.Conv3d, norm_op=nn.BatchNorm3d, norm_op_kwargs=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            conv_op(in_features, hidden_features, 1, stride=1, padding=0),
            norm_op(hidden_features, **norm_op_kwargs),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            conv_op(hidden_features, out_features, 1, stride=1, padding=0),
            norm_op(out_features, **norm_op_kwargs),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x

class MRConv(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True, conv_op=nn.Conv3d, dropout_op=nn.Dropout3d):
        super(MRConv, self).__init__()
        self.conv_op = conv_op
        self.nn = BasicConv([in_channels*2, out_channels], act=act, norm=norm, bias=bias, drop=0., conv_op=conv_op, dropout_op=dropout_op)

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(x_j - x_i, -1, keepdim=True)
        b, c, n, _ = x.shape
        x = torch.cat([x.unsqueeze(2), x_j.unsqueeze(2)], dim=2).reshape(b, 2 * c, n, _)
        
        if self.conv_op == nn.Conv2d:
            pass
        elif self.conv_op == nn.Conv3d:
            x = torch.unsqueeze(x, dim=4) 
        else:
            raise NotImplementedError('conv operation [%s] is not found' % self.conv_op)
    
        return self.nn(x)
    
class GraphConv(nn.Module):
    """
    Static graph convolution layer
    """
    def __init__(self, in_channels, out_channels, conv='edge', act='relu', norm=None, bias=True, conv_op=nn.Conv3d, dropout_op=nn.Dropout3d):
        super(GraphConv, self).__init__()
        if conv == 'mr':
            self.gconv = MRConv(in_channels, out_channels, act, norm, bias, conv_op, dropout_op)
        else:
            raise NotImplementedError('conv:{} is not supported'.format(conv))

    def forward(self, x, edge_index, y=None):
        return self.gconv(x, edge_index, y)

def get_resample_pos_list(pos_list, scale_list):
    return np.ceil((pos_list + 1) / scale_list).astype(np.int32) - 1 if len(pos_list) != 0 else np.array([])

def batch_windows_pos_indexs(pos, window_size):
    pos_x, pos_y, pos_z = pos

    window_size_x, window_size_y, window_size_z = window_size

    window_ids_x = pos_x // window_size_x
    window_ids_y = pos_y // window_size_y
    window_ids_z = pos_z // window_size_z

    w_i_x = pos_x % window_size_x
    w_i_y = pos_y % window_size_y
    w_i_z = pos_z % window_size_z

    return np.stack((window_ids_x, window_ids_y, window_ids_z), axis=-1), np.stack((w_i_x, w_i_y, w_i_z), axis=-1)

def get_negative_positive_ratio(epoch, alpha, max_epoch=1000, max_ratio=100):
    # negative_positive_ratio = min(15 + max_ratio * 0.75 * ((epoch + 1) / (max_epoch + 1)) ** alpha, max_ratio * 0.8)
    negative_positive_ratio = min(1 + max_ratio * 0.75 * ((epoch + 1) / (max_epoch + 1)) ** alpha, max_ratio * 0.8)
    return negative_positive_ratio

def set_topk_adjacency_matrix(matrix, K):
    _, topk_indices = torch.topk(matrix, K, dim=2)
    mask = torch.zeros_like(matrix).to(matrix.device).float()
    mask.scatter_(2, topk_indices, torch.ones_like(mask).to(matrix.device).float())
    mask_t = mask.transpose(1, 2)
    # mask = torch.max(mask, mask_t).float()
    mask_1_0 = torch.min(mask, mask_t).float()
    mask_0_1 = torch.ones_like(mask).to(matrix.device).float() - mask_1_0
    return mask_1_0, mask_0_1

def reshape_and_reindex_tensor(windows_degree_tensor, Batch, window_size, num_windows):
    w_x, w_y, w_z = window_size
    n_x, n_y, n_z = num_windows
    x, y, z = w_x * n_x, w_y * n_y, w_z * n_z
    N = w_x * w_y * w_z
    B_nW, _, K = windows_degree_tensor.shape
    nW = B_nW // Batch
    windows_degree_tensor = windows_degree_tensor.reshape(Batch, nW, N, K)
    index_map = torch.zeros_like(windows_degree_tensor).to(windows_degree_tensor.device)
    
    windows_pos = torch.nonzero(windows_degree_tensor >= 0, as_tuple=False)
    window_ids = windows_pos[:, 1]
    window_ids_x = window_ids // (n_y * n_z)
    window_ids_y = (window_ids % (n_y * n_z)) // n_z
    window_ids_z = window_ids % n_z

    window_pos_id = windows_degree_tensor[windows_degree_tensor >= 0]

    w_i_x = window_pos_id // (w_y * w_z)
    w_i_y = (window_pos_id % (w_y * w_z)) // w_z
    w_i_z = window_pos_id % w_z
    
    pos_x = window_ids_x * w_x + w_i_x
    pos_y = window_ids_y * w_y + w_i_y
    pos_z = window_ids_z * w_z + w_i_z

    windows_nodes_pos = torch.nonzero(windows_degree_tensor >= 0, as_tuple=False)
    for i in range(0, windows_nodes_pos.shape[0]):
        b_i = windows_nodes_pos[i, 0]
        w_i = windows_nodes_pos[i, 1]
        k_i = windows_nodes_pos[i, 3]
        index_map[b_i, w_i, windows_nodes_pos[i, 2], k_i] = pos_x[i] * y * z + pos_y[i] * z + pos_z[i]
    
    windows_degree_tensor = torch.where(windows_degree_tensor >= 0, index_map, windows_degree_tensor)
    degree_tensor = windows_degree_tensor.reshape(Batch, n_x, n_y, n_z, w_x, w_y, w_z, K).permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(Batch, n_x * w_x * n_y * w_y * n_z * w_z, K)
    
    return degree_tensor

def get_3d_coordinates(indices, shape):
    z = indices // (shape[0] * shape[1])
    y = (indices % (shape[0] * shape[1])) // shape[0]
    x = indices % shape[0]
    return torch.stack((x, y, z), dim=-1)

def process_window(w_i, mask_bool_1_b_i, mask_bool_0_b_i, windows_graph_edges_target_b_i, graph_output_split):
    mask_bool_0_b_i_pos = torch.nonzero(mask_bool_0_b_i[w_i], as_tuple=False)
    pre_nodes_indexs = (graph_output_split[w_i].permute(1, 0) * mask_bool_0_b_i_pos).flatten()
    pre_nodes_indexs = pre_nodes_indexs[pre_nodes_indexs != 0]

    mask_bool_pre_1 = torch.zeros_like(mask_bool_1_b_i[w_i]).squeeze(1).scatter(0, pre_nodes_indexs, 1).unsqueeze(1)

    indices = torch.nonzero(mask_bool_pre_1, as_tuple=True)[0]
    indices_neg = torch.nonzero(~mask_bool_pre_1, as_tuple=True)[0]
    indices_neg_select = torch.randperm(indices_neg.numel())[:indices.numel()]
    indices = torch.cat((indices, indices_neg[indices_neg_select]), dim=0)

    S = indices.shape[0]
    S_adj_matrix = torch.ones((S, S), dtype=torch.bool)
    edge_indexs = S_adj_matrix.nonzero(as_tuple=False)
    src_edge, dst_edge = edge_indexs[:, 0], edge_indexs[:, 1]

    temp_tensor = windows_graph_edges_target_b_i[w_i][indices]
    non_zero_rows_mask = temp_tensor.any(dim=1)
    non_zero_rows = temp_tensor[non_zero_rows_mask]
    non_zero_rows -= 1

    idx_result_list = []

    indices_dict = {v.item(): k for k, v in enumerate(indices)}

    for row in non_zero_rows:
        first_element = row[0]
        non_zero_elements = row[row != -1]

        for non_zero in non_zero_elements:
            if first_element.item() in indices_dict and non_zero.item() in indices_dict:
                idx_result_list.append([indices_dict[first_element.item()], indices_dict[non_zero.item()]])

    idx_result_set = set(tuple(x) for x in idx_result_list)
    edge_labels = torch.tensor([1 if (src_edge[e_i].item(), dst_edge[e_i].item()) in idx_result_set else 0 for e_i in range(src_edge.shape[0])])

    return edge_labels, src_edge, dst_edge, indices

def compute_euclidean_distance_map(segmentation, centerline):
	seg = segmentation.astype(np.bool)
	cen = centerline.astype(np.bool)
	map = ndimage.distance_transform_edt(np.logical_not(cen))

	map_max = np.max(map[seg > 0])
	map /= map_max
	map[seg == 0] = 1
	map = np.log2(map, out=np.zeros_like(map), where=(map!=0))
	map_min = np.min(map)
	map[cen>0] = map_min
	map -= np.min(map)
	map /= np.max(map)

	return map


def compute_normal_direction(centerline):
    # Compute the tangent direction at each point in the centerline
    tangents = (np.roll(centerline, -1, axis=0) - np.roll(centerline, 1, axis=0))[:-2]
    
    tangents = tangents.astype(np.float32)

    # Normalize the tangents
    tangents /= np.linalg.norm(tangents, axis=-1, keepdims=True)

    # Compute the normal direction by taking cross product of tangents with Z-axis
    normals = np.cross(tangents, np.array([0, 0, 1]))

    # Normalize the normals
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
   
    return normals

def create_graph_from_centerline(centerline, num_neighbors):
    if len(centerline) < num_neighbors:
        return None
    nbrs = NearestNeighbors(n_neighbors=num_neighbors, algorithm='ball_tree').fit(centerline)
    distances, indices = nbrs.kneighbors(centerline)
    G = nx.Graph()
    for i in range(len(centerline)):
        G.add_node(i, pos=centerline[i])

    edges = [(i, indices[i, j], distances[i, j]) for i in range(len(centerline)) for j in range(1, num_neighbors)]
    G.add_weighted_edges_from(edges)
    T = nx.minimum_spanning_tree(G, algorithm="kruskal", weight="weight")
    return T

def batch_id2pos_indexs_k(id, size):
    x, y, z = size
    
    pos_x = (id // (y * z)).type(torch.int32)
    pos_y = ((id // z) % y).type(torch.int32)
    pos_z = (id % z).type(torch.int32)
    
    pos = torch.stack((pos_x, pos_y, pos_z))
    
    return pos

def get_radius_weights(y_true, skel_true, H, W, D):
    # dist_map_3d = distmap.euclidean_distance_transform(y_true)
    # https://docs.cupy.dev/en/stable/user_guide/interoperability.html
    y_true_cupy_array = cp.from_dlpack(to_dlpack(y_true))
    dist_map_3d_cupy_array = distance_transform_edt_cupy(y_true_cupy_array)
    dist_map_3d = from_dlpack(dist_map_3d_cupy_array.toDlpack())
    
    dist_map_3d[y_true == 0] = 0
    vessel_radius = dist_map_3d[skel_true == 1]

    smooth = 0.001
    if vessel_radius.shape[0] == 0 or vessel_radius.min() == vessel_radius.max():
        vessel_radius_weights = skel_true.clone()
    else:
        #vessel_radius *= 3.1416
        #vessel_radius_min = vessel_radius.min()
        #vessel_radius_max = vessel_radius.max()
        #vessel_radius = (vessel_radius - vessel_radius_min) / (vessel_radius_max - vessel_radius_min)
        #vessel_radius = (1 + smooth) / (vessel_radius**2 + smooth)
        
        smooth = 1e-7
        #vessel_radius_min = vessel_radius.min()
        vessel_radius_max = vessel_radius.max()
        vessel_radius_0_1 = vessel_radius / vessel_radius_max
        vessel_radius = (1 + smooth) / (vessel_radius_0_1**1 + smooth)

        #vessel_radius_min = vessel_radius.min()
        #vessel_radius_max = vessel_radius.max()
        #vessel_radius_0_1 = (vessel_radius - vessel_radius_min) / (vessel_radius_max - vessel_radius_min)
        #vessel_radius = (1 + smooth) / (3.1416*vessel_radius_0_1**2 + smooth)

        #vessel_radius_weight = (1 + smooth) / (3.1416*vessel_radius**2 + smooth)
        #vessel_radius_weight_min = vessel_radius_weight.min()
        #vessel_radius_weight_max = vessel_radius_weight.max()
        #vessel_radius = (vessel_radius_weight - vessel_radius_weight_min) / (vessel_radius_weight_max - vessel_radius_weight_min)

        vessel_radius_weights = torch.zeros_like(skel_true)
        N = H * W * D
        skel_N = skel_true.view(N)
        nodes = (skel_N == 1).nonzero(as_tuple=False).squeeze()
        nodes_pos = batch_id2pos_indexs_k(nodes, (H, W, D)).T
        for node_index in range(nodes_pos.shape[0]):
            pos = nodes_pos[node_index]
            vessel_radius_weights[pos[0], pos[1], pos[2]] = vessel_radius[node_index]
    
    return vessel_radius_weights

class SkeletonGraph():
    def __init__(self, stage_shape_list):
        self.stage_shape_list = stage_shape_list
    
    def process_skeleton_prediction(self, skel_pred, current_epoch, device, k, N):
        self.windows_graph_degree_tensor_dict = {}
        for b_i in range(skel_pred.shape[0]):
            self.process_single_batch(skel_pred[b_i], current_epoch, k)            

        edge_index = self.post_process(device, k, N)
        return edge_index
    
    def process_single_batch(self, skel_pred_single, current_epoch, k):
        skel_pred_np = skel_pred_single.detach().cpu().numpy()
        raw_shape = np.array(skel_pred_np.shape)
        nodes_pos_np = np.array(np.nonzero(skel_pred_np)).transpose(1, 0)
        pos_list = nodes_pos_np
        bidirectional_edges_list = []
        
        if current_epoch != 0:
            G = create_graph_from_centerline(nodes_pos_np, 5)
            if G is not None:
                new_edges_list = [list(edge) for edge in G.edges]
                new_edges_reverse_list = [edge[::-1] for edge in new_edges_list]
                bidirectional_edges_list = new_edges_list + new_edges_reverse_list

        for s in [0]:
            new_shape = np.array(self.stage_shape_list[s])
            scale_list = raw_shape // new_shape
            resample_pos_list = get_resample_pos_list(np.array(pos_list), scale_list)
            index_dict = {tuple(pos): new_shape[1] * new_shape[2] * pos[0] + new_shape[2] * pos[1] + pos[2] for pos in resample_pos_list}

            resample_points_data_3D_degree = np.zeros((k,) + tuple(new_shape))
            resample_points_data_3D_degree_dict = {}

            for edge in bidirectional_edges_list:
                node1, node2 = edge
                node1_index = index_dict[tuple(resample_pos_list[node1])]
                node2_index = index_dict[tuple(resample_pos_list[node2])]

                resample_points_data_3D_degree_dict.setdefault(node1_index, set())

                if not resample_points_data_3D_degree[0, resample_pos_list[node1][0], resample_pos_list[node1][1], resample_pos_list[node1][2]]:
                    resample_points_data_3D_degree[0, resample_pos_list[node1][0], resample_pos_list[node1][1], resample_pos_list[node1][2]] = node1_index
                    resample_points_data_3D_degree_dict[node1_index].add(node1_index)

                for d in range(1, k):
                    if resample_points_data_3D_degree[d, resample_pos_list[node1][0], resample_pos_list[node1][1], resample_pos_list[node1][2]] == 0 and node2_index not in resample_points_data_3D_degree_dict[node1_index]:
                        resample_points_data_3D_degree[d, resample_pos_list[node1][0], resample_pos_list[node1][1], resample_pos_list[node1][2]] = node2_index
                        resample_points_data_3D_degree_dict[node1_index].add(node2_index)
                        break
                        
            if s not in self.windows_graph_degree_tensor_dict:
                self.windows_graph_degree_tensor_dict[s] = []
            self.windows_graph_degree_tensor_dict[s].append(resample_points_data_3D_degree)
            
    def post_process(self, device, k, N):
        for stage in self.windows_graph_degree_tensor_dict:
            windows_graph_degree_stage_tensor = torch.tensor(np.array(self.windows_graph_degree_tensor_dict[stage], dtype=np.float32)).float()
            self.windows_graph_degree_tensor_dict[stage] = windows_graph_degree_stage_tensor
        
        windows_graph_edges_target_dict = {}
        for s in [0]:
            tensor = self.windows_graph_degree_tensor_dict[s]
            B, nD, H, W, D = tensor.shape
            tensor = tensor.reshape(B, nD, H * W * D).long()
            windows_graph_edges_target_dict[s] = tensor.permute(0, 2, 1)

        s = 0
        windows_graph_edges_target = windows_graph_edges_target_dict[s]
        windows_graph_edges_target = windows_graph_edges_target.reshape(B, N, k).to(device)
        windows_degree_tensor_pre_b0 = -N*torch.ones_like(windows_graph_edges_target).to(device)
        
        windows_degree_tensor_pre_b0 = windows_degree_tensor_pre_b0
        windows_graph_edges_target = windows_graph_edges_target
        windows_N_k = torch.arange(N, device=windows_degree_tensor_pre_b0.device).view(1, -1, 1).expand(B, -1, k)
        windows_degree_tensor_pre = torch.where(windows_degree_tensor_pre_b0 > 0, windows_degree_tensor_pre_b0, windows_N_k)

        n_points = N
        center_idx = torch.arange(0, n_points, device=device).repeat(B, k, 1).transpose(2, 1)
    
        nn_idx = windows_degree_tensor_pre
        edge_index = torch.stack((nn_idx, center_idx), dim=0)
        return edge_index

class DyGraphConv(GraphConv):
    """
    Dynamic graph convolution layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='edge', act='relu',
                 norm=None, bias=True, stochastic=False, epsilon=0.0, r=1, conv_op=nn.Conv3d, dropout_op=nn.Dropout3d):
        super(DyGraphConv, self).__init__(in_channels, out_channels, conv, act, norm, bias, conv_op, dropout_op)
        self.k = kernel_size
        self.d = dilation
        self.r = r
        self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)
        self.conv_op = conv_op
        self.dropout_op = dropout_op
        if self.conv_op == nn.Conv2d:
            self.avg_pool = F.avg_pool2d
        elif self.conv_op == nn.Conv3d:
            self.avg_pool = F.avg_pool3d
        else:
            raise NotImplementedError('conv operation [%s] is not found' % self.conv_op)
        
        self.mse_loss = nn.MSELoss(reduction='mean')

        self.graph_predictor = GraphPredictor(in_channels)
        #stage_shape_list = np.array([[64, 224, 224], [32, 112, 112], [16, 56, 56], [8, 28, 28], [4, 14, 14]])
        stage_shape_list = np.array([[64, 192, 192], [32, 96, 96], [16, 48, 48], [8, 24, 24], [4, 12, 12]])
        self.skel2graph = SkeletonGraph(stage_shape_list)

        self.graph_node_loss = DC_and_CE_loss({'batch_dice': True,
                                   'smooth': 1e-5, 'do_bg': False, 'ddp': False}, {}, weight_ce=0.5, weight_dice=0.5,
                                  ignore_label=None, dice_class=MemoryEfficientSoftDiceLoss)


    def forward(self, x, relative_pos=None, windows_seg_target=None, windows_graph_edges_target=None, windows_graph_nodes=None, img_shape=None, current_epoch=None, m2g2m=True, val_flag=False, test_flag=True):
        if self.conv_op == nn.Conv2d:
            B, C, H, W = x.shape
        elif self.conv_op == nn.Conv3d:
            B, C, H, W, D = x.shape
        else:
            raise NotImplementedError('conv operation [%s] is not found' % self.conv_op)

        y = None
        if self.r > 1:
            y = self.avg_pool(x, self.r, self.r)
            y = y.reshape(B, C, -1, 1).contiguous()            
        x = x.reshape(B, C, -1, 1).contiguous()

        self.f_shape = img_shape
        device = x.device
        
        if m2g2m == True:
            if test_flag == False:
                Batch = B
                N = H * W * D

                feature_map_windows = x.reshape(Batch, C, H, W, D)

                target_save_np = windows_graph_nodes.squeeze(1).detach().cpu().numpy()

                windows_seg_target_save_np = windows_seg_target.squeeze(1).detach().cpu().numpy()

                seg_node_3d_np = np.where(target_save_np == 1, 2, 0)
                seg_node_3d_np = np.where(seg_node_3d_np == 2, 2, windows_seg_target_save_np)

                seg_node_3d = torch.from_numpy(seg_node_3d_np).to(device).unsqueeze(1).float()

                pre_results = self.graph_predictor(feature_map_windows)
                seg_results = pre_results
                
                loss_graph_nodes_mean = self.graph_node_loss(seg_results, seg_node_3d.detach())
                seg_results_prob = torch.softmax(seg_results, 1)

                seg_pre = torch.argmax(seg_results_prob, dim=1)
                skel_pred = torch.where(seg_pre == 2, 1, 0).squeeze(1)
                skel_true = torch.where(seg_node_3d.detach() == 2, 1, 0).squeeze(1)
                y_pred = torch.where(seg_pre > 0, 1, 0).squeeze(1)
                y_true = torch.where(seg_node_3d.detach() > 0, 1, 0).squeeze(1)

                self.smooth = 1.

                if loss_graph_nodes_mean > 0.3: #current_epoch == 0:
                    radii_weights_true = skel_true
                    radii_weights_pred = skel_pred
                else:
                    radii_weights_true = torch.zeros_like(y_true)
                    radii_weights_pred = torch.zeros_like(y_pred)
                    
                    for b_i in range(Batch):
                        radii_weights_true[b_i] = get_radius_weights(y_true[b_i], skel_true[b_i], H, W, D)
                        radii_weights_pred[b_i] = get_radius_weights(y_pred[b_i], skel_pred[b_i], H, W, D)

                weighted_tprec = (torch.sum(torch.multiply(radii_weights_pred, y_true))+self.smooth)/(torch.sum(radii_weights_pred)+self.smooth)
                weighted_tsens = (torch.sum(torch.multiply(radii_weights_true, y_pred))+self.smooth)/(torch.sum(radii_weights_true)+self.smooth)
                cl_dice = - 2.0 * (weighted_tprec * weighted_tsens) / (weighted_tprec + weighted_tsens)
                print("cl_dice: ", cl_dice)

                print("loss_graph_nodes_mean: ", loss_graph_nodes_mean)
                loss_graph_nodes_mean = loss_graph_nodes_mean + 0.5 * cl_dice
                
                edge_index = self.skel2graph.process_skeleton_prediction(skel_pred, current_epoch, device, self.k, N)

                loss_graph_mean = loss_graph_nodes_mean

            else:
                Batch = B
                N = H * W * D

                feature_map_windows = x.reshape(Batch, C, H, W, D)

                pre_results = self.graph_predictor(feature_map_windows)
                seg_results = pre_results

                seg_results_prob = torch.softmax(seg_results, 1)
                
                seg_pre = torch.argmax(seg_results_prob, dim=1)
                skel_pred = torch.where(seg_pre == 2, 1, 0).squeeze(1)

                edge_index = self.skel2graph.process_skeleton_prediction(skel_pred, current_epoch, device, self.k, N)

                loss_graph_mean = 0
                radii_weights_true = 0
                radii_weights_pred = 0
                res_dict = {0: radii_weights_true, 1: radii_weights_pred, 2: loss_graph_mean}

                edge_index_dst, edge_index_src = edge_index[0], edge_index[1]
                B, shape_multi, k = edge_index_src.shape
                
                x= super(DyGraphConv, self).forward(x, edge_index, y)

                if self.conv_op == nn.Conv2d:
                    return x.reshape(B, -1, H, W).contiguous(), res_dict, seg_results
                elif self.conv_op == nn.Conv3d:
                    return x.reshape(B, -1, H, W, D).contiguous(), res_dict, seg_results
                else:
                    raise NotImplementedError('conv operation [%s] is not found' % self.conv_op)
                
        else:
            loss_graph_mean = 0
            radii_weights_true = 0
            radii_weights_pred = 0
            nn_idx = 0
            edge_index = self.dilated_knn_graph(x, y, relative_pos)
        
        res_dict = {0: radii_weights_true, 1: radii_weights_pred, 2: loss_graph_mean}
        edge_index_dst, edge_index_src = edge_index[0], edge_index[1]
        B, shape_multi, k = edge_index_src.shape
        
        x= super(DyGraphConv, self).forward(x, edge_index, y)

        if self.conv_op == nn.Conv2d:
            return x.reshape(B, -1, H, W).contiguous(), res_dict
        elif self.conv_op == nn.Conv3d:
            return x.reshape(B, -1, H, W, D).contiguous(), res_dict
        else:
            raise NotImplementedError('conv operation [%s] is not found' % self.conv_op)

class VesselGrapher(nn.Module):
    """
    VesselGrapher module with graph convolution and fc layers
    """
    def __init__(self, in_channels, img_shape, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
                 bias=True,  stochastic=False, epsilon=0.0, r=1, n=196, drop_path=0.0, relative_pos=False, 
                 conv_op=nn.Conv3d, norm_op=nn.BatchNorm3d, norm_op_kwargs=None, dropout_op=nn.Dropout3d):
        super(VesselGrapher, self).__init__()
        self.channels = in_channels
        # self.n = n
        self.r = r
        self.conv_op = conv_op
        self.img_shape = img_shape

        self.fc1 = nn.Sequential(
            conv_op(in_channels, in_channels, 1, stride=1, padding=0),
            norm_op(in_channels, **norm_op_kwargs),
        )
        norm = 'batch'
        self.graph_conv = DyGraphConv(in_channels, in_channels * 2, kernel_size, dilation, conv,
                                      act, norm, bias, stochastic, epsilon, r, conv_op, dropout_op)
        self.fc2 = nn.Sequential(
            conv_op(in_channels * 2, in_channels, 1, stride=1, padding=0),
            norm_op(in_channels, **norm_op_kwargs),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.relative_pos = None
    
    def forward(self, x, windows_seg_target, windows_graph_edges_target, windows_graph_nodes, current_epoch, m2g2m=True, val_flag=False, test_flag=True):
        _tmp = x
        if self.conv_op == nn.Conv2d:
            B, C, H, W = x.shape
            size_tuple = (H, W)
            h, w = self.img_shape
            assert torch.all(torch.tensor(H).eq(h)) and torch.all(torch.tensor(W).eq(w)), "input feature has wrong size"
        elif self.conv_op == nn.Conv3d:
            B, C, S, H, W = x.shape
            size_tuple = (S, H, W)
            s, h, w = self.img_shape
            assert torch.all(torch.tensor(S).eq(s)) and torch.all(torch.tensor(H).eq(h)) and torch.all(torch.tensor(W).eq(w)), "input feature has wrong size"
        else:
            raise NotImplementedError('conv operation [%s] is not found' % self.conv_op)

        x = self.fc1(x)

        if m2g2m and test_flag:
            x, loss_graph_mean, skel_pred = self.graph_conv(x, relative_pos=None, windows_seg_target=windows_seg_target, windows_graph_edges_target=windows_graph_edges_target, windows_graph_nodes=windows_graph_nodes, img_shape=self.img_shape, current_epoch=current_epoch, m2g2m=m2g2m, val_flag=val_flag, test_flag=test_flag)
            x = self.fc2(x)
            x = self.drop_path(x) + _tmp
            return x, loss_graph_mean, skel_pred
        
        else:
            x, loss_graph_mean = self.graph_conv(x, relative_pos=None, windows_seg_target=windows_seg_target, windows_graph_edges_target=windows_graph_edges_target, windows_graph_nodes=windows_graph_nodes, img_shape=self.img_shape, current_epoch=current_epoch, m2g2m=m2g2m, val_flag=val_flag, test_flag=test_flag)
            x = self.fc2(x)
            x = self.drop_path(x) + _tmp
            return x, loss_graph_mean

class Efficient_ViG_blocks(nn.Module):
    def __init__(self, channels, img_shape, index, conv_layer_d_num, opt=None, conv_op=nn.Conv3d, norm_op=nn.BatchNorm3d, norm_op_kwargs=None,
                    dropout_op=nn.Dropout3d, **kwargs):
        super(Efficient_ViG_blocks, self).__init__()

        vig_blocks = []
        ffn_blocks = []
        k = opt.k
        conv = opt.conv
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        epsilon = opt.epsilon
        stochastic = opt.use_stochastic
        drop_path = opt.drop_path
        reduce_ratios = opt.reduce_ratios 
        blocks_num_list = opt.blocks
        n_size_list = opt.n_size_list
        img_min_shape = opt.img_min_shape

        self.n_blocks = sum(blocks_num_list)        
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]
        sum_blocks = sum(blocks_num_list[conv_layer_d_num-2:index])
        idx_list = [(k+sum_blocks) for k in range(0, blocks_num_list[index])]
        
        
        if conv_op == nn.Conv2d:
            H_min, W_min = img_min_shape
            max_dilation = (H_min * W_min) // max(k)
            window_size = img_min_shape
            window_size_n = window_size[0] * window_size[1]   
        elif conv_op == nn.Conv3d:
            H_min, W_min, D_min = img_min_shape
            max_dilation = (H_min * W_min * D_min) // max(k)  
            window_size = img_min_shape
            window_size_n = window_size[0] * window_size[1] * window_size[2]      
        else:
            raise NotImplementedError('conv operation [%s] is not found' % conv_op)

        i = conv_layer_d_num-2 + index
        for j in range(blocks_num_list[index]):
            idx = idx_list[j]
            if conv_op == nn.Conv2d:
                shift_size = [window_size[0] // 2, window_size[1] // 2]
            elif conv_op == nn.Conv3d:
                shift_size = [window_size[0] // 2, window_size[1] // 2, window_size[2] // 2]
            else:
                raise NotImplementedError('conv operation [%s] is not found' % conv_op)

            vig_blocks.append(
                        VesselGrapher(channels, img_shape, k[i], min(idx // 4 + 1, max_dilation), conv, act, norm,
                        bias, stochastic, epsilon, 1, window_size_n, drop_path=dpr[idx], relative_pos=True, conv_op=conv_op, norm_op=norm_op, 
                        norm_op_kwargs=norm_op_kwargs, dropout_op=dropout_op))

            ffn_blocks.append(FFN(channels, channels * 4, act=act, drop_path=dpr[idx], conv_op=conv_op, norm_op=norm_op, norm_op_kwargs=norm_op_kwargs))

        self.vig_blocks = nn.ModuleList(vig_blocks)
        self.ffn_blocks = nn.ModuleList(ffn_blocks)

    def forward(self, x, windows_seg_target, windows_graph_edges_target, windows_graph_nodes, current_epoch, m2g2m=True, val_flag=False, test_flag=True): 
        if m2g2m and test_flag:
            for i in range(len(self.vig_blocks)):
                x, loss_graph_mean, skel_pred = self.vig_blocks[i](x, windows_seg_target, windows_graph_edges_target, windows_graph_nodes, current_epoch, m2g2m=m2g2m, val_flag=val_flag, test_flag=test_flag)
                x = self.ffn_blocks[i](x)
            return x, loss_graph_mean, skel_pred
        else:
            for i in range(len(self.vig_blocks)):
                x, loss_graph_mean = self.vig_blocks[i](x, windows_seg_target, windows_graph_edges_target, windows_graph_nodes, current_epoch, m2g2m=m2g2m, val_flag=val_flag, test_flag=test_flag)
                x = self.ffn_blocks[i](x)
            return x, loss_graph_mean
    
class GraphPredictor(nn.Module):

    def __init__(self, decoder_dim):
        super(GraphPredictor, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Conv3d(decoder_dim, 3, 1, stride=1, padding=0, bias=True),
        )

    def forward(self, feature_map):
        graph_mlp_results = self.fc1(feature_map)
        return graph_mlp_results

