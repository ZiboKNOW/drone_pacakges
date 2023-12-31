from os.path import join
import torch
from torch import nn, sigmoid
from torch._C import device
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import kornia
import matplotlib.pyplot as plt
import cv2
import math
import numpy as np
from ..models.networks.DCNv2.dcn_v2 import DCN
from Conet.lib.utils.image import get_affine_transform
import sys
from Conet.lib.opts import opts
BN_MOMENTUM = 0.1
def get_model_url(data='imagenet', name='dla34', hash='ba72cf86'):
    return join('http://dl.yf.io/dla/models', data, '{}-{}.pth'.format(name, hash))

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]

def dla34(pretrained=True, **kwargs):  # DLA-34
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 128, 256, 512],
                block=BasicBlock, **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla34', hash='ba72cf86')
    return model
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out

class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x

class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride,
                               dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1,
                               dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x

class DLA(nn.Module):
    def __init__(self, levels, channels, num_classes=1000,
                 block=BasicBlock, residual_root=False, linear_root=False):
        super(DLA, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                      padding=3, bias=False),
            nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False,
                           root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_level(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.MaxPool2d(stride, stride=stride),
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        return y

    def load_pretrained_model(self, data='imagenet', name='dla34', hash='ba72cf86'):
        # fc = self.fc
        if name.endswith('.pth'):
            model_weights = torch.load(data + name)
        else:
            model_url = get_model_url(data, name, hash)
            model_weights = model_zoo.load_url(model_url)
        num_classes = len(model_weights[list(model_weights.keys())[-1]])
        self.fc = nn.Conv2d(
            self.channels[-1], num_classes,
            kernel_size=1, stride=1, padding=0, bias=True)
        self.load_state_dict(model_weights)
        # self.fc = fc

class DLAUp(nn.Module):
    def __init__(self, startp, channels, scales, in_channels=None):
        super(DLAUp, self).__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels) #每一层的channel数
        scales = np.array(scales, dtype=int) #每一层的scales
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i),
                    IDAUp(channels[j], in_channels[j:],
                          scales[j:] // scales[j]))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        out = [layers[-1]] # start with 32
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) -i - 2, len(layers))
            out.insert(0, layers[-1])
        return out
    
class IDAUp(nn.Module):

    def __init__(self, o, channels, up_f):
        super(IDAUp, self).__init__()
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])  
            proj = DeformConv(c, o)
            node = DeformConv(o, o)
     
            up = nn.ConvTranspose2d(o, o, f * 2, stride=f, 
                                    padding=f // 2, output_padding=0,
                                    groups=o, bias=False)
            fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)
                 
        
    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])
            # if layers[i].shape == layers[i - 1].shape:
            #     layers[i] = node(layers[i] + layers[i - 1])
            # else:
            #     layers[i] = node(layers[i])

class DeformConv(nn.Module):
    def __init__(self, chi, cho, k_size=3, dilation=1, mode='DCN'):
        super(DeformConv, self).__init__()
        self.mode = mode
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.conv = eval(mode)(chi, cho, kernel_size=(k_size,k_size), stride=1, padding=(k_size-1)//2, dilation=dilation, deformable_groups=1)

    def forward(self, x, return_offset=False):
        if self.mode == 'DCN':
            x = self.conv(x)
            x = self.actf(x)
            return x
        else:
            x, offset, mask = self.conv(x)
            x = self.actf(x)
            if return_offset:
                return x, offset, mask
            else:
                return x


class features_extractor(nn.Module):
    def __init__(self, pretrained, down_ratio, feat_shape, feat_mode, trans_layer, in_channels=None, vis =False):
        super(features_extractor, self).__init__()
        self.opt = opts().parse()
        self.opt=opts().update_dataset_info_and_set_heads(self.opt)
        self.feat_H, self.feat_W = feat_shape
        self.first_level = int(np.log2(down_ratio))
        self.feat_mode = feat_mode
        self.vis = vis
        self.trans_layer = trans_layer
        self.base = dla34(pretrained=True)
        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales)
        pretrain_dict = torch.load('/home/cmic2/ros_workspace/src/drone_detection/model/model_best.pth')
        # print('pretrain_dict: ',pretrain_dict)
        model_dict = {}
        state_dict = self.dla_up.state_dict()
        backbone_dict = {}
        state_dict_backbone = self.base.state_dict()
        len_total_parameter_encoder_model = len(state_dict_backbone) +len(state_dict)
        print('all the parameter: ',len_total_parameter_encoder_model)
        update_num_encoder = 0
        total_num = 0
        for k, v in pretrain_dict["state_dict"].items():
            total_num+=1
            for name in state_dict.keys():
                if name in k:
                    model_dict[name] = v
                    update_num_encoder+=1
            for backbone_name in state_dict_backbone.keys():
                if 'base.' + backbone_name in k:
                    update_num_encoder+=1
                    backbone_dict[backbone_name] = v
                    # print('origin_name: ',k)
                    # print('backbone_name: ','base.' + backbone_name)
        print('update_num_encoder: ',update_num_encoder)
        print('total_num_pretrain: ',total_num)
        state_dict.update(model_dict)
        self.dla_up.load_state_dict(state_dict)
        state_dict_backbone.update(backbone_dict)
        self.base.load_state_dict(state_dict_backbone)
                        
    # def hard_warping(self, shift_mats, trans_mats, cur_x, scale, images):
    #     warp_images_list = []
    #     global_x = []
    #     for c_layer, feat_map in enumerate(cur_x): # list [(B, 64, 112, 200), (B, 128, 56, 100), (B, 256, 28, 50), (B, 512, 14, 25)]
    #         b, c, h, w = feat_map.size()
    #         init_shift_mats = shift_mats[c_layer].view(b, 3, 3).contiguous()

    #         worldgrid2worldcoord_mat = torch.Tensor(np.array([[2**c_layer*scale, 0, 0], [0, 2**c_layer*scale, 0], [0, 0, 1]])).to(trans_mats.device)
    #         img_trans_mats = init_shift_mats @ torch.inverse(trans_mats @ worldgrid2worldcoord_mat).contiguous() #from camera to BEV in UAV view
    #         warp_images = kornia.warp_perspective(images, img_trans_mats, dsize=(int(self.feat_h/2**c_layer*scale), int(self.feat_w/2**c_layer*scale)))
    #         warp_images_list.append(warp_images)
    #         #########################################################################
    #         #                              Hard Warping                             #
    #         #########################################################################
    #         # uav_i --> global coord
    #         worldgrid2worldcoord_mat = torch.Tensor(np.array([[2**c_layer/scale, 0, 0], [0, 2**c_layer/scale, 0], [0, 0, 1]])).to(trans_mats.device)
    #         feat_zoom_mats = torch.Tensor(np.array(np.diag([2**(c_layer+2), 2**(c_layer+2), 1]), dtype=np.float32)).to(trans_mats.device)
    #         cur_trans_mats = init_shift_mats @ torch.inverse(trans_mats @ worldgrid2worldcoord_mat).contiguous() @ feat_zoom_mats
    #         global_feat = kornia.warp_perspective(feat_map, cur_trans_mats, dsize=(int(self.feat_h/2**c_layer*scale), int(self.feat_w/2**c_layer*scale)))
    #         global_x.append(global_feat) # 
    #     return warp_images_list, global_x
    def pre_process(self, image, scale, meta=None):
        height, width = image.shape[0:2]
        new_height = int(height * scale)
        new_width = int(width * scale)
        # true
        if self.opt.fix_res:
            # 480 736
            inp_height, inp_width = self.opt.input_h, self.opt.input_w
            c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
            s = max(height, width) * 1.0
        else:
            inp_height = (new_height | self.opt.pad) + 1
            inp_width = (new_width | self.opt.pad) + 1
            c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
            s = np.array([inp_width, inp_height], dtype=np.float32)
        trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
        resized_image = cv2.resize(image, (new_width, new_height))
        inp_image = cv2.warpAffine(
            resized_image, trans_input, (inp_width, inp_height),
            flags=cv2.INTER_LINEAR)
        # inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)
        inp_image = (inp_image / 255.).astype(np.float32)

        images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
        if self.opt.flip_test:
            images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
        images = torch.from_numpy(images)
        meta_i = {'c': c, 's': s,
                'out_height': inp_height // 4,
                'out_width': inp_width // 4}

        feat_h, feat_w = self.opt.feat_shape
        c = np.array([feat_w/(2*self.opt.map_scale), feat_h/(2*self.opt.map_scale)])
        s = np.array([feat_w/(self.opt.map_scale), feat_h/(self.opt.map_scale)])
        c = torch.from_numpy(c).to(torch.float32)
        s = torch.from_numpy(s).to(torch.float32)
        out_height = np.array([feat_h/(self.opt.map_scale)])
        out_width = np.array([feat_w/(self.opt.map_scale)])
        out_height = torch.from_numpy(out_height).to(torch.float32)
        out_width = torch.from_numpy(out_width).to(torch.float32)
        meta = {'c': c, 's': s,
                'out_height': out_height,
                'out_width': out_width}            
        if self.opt.coord == 'Local':
            return images, meta_i
        elif self.opt.coord == 'Global':
            return images, meta
        elif self.opt.coord == 'Joint':
            return images, [meta, meta_i]

    def forward(self, images, trans_mats, shift_mats, map_scale):
        b, num_agents, img_c, img_h, img_w = images.size()
        print('single detection in GlobalCoord_forward: ', self.feat_mode)
        images = images.view(b*num_agents, img_c, img_h, img_w)
        
        if self.vis:
            cur_trans_mats = trans_mats[0].view(b*num_agents, 3, 3)
            worldgrid2worldcoord_mat = torch.Tensor(np.array([[map_scale, 0, 0], [0, map_scale, 0], [0, 0, 1]])).to(cur_trans_mats.device)
            cur_trans_mats = shift_mats[0].view(b*num_agents, 3, 3) @ torch.inverse(cur_trans_mats @ worldgrid2worldcoord_mat).contiguous()
            images_warped = kornia.warp_perspective(images, cur_trans_mats, dsize=(self.feat_H, self.feat_W))
            images_warped = images_warped.contiguous().view(b, num_agents, 3, self.feat_H, self.feat_W).contiguous()
            images_warped = self.get_colla_feats([images_warped], shift_mats, [0], with_pos=False)

        if self.feat_mode in ['early', 'fused']:
            # print('Warp image')
            cur_trans_mats = trans_mats[0].view(b*num_agents, 3, 3)
            worldgrid2worldcoord_mat = torch.Tensor(np.array([[map_scale, 0, 0], [0, map_scale, 0], [0, 0, 1]])).to(cur_trans_mats.device)
            feat_zoom_mats = torch.Tensor(np.array(np.diag([4, 4, 1]), dtype=np.float32)).to(cur_trans_mats.device)
            cur_trans_mats = feat_zoom_mats @ shift_mats[0].view(b*num_agents, 3, 3) @ torch.inverse(cur_trans_mats @ worldgrid2worldcoord_mat).contiguous()
            images_warped = kornia.warp_perspective(images, cur_trans_mats, dsize=(self.feat_H*4, self.feat_W*4))
            # image = images[0].detach().cpu() * 255.
            # image = kornia.tensor_to_image(image.byte())
            # cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # cv2.imwrite('warp_img.png', image)
            
            x_early = self.base_early(images_warped)
            x_early = self.dla_up_early(x_early)

        if self.feat_mode in ['inter', 'fused']:
            # Encoder
            x_inter = self.base(images)
            x_inter = self.dla_up(x_inter)  # list [(B, 64, 112, 200), (B, 128, 56, 100), (B, 256, 28, 50), (B, 512, 14, 25)] B = b * num_agents
            print('base len: ',len(self.base.state_dict()))
            print('dla_up len: ',len(self.dla_up.state_dict()))
        if self.feat_mode in ['early', 'inter']:
            x = x_early if self.feat_mode == 'early' else x_inter
        elif self.feat_mode == 'fused':
            x = x_inter

        scale = 1/map_scale
        trans_mats = [trans_mats[0]]
        depth_weighted_feat_maps = [x]
        
        global_x_multi_depth = []
        
        for depth_layer, init_trans_mats in enumerate(trans_mats):
            cur_x = depth_weighted_feat_maps[depth_layer]
            global_x = []
            warp_images_list = []
            init_trans_mats = init_trans_mats.view(b*num_agents, 3, 3)
            # cur_trans_mats = init_trans_mats.view(b*num_agents, 3, 3)
            # fig, axes = plt.subplots(5, 4)

            for c_layer, feat_map in enumerate(cur_x):
                _, c, h, w = feat_map.size()
                init_shift_mats = shift_mats[c_layer].view(b*num_agents, 3, 3).contiguous()

                worldgrid2worldcoord_mat = torch.Tensor(np.array([[2**c_layer*scale, 0, 0], [0, 2**c_layer*scale, 0], [0, 0, 1]])).to(init_trans_mats.device)
                img_trans_mats = init_shift_mats @ torch.inverse(init_trans_mats @ worldgrid2worldcoord_mat).contiguous()
                warp_images = kornia.warp_perspective(images, img_trans_mats, dsize=(int(self.feat_H/2**c_layer*scale), int(self.feat_W/2**c_layer*scale)))
                warp_images_list.append(warp_images)

                #########################################################################
                #                              Hard Warping                             #
                #########################################################################
                # uav_i --> global coord
                worldgrid2worldcoord_mat = torch.Tensor(np.array([[2**c_layer*map_scale, 0, 0], [0, 2**c_layer*map_scale, 0], [0, 0, 1]])).to(init_trans_mats.device)
                feat_zoom_mats = torch.Tensor(np.array(np.diag([2**(c_layer+2), 2**(c_layer+2), 1]), dtype=np.float32)).to(init_trans_mats.device)
                cur_trans_mats = init_shift_mats @ torch.inverse(init_trans_mats @ worldgrid2worldcoord_mat).contiguous() @ feat_zoom_mats
                global_feat = kornia.warp_perspective(feat_map, cur_trans_mats, dsize=(int(self.feat_H/2**c_layer*scale), int(self.feat_W/2**c_layer*scale)))
                global_x.append(global_feat)
                
            global_x_multi_depth.append(global_x)
    
        if len(global_x_multi_depth) > 1:
            global_x = []
            for c_layer in range(len(x)):
                cur_feat_map = [global_x_multi_depth[i][c_layer].unsqueeze(1) for i in range(len(global_x_multi_depth))]    # (B,D,C,H,W)
                cur_feat_map = torch.cat(cur_feat_map, dim=1)
                cur_feat_map = torch.mean(cur_feat_map, dim=1)
                global_x.append(cur_feat_map)        
        else:
            global_x = global_x_multi_depth[0]
        
        #########################################################################
        #        Merge the feature of multi-agents (with bandwidth cost)        #
        #########################################################################
        if self.feat_mode == 'fused':
            global_x_fused = []
            for c_layer, feat_map in enumerate(x_early):
                # Mean
                # global_x[c_layer] = global_x[c_layer] + feat_map
                # global_x_fused.append(torch.cat([global_x[c_layer].unsqueeze(1), feat_map.unsqueeze(1)], dim=1).mean(dim=1))
                # Max
                global_x_fused.append(torch.cat([global_x[c_layer].unsqueeze(1), feat_map.unsqueeze(1)], dim=1).max(dim=1)[0])
        
        communication_rate = None
        results = None
        if self.trans_layer[-1] == -2:
            print('output the encoder')
            if self.feat_mode == 'fused':
                return global_x_fused, warp_images_list
            elif self.feat_mode in ['early', 'inter']:
                return global_x, warp_images_list
        else:
            if self.feat_mode in ['early', 'inter']:
                single_x = global_x
            elif self.feat_mode == 'fused':
                single_x = global_x_fused

            for c_layer, feat_map in enumerate(single_x):
                _, c, h, w = feat_map.shape
                feat_map = feat_map.view(b, num_agents, c, h, w)
                single_x[c_layer] = feat_map
            return single_x, warp_images_list

    
class decoder(nn.Module):
    def __init__(self, heads, channels, down_ratio,feat_mode, head_conv = 256, out_channel=0, last_level = 5, final_kernel = 1):
        super(decoder, self).__init__()
        self.heads = heads
        self.first_level = int(np.log2(down_ratio))
        self.last_level = last_level
        self.feat_mode = feat_mode
        if out_channel == 0:
            out_channel = channels[self.first_level]

        self.ida_up = IDAUp(out_channel, channels[self.first_level:self.last_level], 
                    [2 ** i for i in range(self.last_level - self.first_level)])
        print('decoder ida len: ', len(self.ida_up.state_dict()))
        for head in self.heads:
            classes = self.heads[head] #检测类别的个数
            if head_conv > 0: # 256 output channel
              fc = nn.Sequential(
                  nn.Conv2d(channels[self.first_level], head_conv,
                    kernel_size=3, padding=1, bias=True),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(head_conv, classes, 
                    kernel_size=final_kernel, stride=1, 
                    padding=final_kernel // 2, bias=True))
              if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19) #why 2.19？
              else:
                self.fill_fc_weights(fc) #constant bias = 0
            else: 
              fc = nn.Conv2d(channels[self.first_level], classes, 
                  kernel_size=final_kernel, stride=1, 
                  padding=final_kernel // 2, bias=True)
              if 'hm' in head:
                fc.bias.data.fill_(-2.19)
              else:
                self.fill_fc_weights(fc)
            self.__setattr__(head, fc)            

        #######load weights#########
        pretrain_dict = torch.load('/home/cmic2/ros_workspace/src/drone_detection/model/model_best.pth')
        model_dict = {}
        state_dict = self.ida_up.state_dict()
        heads_dict={}
        decoder_num = 0
        for head in self.heads:
            heads_dict[head] = {}
        for k, v in pretrain_dict["state_dict"].items():
            for name in state_dict.keys():
                if 'ida_up.' + name in k:
                    model_dict[name] = v
                    decoder_num +=1
            for head in self.heads:
                for name_head in self.__getattr__(head).state_dict().keys():
                    # print('heads: ', head +'.' + name_head)
                    if head + '.' + name_head in k:
                        heads_dict[head][name_head] = v
                        decoder_num +=1
        state_dict.update(model_dict)
        self.ida_up.load_state_dict(state_dict)
        for head in self.heads:
            head_model_dict = self.__getattr__(head).state_dict()
            head_model_dict.update(heads_dict[head])
            self.__getattr__(head).load_state_dict(head_model_dict)
            # print('len_{}: '.format(head),len(self.__getattr__(head).state_dict()))
        print('updated_decoder_num: ',decoder_num)
    def fill_fc_weights(self, layers):
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, global_x, round_id):
        communication_rate = None
        results = None

        if self.feat_mode in ['early', 'inter']:
            y = []
            for i in range(self.last_level - self.first_level):
                y.append(global_x[i].clone())
            self.ida_up(y, 0, len(y))
            print("decoder ida: ",len(self.ida_up.state_dict()))
        elif self.feat_mode == 'fused':
            y = []
            y_early = []
            y_fused = []
            for i in range(self.last_level - self.first_level):
                y.append(global_x[i].clone())
                y_early.append(global_x[i].clone())
                y_fused.append(global_x[i].clone())
            self.ida_up(y, 0, len(y))
            self.ida_up(y_early, 0, len(y_early))
            # self.ida_up_early(y_early, 0, len(y_early)
            self.ida_up(y_fused, 0, len(y_fused))

        _, _, h, w = y[-1].shape
        # trans_mats_inverse = torch.inverse(trans_mats_list[0]).view(b, num_agents, 3, 3).contiguous().view(b*num_agents, 3, 3).contiguous()
        global_z = {}
        z = {}
        for head in self.heads:
            global_z[head] = self.__getattr__(head)(y[-1]) # (b*num_agent, 2, 112, 200)
            print('len_{}: '.format(head),len(self.__getattr__(head).state_dict()))
            if self.feat_mode == 'fused':
                global_z[head+'_early'] = self.__getattr__(head)(y_early[-1])
                global_z[head+'_fused'] = self.__getattr__(head)(y_fused[-1])

        # Visualize weight mats (b, N, h, w)
        if results is not None:
            global_z.update(results)
        global_z['comm_rate'] = torch.zeros(1,1).to(global_x[0].device) if communication_rate is None else communication_rate
        return [global_z]
    
    # def process(self):
    #     s