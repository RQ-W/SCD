import warnings
import scipy
import torch
import torch.nn as nn
import math
# from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import numpy as np


def featuremap_2_heatmap2(feature_map):
    assert isinstance(feature_map, torch.Tensor)
    feature_map = feature_map.detach()
    # heatmap = feature_map[:,0,:,:]*0    #
    heatmap = feature_map[:1,:,:,:].squeeze(0)
    heatmap1 = np.max(heatmap.cpu().numpy(), axis=0)
    hmax = np.max(heatmap1)
    hmin = np.min(heatmap1)
    heatmap2 = (heatmap1 - hmin) / (hmax - hmin + 1e-6)

    #heatmaps.append(heatmap)

    return heatmap2


def draw_feature_map(features, _i=0, save_dir = 'feature_map',name = None, ):
    img_path = r'D:\\mmcv\\mmdetection-2.24.1\\dota_data\\dota\\val\\P0217__1__0___0.jpg'
    # img = Image.open(img_path).convert('RGB')
    img = cv2.imread(img_path)  # 读取文件路径

    i=0
    if isinstance(features, torch.Tensor):  # 如果是单层
        features = [features]  # 转为列表
    for heat_maps in features:
    #     heat_maps=features.squeeze(0)
    #     heatmap = torch.mean(heat_maps, dim=0)
    #     heatmap = heatmap.cpu().numpy()
    #     heatmap = np.maximum(heatmap, 0)
    #     heatmap /= np.max(heatmap)  # minmax归一化处理
    #     heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))  # 变换heatmap图像尺寸,使之与原图匹配,方便后续可视化
    #     heatmap = np.uint8(255 * heatmap)  # 像素值缩放至(0,255)之间,uint8类型,这也是前面需要做归一化的原因,否则像素值会溢出255(也就是8位颜色通道)
    #     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HSV)  # 颜色变换
    #     plt.imshow(heatmap)
    #     plt.show()
    #     superimposed_img = heatmap * 0.4 + np.array(img)[:,:,::-1]
    #
    #     cv2.imwrite('superimg.jpg', superimposed_img)  # 保存结果
    #     # 可视化叠加至源图像的结果
    #     img_ = np.array(Image.open('superimg.jpg').convert('RGB'))
    #     plt.imshow(img_)
    #     plt.show()
    #     heat_maps = norm(heat_maps)
        heatmaps = featuremap_2_heatmap2(heat_maps)



        # heatmaps = cv2.resize(heatmaps, (img.shape[1], img.shape[0]))
        # scipy.io.savemat('D:\\mmcv\\mmdetection-2.24.1\\feature_visualization\\fpn_feature_'+str(i)+'.mat',
        #              {'feature': heatmaps})

        heatmap0 = np.uint8(255 * heatmaps)
        # img = np.uint8(255 * img)
        # 下面这行将热力图转换为RGB格式 ，如果注释掉就是灰度图
        heatmap = cv2.applyColorMap(heatmap0, cv2.COLORMAP_HSV)

        # heatmap = heatmap / 255.
        # superimposed_img = heatmap
        #img = cv2.resize(img, (heatmap.shape[1], heatmap.shape[0]))
        plt.imshow(heatmap, cmap="hsv")
        # # plt.colorbar()
        plt.show()
        # cv2.imshow("image", heatmap)
        # cv2.waitKey(0)
        # superimposed_img = heatmap * 0.5 + img * 0.5
        # superimposed_img = superimposed_img.astype("uint8")
        # # cv2.imshow("image1", superimposed_img)
        #
        # plt.imshow(superimposed_img, cmap="hsv")
        # # plt.savefig(
        # # 'D:\\mmcv\\mmdetection-2.24.1\\feature_visualization\\' + "_INTER_LINEAR" + os.path.join(str(i) + '.jpg'),
        # # dpi=600)
        # plt.show()
        #
        # #cv2.imwrite('D:\\mmcv\\mmdetection-2.24.1\\feature_visualization\\'+"_INTER_LANCZOS4"+os.path.join(str(i) + '.jpg'), superimposed_img)
        # i += 1




class trident_block(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, padding=[1, 2, 3], dilate=[1, 2, 3]):
        super(trident_block, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilate = dilate
        self.downsample = downsample
        self.share_weight4conv1 = nn.Parameter(torch.randn(planes, inplanes, 1, 1))
        self.share_weight4conv2 = nn.Parameter(torch.randn(planes, planes, 3, 3))
        self.share_weight4conv3 = nn.Parameter(torch.randn(planes * self.expansion, planes, 1, 1))#1*1/64, 3*3/64, 1*1/256

        self.bn11 = nn.BatchNorm2d(planes)#bn层
        self.bn12 = nn.BatchNorm2d(planes)
        self.bn13 = nn.BatchNorm2d(planes * self.expansion)

        self.bn21 = nn.BatchNorm2d(planes)
        self.bn22 = nn.BatchNorm2d(planes)
        self.bn23 = nn.BatchNorm2d(planes * self.expansion)

        self.bn31 = nn.BatchNorm2d(planes)
        self.bn32 = nn.BatchNorm2d(planes)
        self.bn33 = nn.BatchNorm2d(planes * self.expansion)

        self.relu1 = nn.ReLU(inplace=True)#relu层
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward_for_small(self, x):
        residual = x

        out = nn.functional.conv2d(x, self.share_weight4conv1, bias=None)
        out = self.bn11(out)
        out = self.relu1(out)

        out = nn.functional.conv2d(out, self.share_weight4conv2, bias=None, stride=self.stride, padding=self.padding[0], dilation=self.dilate[0])

        out = self.bn12(out)
        out = self.relu1(out)

        out = nn.functional.conv2d(out, self.share_weight4conv3, bias=None)
        out = self.bn13(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu1(out)

        return out

    def forward_for_middle(self, x):
        residual = x

        out = nn.functional.conv2d(x, self.share_weight4conv1, bias=None)
        out = self.bn21(out)
        out = self.relu2(out)

        out = nn.functional.conv2d(out, self.share_weight4conv2, bias=None, stride=self.stride, padding=self.padding[1],dilation=self.dilate[1])

        out = self.bn22(out)
        out = self.relu2(out)

        out = nn.functional.conv2d(out, self.share_weight4conv3, bias=None)
        out = self.bn23(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        # print(out.shape)
        # print(residual.shape)

        out += residual
        out = self.relu2(out)

        return out

    def forward_for_big(self, x):
        residual = x

        out = nn.functional.conv2d(x, self.share_weight4conv1, bias=None)
        out = self.bn31(out)
        out = self.relu3(out)

        out = nn.functional.conv2d(out, self.share_weight4conv2, bias=None, stride=self.stride, padding=self.padding[2], dilation=self.dilate[2])

        out = self.bn32(out)
        out = self.relu3(out)
        out = nn.functional.conv2d(out, self.share_weight4conv3, bias=None)#对输入平面实施2D卷积
        out = self.bn33(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu3(out)

        return out

    def forward(self, x):
        xm=x
        base_feat=[]#重新定义数组


        if self.downsample is not None:#衔接段需要downsample
            x1 = self.forward_for_small(x)
            base_feat.append(x1)
            x2 = self.forward_for_middle(x)
            base_feat.append(x2)
            x3 = self.forward_for_big(x)
            base_feat.append(x3)
        else:
            x1 = self.forward_for_small(xm[0])
            base_feat.append(x1)
            x2 = self.forward_for_middle(xm[1])
            base_feat.append(x2)
            x3 = self.forward_for_big(xm[2])
            base_feat.append(x3)
        return base_feat #三个分支



def trident_net(block1, inplanes, planes, blocks, stride=1):

    downsample = None
    if stride != 1 or inplanes != planes * block1.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block1.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block1.expansion),  # shortcut用1*1卷积
        )

    layers = []
    layers.append(block1(inplanes, planes, stride, downsample))  # 衔接段会出现通道不匹配，需要借助downsample
    inplanes = planes * block1.expansion  # 维度保持一致
    for i in range(1, blocks):
        layers.append(block1(inplanes, planes))  # 堆叠的block

    return nn.Sequential(*layers)  # 一个trident-block卷积





def CrossEntropy(outputs, targets):
    log_softmax_outputs = F.log_softmax(outputs, dim=1)
    softmax_targets = F.softmax(targets, dim=1)
    return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()


def dist2(tensor_a, tensor_b, attention_mask=None, channel_attention_mask=None):
    diff = (tensor_a - tensor_b) ** 2
    #   print(diff.size())      batchsize x 1 x W x H,
    #   print(attention_mask.size()) batchsize x 1 x W x H
    if attention_mask is not None:
        diff = diff * attention_mask
    if channel_attention_mask is not None:
        diff = diff * channel_attention_mask
    diff = torch.sum(diff) ** 0.5
    return diff


def plot_attention_mask(mask):
    mask = torch.squeeze(mask, dim=0)
    mask = mask.cpu().detach().numpy()
    plt.imshow(mask)
    plt.plot(mask)
    plt.savefig('1.png')
    print('saved')
    input()


@DETECTORS.register_module()
class TwoStageDetector(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(TwoStageDetector, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)

        self.tea_tri = nn.ModuleList()
        self.tea_tri.append(trident_net(trident_block, 64, 16, 6, stride=2))

        self.stu_tri= nn.ModuleList()
        self.stu_tri.append(trident_net(trident_block, 64, 16, 6, stride=2))

        self.adaptation_layers = nn.ModuleList([
            nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
        ])


        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            roi_head.pretrained = pretrained
            self.roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    # def init_weights(self, pretrained=None):
    #     """Initialize the weights in detector.
    #
    #     Args:
    #         pretrained (str, optional): Path to pre-trained weights.
    #             Defaults to None.
    #     """
    #     super(TwoStageDetector, self).init_weights(pretrained)
    #     self.backbone.init_weights(pretrained=pretrained)
    #     if self.with_neck:
    #         if isinstance(self.neck, nn.Sequential):
    #             for m in self.neck:
    #                 m.init_weights()
    #         else:
    #             self.neck.init_weights()
    #     if self.with_rpn:
    #         self.rpn_head.init_weights()
    #     if self.with_roi_head:
    #         self.roi_head.init_weights(pretrained)

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        bb = self.backbone(img)
        if self.with_neck:
            x = self.neck(bb)

        # stu_feat = self.stu_tri[0](bb[len(bb) - 1])
        # for i in range(len(stu_feat)):
        #     draw_feature_map(stu_feat[i])


        return x, bb

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        outs = ()
        # backbone
        x, bb = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs,)
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs,)
        return outs

    def get_teacher_info(self,
                         img,
                         img_metas,
                         gt_bboxes,
                         gt_labels,
                         gt_bboxes_ignore=None,
                         gt_masks=None,
                         proposals=None,
                         t_feats=None,
                         **kwargs):
        teacher_info = {}
        x, bb = self.extract_feat(img)
        tea_bbox_outs = self.rpn_head(x)
        teacher_info.update({'feat': x, "bb": bb, "tea_bbox_outs": tea_bbox_outs})
        # RPN forward and loss
        '''
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list, rpn_outs = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            teacher_info.update({'proposal_list': proposal_list})
            #   teacher_info.update({'rpn_out': rpn_outs})
        else:
            proposal_list = proposals
        '''
        '''
        roi_losses, roi_out = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                          gt_bboxes, gt_labels,
                                                          gt_bboxes_ignore, gt_masks, get_out=True,
                                                          **kwargs)
        teacher_info.update(
            cls_score=roi_out['cls_score'],
            pos_index=roi_out['pos_index'],
            bbox_pred=roi_out['bbox_pred'],
            labels=roi_out['labels'],
            bbox_feats=roi_out['bbox_feats'],
            x_cls=roi_out['x_cls'],
            x_reg=roi_out['x_reg']
        )
        '''
        return teacher_info

    def with_student_proposal(self,
                              img,
                              img_metas,
                              gt_bboxes,
                              gt_labels,
                              gt_bboxes_ignore=None,
                              gt_masks=None,
                              proposals=None,
                              s_info=None,
                              t_info=None,
                              **kwargs):

        with torch.no_grad():
            _, t_roi_out = self.roi_head.forward_train(t_info['feat'], img_metas, s_info['proposal_list'],
                                                       gt_bboxes, gt_labels,
                                                       gt_bboxes_ignore, gt_masks, get_out=True,
                                                       **kwargs)

        t_cls, s_cls, pos_index, labels = t_roi_out['cls_score'], s_info['cls_score'], t_roi_out[
            'pos_index'], t_roi_out['labels']
        t_cls_pos, s_cls_pos, labels_pos = t_cls[pos_index.type(torch.bool)], s_cls[pos_index.type(torch.bool)], labels[
            pos_index.type(torch.bool)]
        teacher_prediction = torch.max(t_cls_pos, dim=1)[1]
        correct_index = (teacher_prediction == labels_pos).detach()
        t_cls_pos_correct, s_cls_pos_correct = t_cls_pos[correct_index], s_cls_pos[correct_index]
        kd_pos_cls_loss = CrossEntropy(s_cls_pos_correct, t_cls_pos_correct) * 0.005
        kd_loss = dict(kd_pos_cls_loss=kd_pos_cls_loss)
        return kd_loss

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      t_info=None,
                      epoch=None,
                      iter=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """


        x, bb = self.extract_feat(img)
        tea_bbox_outs = t_info['tea_bbox_outs']
        stu_bbox_outs = self.rpn_head(x)
        losses = dict()
        kd_feat_loss = 0
        loss_reg = 0
        loss_cls = 0
        c_t = 0.1
        c_s_ratio = 1.0
        if t_info is not None:
            t_feats = t_info['bb']
            tea_feat = self.tea_tri[0](self.adaptation_layers[0](t_feats[0]))
            stu_feat = self.stu_tri[0](bb[0])
            # for i in range(len(tea_feat)):
            #     tea_feat[i] = torch.flatten(tea_feat[i], start_dim=1)
            #     stu_feat[i] = torch.flatten(stu_feat[i], start_dim=1)


            t_fea = torch.cat((tea_feat[0], tea_feat[1], tea_feat[2]), 1)
            s_fea = torch.cat((stu_feat[0], stu_feat[1], stu_feat[2]), 1)

            c_t_attention_mask = torch.mean(torch.abs(t_fea), [2, 3], keepdim=True)  # 2 x 256 x 1 x1
            c_size = c_t_attention_mask.size()
            c_t_attention_mask = c_t_attention_mask.view(x[0].size(0), -1)  # 2 x 256
            c_t_attention_mask = torch.softmax(c_t_attention_mask / c_t, dim=1) * t_fea.shape[1]
            c_t_attention_mask = c_t_attention_mask.view(c_size)  # 2 x 256 -> 2 x 256 x 1 x 1

            c_s_attention_mask = torch.mean(torch.abs(s_fea), [2, 3], keepdim=True)  # 2 x 256 x 1 x1
            c_size = c_s_attention_mask.size()
            c_s_attention_mask = c_s_attention_mask.view(x[0].size(0), -1)  # 2 x 256
            c_s_attention_mask = torch.softmax(c_s_attention_mask / c_t, dim=1) * s_fea.shape[1]
            c_s_attention_mask = c_s_attention_mask.view(c_size)  # 2 x 256 -> 2 x 256 x 1 x 1

            c_sum_attention_mask = (c_t_attention_mask + c_s_attention_mask * c_s_ratio) / (1 + c_s_ratio)
            c_sum_attention_mask = c_sum_attention_mask.detach()

            kd_feat_loss = dist2(t_fea, s_fea,
                                  channel_attention_mask=c_sum_attention_mask) * 7e-5 * 10

            # stu_cls_score = stu_bbox_outs[0]
            # tea_cls_score = tea_bbox_outs[0]
            # stu_reg_score = stu_bbox_outs[1]
            # tea_reg_score = tea_bbox_outs[1]
            loss_reg, loss_cls = self.rpn_head.reg_two_stage_SRKD_distill(stu_reg=stu_reg_score, tea_reg=tea_reg_score, tea_cls=tea_cls_score,
                                                 stu_cls=stu_cls_score, gt_truth=gt_bboxes, img_metas=img_metas)


        losses.update({'distill_reg_loss': loss_reg * 0.1})
        losses.update({'distill_cls_loss': loss_cls * 0.8})

        losses.update({'kd_feat_loss': kd_feat_loss})
        # print(kd_nonlocal_loss* 7e-5)






        
        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)

            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
            )
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                          gt_bboxes, gt_labels,
                                                          gt_bboxes_ignore, gt_masks,
                                                          **kwargs)
        
        losses.update(roi_losses)
        return losses

    '''
    s_info.update(proposal_list=proposal_list, cls_score=roi_out['cls_score'])
    _, student_with_teacher_proposal_outs = self.roi_head.forward_train(x, img_metas, t_info['proposal_list'],
                                                                        gt_bboxes, gt_labels,
                                                                        gt_bboxes_ignore, gt_masks, get_out=True,
                                                                        **kwargs)
    pos_index, s_cls, s_reg, t_cls, t_reg = t_info['pos_index'], student_with_teacher_proposal_outs['x_cls'], student_with_teacher_proposal_outs['x_reg'], t_info['x_cls'], t_info['x_reg']
    kd_feat_reg_loss = torch.dist(self.reg_adaptation(s_reg[pos_index]), t_reg[pos_index]) * 1e-4
    kd_feat_cls_loss = torch.dist(self.cls_adaptation(s_cls), t_cls) * 1e-4
    losses.update(kd_feat_reg_loss=kd_feat_reg_loss, kd_feat_cls_loss=kd_feat_cls_loss)
    '''

    '''
    #   distill positive objects
    t_feat, s_feat, pos_index = t_info['bbox_feats'], student_with_teacher_proposal_outs['bbox_feats'], t_info['pos_index']
    t_feat_pos, s_feat_pos = t_feat[pos_index], s_feat[pos_index]
    kd_bbox_feat_loss = torch.dist(t_feat_pos, self.bbox_feat_adaptation(s_feat_pos), p=2) * 1e-4
    t_feat_pos_flat, s_feat_pos_flat = torch.flatten(t_feat_pos, start_dim=1), torch.flatten(s_feat_pos, start_dim=1)
    t_feat_pos_relation = F.normalize(torch.mm(t_feat_pos_flat, t_feat_pos_flat.t()), p=2)
    s_feat_pos_relation = F.normalize(torch.mm(s_feat_pos_flat, s_feat_pos_flat.t()), p=2)
    kd_bbox_feat_relation_loss = torch.dist(s_feat_pos_relation, t_feat_pos_relation, p=2) * 0.01
    losses.update(kd_bbox_feat_relation_loss=kd_bbox_feat_relation_loss)
    losses.update(kd_bbox_feat_loss=kd_bbox_feat_loss)
    '''

    '''
    t_cls, s_cls, pos_index, labels = t_info['cls_score'], student_with_teacher_proposal_outs['cls_score'], t_info[
        'pos_index'], student_with_teacher_proposal_outs['labels']
    t_cls_pos, s_cls_pos, labels_pos = t_cls[pos_index.type(torch.bool)], s_cls[pos_index.type(torch.bool)], labels[
        pos_index.type(torch.bool)]
    t_prediction = torch.max(t_cls_pos, dim=1)[1]
    correct_index = t_prediction == labels_pos
    t_cls_pos_correct, s_cls_pos_correct = t_cls_pos[correct_index], s_cls_pos[correct_index]
    kd_pos_correct_cls_loss = CrossEntropy(s_cls_pos_correct, t_cls_pos_correct) * 0.05
    losses.update(kd_cls_teacher_loss=kd_pos_correct_cls_loss)
    '''
    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x, bb = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        x, bb = self.extract_feat(img)

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)
    


    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        x, bb = self.extract_feat(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)
