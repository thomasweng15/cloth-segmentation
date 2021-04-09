import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import sklearn.metrics
from matplotlib import pyplot as plt

def normalize(img_depth):
    min_I = img_depth.min()
    max_I = img_depth.max()
    img_depth[img_depth<=min_I] = min_I
    img_depth = (img_depth - min_I) / (max_I - min_I)
    return img_depth

def compute_map(gt, pred, n_class, average=None):
    """
    Compute the multi-label classification accuracy.
    Args:
        gt (np.ndarray): Shape Nxn_class, 0 or 1, 1 if the object i is present in that
            image.
        pred (np.ndarray): Shape Nxn_class, probability of that object in the image
            (output probablitiy).
    Returns:
        MAP (scalar): average precision for all classes
    """
    gt = gt.reshape(-1, n_class)
    pred = pred.reshape(-1, n_class)
    AP = []
    for cid in range(n_class):
        gt_cls = gt[:, cid].astype('float32')
        pred_cls = pred[:, cid].astype('float32')
        # As per PhilK. code:
        # https://github.com/philkr/voc-classification/blob/master/src/train_cls.py
        pred_cls -= 1e-5 * gt_cls
        ap = sklearn.metrics.average_precision_score(gt_cls, pred_cls, average=average)
        AP.append(ap)
    return AP

def compute_iou(gt, pred, n_class):
    '''
    gt, pred -- h * w * n_class
    '''
    thres = 0.6
    ious = []
    for cid in range(n_class):
        pred_inds = pred[:, :, cid].astype('float32') > thres
        target_inds = gt[:, :, cid].astype('bool')
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / max(union, 1))
        # print("cls", cls, pred_inds.sum(), target_inds.sum(), intersection, float(intersection) / max(union, 1))
    return ious

def compute_auc(gt, pred, n_class):
    gt = gt.reshape(-1, n_class)
    pred = pred.reshape(-1, n_class)
    AUC = []
    for cid in range(n_class):
        gt_cls = gt[:, cid].astype('float32')
        pred_cls = pred[:, cid].astype('float32')

        fpr, tpr, thresholds = sklearn.metrics.roc_curve(gt_cls, pred_cls)
        AUC.append(sklearn.metrics.auc(fpr, tpr))
        # AUC.append(sklearn.metrics.roc_auc_score(gt_cls, pred_cls))
    return AUC

def preprocessHeatMap(hm, cmap = plt.get_cmap('jet')):
    hm = torch.Tensor(hm)
    hm = np.uint8(cmap(np.array(hm)) * 255).transpose(2, 0, 1)
    return hm

def print_gradients():
    for m in model.modules():
        if isinstance(m,nn.Conv2d) or isinstance(m,nn.ConvTranspose2d):
            print(m.weight.grad)
            print(m.bias.grad)
        elif isinstance(m,nn.Linear):
            print(m.weight.grad)
            print(m.bias.grad)
            # print(l.weight.grad)

def weights_init(model):
    for m in model.modules():
        if isinstance(m,nn.Conv2d) or isinstance(m,nn.ConvTranspose2d):
            nn.init.xavier_normal_(m.weight.data)
            # nn.init.kaiming_normal_(m.weight.data)
            m.bias.data.fill_(0)
        elif isinstance(m,nn.Linear):
            m.weight.data.normal_()

class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(unetConv2, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, 3, 1, 1), nn.BatchNorm2d(out_size), nn.ReLU()
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_size, out_size, 3, 1, 1), nn.BatchNorm2d(out_size), nn.ReLU()
            )
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1), nn.ReLU())
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1), nn.ReLU())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size, out_size, True)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
    
    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)

        offset = - outputs2.size()[3] + inputs1.size()[3]
        offset2 = - outputs2.size()[2] + inputs1.size()[2]
        if offset % 2:
            if offset2 % 2:
                padding = [offset // 2, offset // 2 + 1, offset2 // 2, offset2 // 2 + 1]
            else:
                padding = [offset // 2, offset // 2 + 1, offset2 // 2, offset2 // 2]
        else:
            if offset2 % 2:
                padding = [offset // 2, offset // 2, offset2 // 2, offset2 // 2 + 1]
            else:
                padding = [offset // 2, offset // 2, offset2 // 2, offset2 // 2 ]

        outputs2 = F.pad(outputs2, padding)
        return self.conv(torch.cat([inputs1, outputs2], 1))
