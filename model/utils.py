import csv

from PIL import Image, ImageOps, ImageFilter
import os

from torch import nn
from torch.utils.data.dataset import Dataset
import random
import numpy as np
import  torch
from torch.nn import init
from datetime import datetime
import argparse
import shutil
from matplotlib import pyplot as plt

class TrainSetLoader(Dataset):

    """Iceberg Segmentation dataset."""
    NUM_CLASS = 1

    def __init__(self, dataset_dir, img_id ,base_size=512,crop_size=480,transform=None,suffix='.png'):
        super(TrainSetLoader, self).__init__()

        self.transform = transform
        self._items = img_id
        self.masks = dataset_dir+'/'+'masks'
        self.images = dataset_dir+'/'+'images'
        self.grads = dataset_dir + '/' + 'GradForeground'
        self.edges = dataset_dir + '/' + 'EdgeForeground'
        self.base_size = base_size
        self.crop_size = crop_size
        self.suffix = suffix

    def _sync_transform(self, img, mask, grad,edge):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            grad = grad.transpose(Image.FLIP_LEFT_RIGHT)
            edge = edge.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge)
        long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        grad = grad.resize((ow, oh), Image.NEAREST)
        edge = edge.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
            grad = ImageOps.expand(grad, border=(0, 0, padw, padh), fill=0)
            edge = ImageOps.expand(edge, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        grad = grad.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        edge = edge.crop((x1, y1, x1 + crop_size, y1 + crop_size))

        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        # final transform
        img, mask, grad,edge = np.array(img), np.array(mask, dtype=np.float32), np.array(grad, dtype=np.float32),np.array(edge, dtype=np.float32)
        return img, mask, grad,edge

    def __getitem__(self, idx):

        img_id     = self._items[idx]                        # idx：('../SIRST', 'Misc_70') 成对出现，因为我的workers设置为了2
        img_path   = self.images+'/'+img_id+self.suffix   # img_id的数值正好补了self._image_path在上面定义的2个空
        label_path = self.masks +'/'+img_id+self.suffix
        grad_path = self.grads +'/'+img_id+self.suffix
        edge_path = self.edges +'/'+img_id+self.suffix

        img = Image.open(img_path).convert('RGB')         ##由于输入的三通道、单通道图像都有，所以统一转成RGB的三通道，这也符合Unet等网络的期待尺寸
        mask = Image.open(label_path)
        grad = Image.open(grad_path)
        edge = Image.open(edge_path)

        # synchronized transform
        img, mask, grad,edge = self._sync_transform(img, mask, grad,edge)

        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        mask = np.expand_dims(mask, axis=0).astype('float32')/ 255.0
        grad = np.expand_dims(grad, axis=0).astype('float32')/ 255.0
        edge = np.expand_dims(edge, axis=0).astype('float32')/ 255.0


        return img, torch.from_numpy(mask), torch.from_numpy(grad), torch.from_numpy(edge) #img_id[-1]

    def __len__(self):
        return len(self._items)

class TestSetLoader(Dataset):
    """Iceberg Segmentation dataset."""
    NUM_CLASS = 1

    def __init__(self, dataset_dir, img_id,transform=None,base_size=512,crop_size=480,suffix='.png'):
        super(TestSetLoader, self).__init__()
        self.transform = transform
        self._items    = img_id
        self.masks     = dataset_dir+'/'+'masks'
        self.images    = dataset_dir+'/'+'images'
        self.grads=dataset_dir+'/'+'GradForeground'
        self.edges=dataset_dir+'/'+'EdgeForeground'
        self.base_size = base_size
        self.crop_size = crop_size
        self.suffix    = suffix

    def _testval_sync_transform(self, img, mask, grad,edge):
        base_size = self.base_size
        img  = img.resize ((base_size, base_size), Image.BILINEAR)
        mask = mask.resize((base_size, base_size), Image.NEAREST)
        grad = grad.resize((base_size, base_size), Image.NEAREST)
        edge = edge.resize((base_size, base_size), Image.NEAREST)
        # final transform
        img, mask, grad,edge = np.array(img), np.array(mask, dtype=np.float32), np.array(grad, dtype=np.float32),np.array(edge, dtype=np.float32)  # img: <class 'mxnet.ndarray.ndarray.NDArray'> (512, 512, 3)
        return img, mask, grad,edge

    def __getitem__(self, idx):
        # print('idx:',idx)
        img_id = self._items[idx]  # idx：('../SIRST', 'Misc_70') 成对出现，因为我的workers设置为了2
        img_path   = self.images+'/'+img_id+self.suffix    # img_id的数值正好补了self._image_path在上面定义的2个空
        label_path = self.masks +'/'+img_id+self.suffix
        grad_path = self.grads +'/'+img_id+self.suffix
        edge_path = self.edges +'/'+img_id+self.suffix
        img  = Image.open(img_path).convert('RGB')  ##由于输入的三通道、单通道图像都有，所以统一转成RGB的三通道，这也符合Unet等网络的期待尺寸
        mask = Image.open(label_path)
        grad = Image.open(grad_path)
        edge = Image.open(edge_path)
        # synchronized transform
        img, mask, grad,edge = self._testval_sync_transform(img, mask,grad,edge)

        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        mask = np.expand_dims(mask, axis=0).astype('float32') / 255.0
        grad = np.expand_dims(grad, axis=0).astype('float32') / 255.0
        edge = np.expand_dims(edge, axis=0).astype('float32') / 255.0

        return img, torch.from_numpy(mask), torch.from_numpy(grad),torch.from_numpy(edge)  # img_id[-1]

    def __len__(self):
        return len(self._items)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_ckpt(state, save_path, filename):
    torch.save(state, os.path.join(save_path,filename))

def save_train_log(args, save_dir):
    dict_args=vars(args)
    args_key=list(dict_args.keys())
    args_value = list(dict_args.values())
    with open('result/%s/train_log.txt'%save_dir ,'w') as  f:
        now = datetime.now()
        f.write("time:--")
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        f.write(dt_string)
        f.write('\n')
        for i in range(len(args_key)):
            f.write(args_key[i])
            f.write(':--')
            f.write(str(args_value[i]))
            f.write('\n')
    return

def save_model_and_result(dt_string, epoch,train_loss, test_loss, best_iou, best_niou,recall, precision, save_IoU_dir, save_other_metric_dir):

    with open(save_IoU_dir, 'a') as f:
        f.write('{} - {:04d}:\t - train_loss: {:04f}:\t - test_loss: {:04f}:\t mIoU {:.4f}\t nIoU {:.4f}\n' .format(dt_string, epoch,train_loss, test_loss, best_iou,best_niou))
    with open(save_other_metric_dir, 'a') as f:
        f.write(dt_string)
        f.write('-')
        f.write(str(epoch))
        f.write('\n')
        f.write('Recall-----:')
        for i in range(len(recall)):
            f.write('   ')
            f.write(str(round(recall[i], 8)))
            f.write('   ')
        f.write('\n')

        f.write('Precision--:')
        for i in range(len(precision)):
            f.write('   ')
            f.write(str(round(precision[i], 8)))
            f.write('   ')
        f.write('\n')

def save_model(mean_IOU, best_iou, n_IoU, save_dir, save_prefix, train_loss, test_loss, recall, precision, epoch, net):
    if mean_IOU > best_iou:
        save_IoU_dir = 'result/' + save_dir + '/' + save_prefix + '_best_IoU.log'
        save_other_metric_dir = 'result/' + save_dir + '/' + save_prefix + '_best_IoU_other_metric.log'
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        best_iou = mean_IOU
        best_niou = n_IoU
        save_model_and_result(dt_string, epoch, train_loss, test_loss, best_iou, best_niou,
                          recall, precision, save_IoU_dir, save_other_metric_dir)


def save_result_for_test(dataset_dir, st_model, epochs, best_iou, best_niou,recall, precision,fa,pd):
    with open(dataset_dir + '/' + 'value_result'+'/' + st_model +'_best_IoU.log', 'a') as f:
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        f.write('{} - {:04d}:\tmIoU: {:.4f}\n'.format(dt_string, epochs, best_iou))
        f.write('{} - {:04d}:\tnIoU: {:.4f}\n'.format(dt_string, epochs, best_niou))

    with open(dataset_dir + '/' +'value_result'+'/'+ st_model + '_best_other_metric.log', 'a') as f:
        f.write(dt_string)
        f.write('-')
        f.write(str(epochs))
        f.write('\n')
        f.write('Recall-----:')
        for i in range(len(recall)):
            f.write('   ')
            f.write(str(round(recall[i], 8)))
            f.write('   ')
        f.write('\n')

        f.write('Precision--:')
        for i in range(len(precision)):
            f.write('   ')
            f.write(str(round(precision[i], 8)))
            f.write('   ')
        f.write('\n')
    with open(dataset_dir+'/'+'value_result'+'/'+st_model+'_PD_FA.log','a') as f:
        f.write('\n FA: {} \n PD: {}'.format(fa,pd))

    return

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.xavier_normal(m.weight.data)

def make_dir(dataset, model):
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    save_dir = "%s_%s_%s_woDS" % (dataset, model, dt_string)
    os.makedirs('result/%s' % save_dir, exist_ok=True)
    return save_dir

def total_visulization_generation(dataset_dir, test_txt, suffix, target_image_path, target_dir,size):
    source_image_path = dataset_dir + '/images'

    txt_path = test_txt
    ids = []
    with open(txt_path, 'r') as f:
        ids += [line.strip() for line in f.readlines()]

    for i in range(len(ids)):
        source_image = source_image_path + '/' + ids[i] + suffix
        target_image = target_image_path + '/' + ids[i] + suffix
        shutil.copy(source_image, target_image)
    for i in range(len(ids)):
        source_image = target_image_path + '/' + ids[i] + suffix
        img = Image.open(source_image)
        img = img.resize((size, size), Image.ANTIALIAS)
        img.save(source_image)
    for m in range(len(ids)):
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 3, 1)
        img = plt.imread(target_image_path + '/' + ids[m] + suffix)
        plt.imshow(img, cmap='gray')
        plt.xlabel("Raw Imamge", size=11)

        plt.subplot(1, 3, 2)
        img = plt.imread(target_image_path + '/' + ids[m] + '_GT' + suffix)
        plt.imshow(img, cmap='gray')
        plt.xlabel("Ground Truth", size=11)

        plt.subplot(1, 3, 3)
        img = plt.imread(target_image_path + '/' + ids[m] + '_Pred' + suffix)
        plt.imshow(img, cmap='gray')
        plt.xlabel("Predicts", size=11)

        plt.savefig(target_dir + '/' + ids[m].split('.')[0] + "_fuse" + suffix, facecolor='w', edgecolor='red')

def make_visulization_dir(target_image_path, target_dir):
    if os.path.exists(target_image_path):
        shutil.rmtree(target_image_path)  # 删除目录，包括目录下的所有文件
    os.mkdir(target_image_path)

    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)  # 删除目录，包括目录下的所有文件
    os.mkdir(target_dir)

def save_Pred_GT(pred, labels, target_image_path, val_img_ids, num, suffix,size):

    predsss = np.array((pred > 0).cpu()).astype('int64') * 255
    predsss = np.uint8(predsss)
    labelsss = labels * 255
    labelsss = np.uint8(labelsss.cpu())

    img = Image.fromarray(predsss.reshape(size, size))
    img.save(target_image_path + '/' + '%s_Pred' % (val_img_ids[num]) +suffix)
    img = Image.fromarray(labelsss.reshape(size, size))
    img.save(target_image_path + '/' + '%s_GT' % (val_img_ids[num]) + suffix)


def save_Pred_GT_visulize(pred, img_demo_dir, img_demo_index, suffix,size):

    predsss = np.array((pred > 0).cpu()).astype('int64') * 255
    predsss = np.uint8(predsss)

    img = Image.fromarray(predsss.reshape(size, size))
    img.save(img_demo_dir + '/' + '%s_Pred' % (img_demo_index) +suffix)

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    img = plt.imread(img_demo_dir + '/' + img_demo_index + suffix)
    plt.imshow(img, cmap='gray')
    plt.xlabel("Raw Imamge", size=11)

    plt.subplot(1, 2, 2)
    img = plt.imread(img_demo_dir + '/' + '%s_Pred' % (img_demo_index) +suffix)
    plt.imshow(img, cmap='gray')
    plt.xlabel("Predicts", size=11)

    plt.savefig(img_demo_dir + '/' + img_demo_index + "_fuse" + suffix, facecolor='w', edgecolor='red')
    plt.show()


### compute model params
def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count

def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_dataset (root, dataset, split_method):
    train_txt = root + '/' + dataset + '/' + split_method + '/' + 'train.txt'
    test_txt  = root + '/' + dataset + '/' + split_method + '/' + 'test.txt'
    train_img_ids = []
    test_img_ids = []
    with open(train_txt, "r") as f:
        line = f.readline()
        while line:
            train_img_ids.append(line.split('\n')[0])
            line = f.readline()
        f.close()
    with open(test_txt, "r") as f:
        line = f.readline()
        while line:
            test_img_ids.append(line.split('\n')[0])
            line = f.readline()
        f.close()
    return train_img_ids,test_img_ids,test_txt

