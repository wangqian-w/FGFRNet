from sklearn.model_selection import KFold
from thop import profile
from tqdm import tqdm
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms, models
from torch.utils.data import DataLoader, Subset

from model.parse_args_train import parse_args

# metric, loss .etc
from model.utils import *
from model.metric import *
from model.loss import *
from model.sample_generator import SampleGenerator

# model
from model.FGFRNet import FGFRNet

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

class Trainer(object):
    def __init__(self, args):
        # Initial
        self.args = args
        self.ROC = ROCMetric(1, 10)
        self.mIoU = mIoU(1)
        self.nIoU = nIoU(1)
        self.save_prefix = '_'.join([args.model, args.dataset])
        self.save_dir = args.save_dir

        self.sample = SampleGenerator()
        self.sample.cuda()
        # Read image index from TXT
        if args.mode == 'TXT':
            dataset_dir = args.root + '/' + args.dataset
            self.train_img_ids, self.test_img_ids, test_txt = load_dataset(args.root, args.dataset,
                                                                    args.split_method)

        # Preprocess and load data
        self.input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        trainset = TrainSetLoader(dataset_dir, img_id=self.train_img_ids, base_size=args.base_size, crop_size=args.crop_size,
                                  transform=self.input_transform, suffix=args.suffix)
        testset = TestSetLoader(dataset_dir, img_id=self.test_img_ids, base_size=args.base_size, crop_size=args.crop_size,
                                transform=self.input_transform, suffix=args.suffix)
        self.train_data = DataLoader(dataset=trainset, batch_size=args.train_batch_size, shuffle=True,
                                     num_workers=args.workers, drop_last=True)
        self.test_data = DataLoader(dataset=testset, batch_size=args.test_batch_size, num_workers=args.workers,
                                    drop_last=False)
        self.dataset_dir = dataset_dir

        if args.model == 'FGFRNet':
            model = FGFRNet()

        model = model.cuda()

        weights_init_xavier(model)
        print("Model Initializing")
        self.model = model

        self.optimizer = optim.AdamW([
            {'params': self.model.parameters()},
            {'params': self.sample.parameters(), 'lr': args.lr * 0.1}
        ], lr=args.lr, weight_decay=1e-4)
        self.scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=20, T_mult=1, eta_min=1e-6)

        self.scheduler.step()

        # Evaluation metrics
        self.best_iou = 0
        self.best_niou = 0
        self.best_recall = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.best_precision = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        self.epoch_train_loss = []
        self.epoch_test_loss = []
        self.epoch_mIoU = []
        self.epoch_nIoU = []

    def training(self, epoch):

        tbar = tqdm(self.train_data)
        self.model.train()
        losses = AverageMeter()
        for i, (data, labels, grads,edges) in enumerate(tbar):
            data = data.cuda()
            labels = labels.cuda()
            grads = grads.cuda()
            edges = edges.cuda()

            output, x_glabel1, x_glabel2, x_glabel3, x_contras,edge1,edge2,edge3,edge4 = self.model(data,labels)

            _,F_fore, F_back, M_fore, M_back, T_fore, T_back = self.sample(x_contras, labels, data)

            loss = FinalLoss(output, labels, x_glabel1, x_glabel2, x_glabel3, grads,
                             F_fore, F_back, M_fore, M_back, T_fore, T_back,
                             edge1,edge2,edge3,edge4,edges)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.update(loss.item(), output.size(0))
            tbar.set_description('Epoch %d, training loss %.4f' % (epoch+1, losses.avg))
        self.train_loss = losses.avg
        self.epoch_train_loss.append(self.train_loss)

    # Testing
    def testing(self, epoch):
        tbar = tqdm(self.test_data)
        self.model.eval()
        self.mIoU.reset()
        self.nIoU.reset()
        losses = AverageMeter()

        with torch.no_grad():
            for i, (data, labels, grads,edges) in enumerate(tbar):
                data = data.cuda()
                labels = labels.cuda()
                grads = grads.cuda()
                edges=edges.cuda()

                output, x_glabel1, x_glabel2, x_glabel3, x_contras,edge1,edge2,edge3,edge4 = self.model(data,labels)

                _,F_fore, F_back, M_fore, M_back, T_fore, T_back = self.sample(x_contras, labels, data)

                loss = FinalLoss(output, labels, x_glabel1, x_glabel2, x_glabel3, grads,
                                 F_fore, F_back, M_fore, M_back, T_fore, T_back,
                                 edge1,edge2,edge3,edge4,edges)

                losses.update(loss.item(), output.size(0))
                self.ROC.update(output, labels)
                self.mIoU.update(output, labels)
                self.nIoU.update(output, labels)
                ture_positive_rate, false_positive_rate, recall, precision = self.ROC.get()
                _, m_IOU = self.mIoU.get()
                _, n_IOU = self.nIoU.get()
                tbar.set_description(
                    'Epoch %d, test loss %.4f, mean_IoU: %.4f, n_IoU: %.4f' % (epoch+1, losses.avg, m_IOU, n_IOU))
            test_loss = losses.avg

        # save high-performance model
        save_model(m_IOU, self.best_iou, n_IOU, self.save_dir, self.save_prefix,
                   self.train_loss, test_loss, recall, precision, epoch+1, self.model.state_dict())

        if m_IOU > self.best_iou:
            self.best_iou = m_IOU
            save_ckpt({
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'loss': test_loss,
                'mean_IOU': m_IOU,
                'n_IoU': n_IOU
            }, save_path='result/' + self.save_dir,
                filename=self.save_prefix + '_best_miou.pth.tar')

        self.epoch_test_loss.append(test_loss)
        self.epoch_mIoU.append(m_IOU)
        self.epoch_nIoU.append(n_IOU)

def main(args):
    trainer = Trainer(args)
    for epoch in range(args.start_epoch, args.epochs):
        trainer.training(epoch)
        trainer.testing(epoch)

if __name__ == "__main__":

    args = parse_args()
    main(args)

