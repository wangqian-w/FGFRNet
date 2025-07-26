# Basic module
import time

from tqdm import tqdm
from model.parse_args_test import  parse_args

# Torch and visulization
from torchvision import transforms
from torch.utils.data import DataLoader

from model.sample_generator import SampleGenerator
# Metric, loss .etc
from model.utils import *
from model.metric import *
from model.loss import *

# Model
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
        self.ROC = ROCMetric(1, args.ROC_thr)
        self.PD_FA = PD_FA(1,args.ROC_thr,args.crop_size)
        self.mIoU = mIoU(1)
        self.nIoU = nIoU(1)
        self.save_prefix = '_'.join([args.model, args.dataset])

        self.sample = SampleGenerator()
        self.sample.cuda()

        # Read image index from TXT
        if args.mode == 'TXT':
            dataset_dir = args.root + '/' + args.dataset
            train_img_ids, test_img_ids, test_txt=load_dataset(args.root, args.dataset,args.split_method)

        # Preprocess and load data
        input_transform = transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        testset = TestSetLoader (dataset_dir,img_id=test_img_ids,base_size=args.base_size, crop_size=args.crop_size, transform=input_transform,suffix=args.suffix)
        self.test_data = DataLoader(dataset=testset,  batch_size=args.test_batch_size, num_workers=args.workers,drop_last=False)

        # Choose and load model (this paper is finished by one GPU)
        if args.model == 'FGFRNet':
            model = FGFRNet()
        model = model.cuda()

        weights_init_xavier(model)
        print("Model Initializing")
        self.model = model

        # Initialize evaluation metrics
        self.best_recall    = [0,0,0,0,0,0,0,0,0,0,0]
        self.best_precision = [0,0,0,0,0,0,0,0,0,0,0]

        # Load trained model
        checkpoint = torch.load('result/' + args.model_dir)
        self.model.load_state_dict(checkpoint['state_dict'])

        # Test
        self.model.eval()
        tbar = tqdm(self.test_data)
        losses = AverageMeter()

        with torch.no_grad():
            num = 0
            for i, ( data, labels, grads,edges) in enumerate(tbar):
                data = data.cuda()
                labels = labels.cuda()
                grads=grads.cuda()
                edges = edges.cuda()

                output, x_glabel1, x_glabel2, x_glabel3, x_contras,edge1,edge2,edge3,edge4 = self.model(data,labels)

                _,F_fore, F_back, M_fore, M_back, T_fore, T_back = self.sample(x_contras, labels, data)

                loss = FinalLoss(output, labels, x_glabel1, x_glabel2, x_glabel3, grads,
                                 F_fore, F_back, M_fore, M_back, T_fore, T_back,
                                 edge1,edge2,edge3,edge4,edges)

                num += 1

                losses.update(loss.item(), output.size(0))
                self.ROC.update(output, labels)
                self.mIoU.update(output, labels)
                self.nIoU.update(output, labels)
                self.PD_FA.update(output, labels)

                _, m_IOU = self.mIoU.get()
                _, n_IOU = self.nIoU.get()

            true_positive_rate, false_positive_rate, recall, precision= self.ROC.get()

            FA, PD = self.PD_FA.get(len(test_img_ids))
            save_result_for_test(dataset_dir, args.st_model,args.epochs, m_IOU, n_IOU, recall, precision,FA,PD,true_positive_rate, false_positive_rate)

def main(args):
    trainer = Trainer(args)

if __name__ == "__main__":
    args = parse_args()
    main(args)