import os

import cv2
import tqdm
import json

import torch
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.decode_helper import extract_dets_from_outputs
from lib.helpers.decode_helper import decode_detections
from lib.helpers.utils_helper import calc_iou
import time
import numpy as np
from utils import misc
import ast

# from huamnrobotinteraction import TkinterApp

def read_txt_as_int_list(file_path):
    with open(file_path, 'r') as file:
        int_list = [int(line.strip()) for line in file]
    return int_list

class Tester(object):
    def __init__(self, cfg, model, dataloader, logger,loss, train_cfg=None, model_name='LM2CNet'):
        self.cfg = cfg
        self.model = model
        self.dataloader = dataloader
        self.max_objs = dataloader.dataset.max_objs    # max objects per images, defined in dataset
        self.class_name = dataloader.dataset.class_name
        self.output_dir = os.path.join('./' + train_cfg['save_path'], model_name)
        self.dataset_type = cfg.get('type', 'LM2CNet')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        self.model_name = model_name
        self.mono3dvg_loss = loss

    def test(self):
        # test a checkpoint
        # checkpoint_path = os.path.join(self.output_dir, self.cfg['pretrain_model'])
        checkpoint_path = self.cfg['pretrain_model']

        assert os.path.exists(checkpoint_path)

        load_checkpoint(model=self.model,
                        optimizer=None,
                        filename=checkpoint_path,
                        map_location=self.device,
                        logger=self.logger)
        self.model.to(self.device)
        self.inference()

    def inference(self):
        root_path = os.getcwd()
        torch.set_grad_enabled(False)
        self.model.eval()
        iou_3dbox_test = {"test_redundant": [], "test_missing": [], "text_normal": [], "Orthers": [], "Overall": []}
        test_redundant = read_txt_as_int_list(root_path + '/LM2CNet/test_redundant.txt')
        test_missing = read_txt_as_int_list(root_path + '/LM2CNet/test_missing.txt')
        test_others = read_txt_as_int_list(root_path + '/LM2CNet/test_others.txt')
        results = {}
        progress_bar = tqdm.tqdm(total=len(self.dataloader), leave=True, desc='Evaluation Progress')
        model_infer_time = 0
        for batch_idx, (inputs, calibs, targets, info) in enumerate(self.dataloader):
            inputs = inputs.to(self.device)
            calibs = calibs.to(self.device)
            img_sizes = info['img_size'].to(self.device)
            gt_3dboxes = info['gt_3dbox']
            gt_3dboxes = [[float(gt_3dboxes[j][i].detach().cpu().numpy()) for j in range(len(gt_3dboxes)) ] for i in range(len(gt_3dboxes[0]))]


            captions = targets["text"]
            captions_index = torch.Tensor([x for x in range(len(captions))])
            captions_index = captions_index.to(self.device)

            ann_id = targets['anno_id']
            batch_size = inputs.shape[0]
            imageid = 1
            instanceid = 1
            start_time = time.time()
            outputs = self.model(inputs, calibs,  img_sizes, captions, imageid, instanceid, ann_id, captions_index)
            end_time = time.time()
            model_infer_time += end_time - start_time

            dets = extract_dets_from_outputs(outputs=outputs, K=self.max_objs, topk=self.cfg['topk'])

            dets = dets.detach().cpu().numpy()

            # get corresponding calibs & transform tensor to numpy
            calibs = [self.dataloader.dataset.get_calib(camer, key_frames) for camer, key_frames in zip(info['camer'], info['key_frames'])]
            info = {key: val.detach().cpu().numpy() for key, val in info.items() if key not in ['gt_3dbox', 'camer', 'key_frames']}
            cls_mean_size = self.dataloader.dataset.cls_mean_size
            dets, pred_3dboxes = decode_detections(
                dets=dets,
                info=info,
                calibs=calibs,
                cls_mean_size=cls_mean_size,)
            for i in range(batch_size):
                annid = info['anno_id'][i]
                pre_box3D_ori = np.array(pred_3dboxes[i])
                gt_box3D_ori = np.array(gt_3dboxes[i])
                pre_box3D = np.array([pre_box3D_ori[3], pre_box3D_ori[4], pre_box3D_ori[5], pre_box3D_ori[0],pre_box3D_ori[1], pre_box3D_ori[2]], dtype=np.float32)
                gt_box3D = np.array([gt_box3D_ori[3], gt_box3D_ori[4], gt_box3D_ori[5], gt_box3D_ori[0], gt_box3D_ori[1], gt_box3D_ori[2]],dtype=np.float32)

                gt_box3D[1] -= gt_box3D[3] / 2
                pre_box3D[1] -= pre_box3D[3] / 2  # real 3D center in 3D space
                IoU = calc_iou(pre_box3D, gt_box3D)
                iou_3dbox_test["Overall"].append(IoU)

                if annid in test_redundant:
                    iou_3dbox_test['test_redundant'].append(IoU)
                elif annid in test_missing:
                    iou_3dbox_test['test_missing'].append(IoU)
                elif annid in test_others:
                    iou_3dbox_test['Orthers'].append(IoU)
                else:
                    iou_3dbox_test['text_normal'].append(IoU)
            results.update(dets)
            progress_bar.update()
            if batch_idx % 30 == 0:
                acc5 = np.sum(np.array((np.array(iou_3dbox_test["Overall"]) > 0.5), dtype=float)) / len(iou_3dbox_test["Overall"])
                acc25 = np.sum(np.array((np.array(iou_3dbox_test["Overall"]) > 0.25), dtype=float)) / len(iou_3dbox_test["Overall"])
                miou = sum(iou_3dbox_test["Overall"]) / len(iou_3dbox_test["Overall"])

                print_str ='Epoch: [{}/{}]\t' \
                           'Accu25 {acc25:.2f}%\t' \
                           'Accu5 {acc5:.2f}%\t' \
                           'Mean_iou {miou:.2f}%\t' \
                    .format(
                    batch_idx, len(self.dataloader), \
                    acc25=acc25 * 100, acc5=acc5 * 100, miou=miou * 100
                )
                self.logger.info(print_str)


        progress_bar.close()
        #
        # save the result for evaluation.
        self.logger.info('==> LM2CNet Evaluation ...')
        a = 1
        for split_name, iou_3dbox in iou_3dbox_test.items():
            print("------------" + split_name + "------------")
            acc5 = np.sum(np.array((np.array(iou_3dbox) > 0.5), dtype=float)) / len(iou_3dbox)
            acc25 = np.sum(np.array((np.array(iou_3dbox) > 0.25), dtype=float)) / len(iou_3dbox)
            miou = sum(iou_3dbox) / len(iou_3dbox)
            print_str = 'Accu25 {acc25:.2f}%\t' \
                        'Accu5 {acc5:.2f}%\t' \
                        'Mean_iou {miou:.2f}%\t' \
                .format(
                acc25=acc25 * 100, acc5=acc5 * 100, miou=miou * 100
            )
            print(print_str)

    def prepare_targets(self, targets, batch_size):
        targets_list = []
        mask = targets['mask_2d']

        key_list = ['labels', 'boxes', 'calibs', 'depth', 'size_3d', 'heading_bin', 'heading_res', 'boxes_3d']
        for bz in range(batch_size):
            target_dict = {}
            for key, val in targets.items():
                if key in key_list:
                    target_dict[key] = val[bz][mask[bz]]
            targets_list.append(target_dict)
        return targets_list



