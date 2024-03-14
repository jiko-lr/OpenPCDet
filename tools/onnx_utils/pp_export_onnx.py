import argparse
import glob
from pathlib import Path


import numpy as np
import torch
import torch.onnx

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils




def main():

    config_path = '/home/iat/ws_jinhan/OpenPCDet/tools/cfgs/custom_models/pointpillars.yaml'
    ckpt_path = '/home/iat/ws_jinhan/OpenPCDet/output/cfgs/custom_models/pointpillars/default/ckpt/checkpoint_epoch_100.pth'


    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    

    dummy_input = torch.randn(1, 4, 12000)
    onnx_path = 'pointpillar.onnx'
    torch.onnx.export(model, dummy_input, onnx_path, verbose=True, input_names=['input'], output_names=['output'])

    print(f'Model successfully exported to {onnx_path}')



if __name__ == '__main__':
    main()
