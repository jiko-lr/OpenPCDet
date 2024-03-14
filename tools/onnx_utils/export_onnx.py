import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets.custom.custom_dataset import CustomDataset
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.utils.common_utils import create_logger

# Load configuration file
config_file = "/home/iat/ws_jinhan/OpenPCDet/tools/cfgs/custom_models/pointpillars.yaml"
cfg_from_yaml_file(config_file, cfg)

# Create logger
logger= create_logger()

# Build the model
model = build_network(cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=CustomDataset)

# Load pre-trained weights
checkpoint_path = "/home/iat/ws_jinhan/OpenPCDet/output/cfgs/custom_models/pointpillars/default/ckpt/checkpoint_epoch_100.pth"
model.load_params_from_file(filename=checkpoint_path, logger=logger, to_cpu=True)

# Set the model to evaluation mode
model.eval()

# Dummy input data for exporting to ONNX
dummy_input = (torch.randn(1, 4, 1200, 1600).cuda(),)

torch.onnx.export(
    model,
    dummy_input,
    "/home/iat/ws_jinhan/OpenPCDet/tools/onnx_utils/output_model.onnx",
    do_constant_folding=True,
    verbose=True
)

print("ONNX export completed successfully.")
