import torch
from chexnet import ChexNet
from pathlib import Path


# convert prethesis model to current chexnet model
PATH = Path('/home/dattran/data/xray-thesis/chestX-ray14/models/')
for d in PATH.iterdir():
    # load pre-thesis modle
    try:
        chexnet = ChexNet(True, d.parts[-1])
        # save to theisis model
        torch.save(chexnet.state_dict(), d/'best.h5')
    except:
        continue
