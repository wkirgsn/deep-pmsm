"""By Importing this file, a certain configured GPU will be written into an
environment variable that will be read by tensorflow to select a GPU."""

import preprocessing.config as cfg
import os

if cfg.keras_cfg['set_gpu_device']:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if cfg.keras_cfg['gpu_auto_set']:
        import GPUtil
        try:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUtil.getFirstAvailable()[0])
        except:
            pass
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.keras_cfg['gpu_device']
elif 'CUDA_VISIBLE_DEVICES' in os.environ:
    del os.environ['CUDA_VISIBLE_DEVICES']