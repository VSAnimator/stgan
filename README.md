# STGAN in Pytorch

This is a Pytorch implementation of the STGAN model from the paper "Cloud Removal in Satellite Images Using Spatiotemporal Generative Models," Sarukkai<sup>\*</sup>, Jain<sup>\*</sup>, Uzkent, and Ermon (https://arxiv.org/abs/1912.06838). This implementation was written by Vishnu Sarukkai (https://github.com/VSAnimator/) and Anirudh Jain. 

This implementation is built on a clone of the implementation of Pix2Pix from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix. The readme also is based on the readme from the CycleGAN/pix2pix repo. That repo contains the code corresponding to the following two papers:

Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks.<br>
[Jun-Yan Zhu](https://people.eecs.berkeley.edu/~junyanz/)\*,  [Taesung Park](https://taesung.me/)\*, [Phillip Isola](https://people.eecs.berkeley.edu/~isola/), [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros). In ICCV 2017. (* equal contributions) [[Bibtex]](https://junyanz.github.io/CycleGAN/CycleGAN.txt)


Image-to-Image Translation with Conditional Adversarial Networks.<br>
[Phillip Isola](https://people.eecs.berkeley.edu/~isola), [Jun-Yan Zhu](https://people.eecs.berkeley.edu/~junyanz), [Tinghui Zhou](https://people.eecs.berkeley.edu/~tinghuiz), [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros). In CVPR 2017. [[Bibtex]](http://people.csail.mit.edu/junyanz/projects/pix2pix/pix2pix.bib)

## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/VSAnimator/stgan
cd stgan
```

- Install [PyTorch](http://pytorch.org/) 0.4+ and other dependencies (e.g., torchvision, [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate)).
  - For pip users, please type the command `pip install -r requirements.txt`.

### STGAN dataset
Download the STGAN dataset from https://doi.org/10.7910/DVN/BSETKZ. This download includes two datasets (singleImage and multipleImage), described both in the paper and on the linked page. Before training a model, these images must be split into train/val/test splits--for instance, for both singleImage/clear and singleImage/cloudy, create three subfolders called "train","val","test", and assign the images from the main directory into the corresponding partitions either using a script that assigns splits at random or a split that keeps images from the same "tile" together (accounting for the possibility that these images are correlated). In our case, we used a 80-10-10 train-val-test split while keeping image crops from the same "tile" together. Make sure that corresponding clear and cloudy images are assigned to the same split.

Once the data is formatted this way, call:
```bash
python datasets/combine_A_and_B.py --fold_A /path/to/data/cloudy --fold_B /path/to/data/clear --fold_AB /path/to/data/combined
```

This will combine each pair of images (cloudy,clear) into a single image file, ready for training.

### STGAN training
- Train a model:
```bash
python train.py --dataroot ./path/to/data/combined  --name stgan --model temporal_branched_ir --netG unet_256_independent --input_nc 4
```
- Options include temporal_branched and temporal_branched_ir for "model" (depending on if we're using infrared data), 3 or 4 for "input_nc" (again depending on if we're using infrared data), unet_256_independent and independent_resnet_9blocks for "netG", and further options such as "batch_size", "preprocess", and "norm". See train.py for detained training options. 
- Our best models often opted for larger batch_size, norm=batch, and both options for netG worked well (with the resnet generator taking longer to train than the unet but sometimes providing improvements in accuracy).
- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097. 

- Test the model (`bash ./scripts/test_pix2pix.sh`):
```bash
python test.py --dataroot ./path/to/data/combined  --name stgan --model temporal_branched_ir --netG unet_256_independent --input_nc 4
```
- The test results will be saved to a html file here: `./results/stgan/test_latest/index.html`. 
