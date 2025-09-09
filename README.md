<!-- TOC start (generated with https://github.com/derlin/bitdowntoc) -->

- [What is MMLPhoto-z?](#what-is-mmlphoto-z)
- [How to install the dependencies?](#how-to-install-the-dependencies)
- [How to run the code?](#how-to-run-the-code)
- [Quick start of the structure of the code](#quick-start-of-the-structure-of-the-code)
- [Csv files enumerate ](#csv-files-enumerate)
- [Image cropped details and preprocessing](#image-cropped-details-and-preprocessing)

<!-- TOC end -->

<!-- TOC --><a name="what-is-mmlphoto-z"></a>

# What is MMLPhoto-z?

MMLPhoto-z is a cross-modal contrastive learning approach for estimating photo-z of quasars. This method employs
adversarial training and contrastive loss functions to promote the mutual conversion between multi-band photometric data
features (magnitude, color) and photometric image features, while extracting modality-invariant features.

<!-- TOC --><a name="how-to-install-the-dependencies"></a>

# How to install the dependencies?

You can install the python dependencies by running the code: `pip install -r requirments.txt`

<!-- TOC --><a name="how-to-run-the-code"></a>

# How to run the code?

You can run the code `python train.py --mode=WISE --task=ESTIMATION --modal=all` to train the model.

1. mode: SDSS/WISE/SKYMAPPER

   **choose what kind of data to train the model**

   <font color='red'> Note: For selecting datasets of different ranges within the same survey type, it is necessary to modify the corresponding CSV file reading accordingly.</font>


2. task: ESTIMATION/CLASSIFICATION

   **choose what kind of downstream task that the model should work on**

3. modal: photo/img/all

   **choose what kind of modal of the data that you choose to train the model.**
4. bands: 0,1,2,3 corresponding to u,v,g,r,i,z bands v,g,r,i,z bands g,r,i,z bands r,i,z bands
   **choose the bands of SkyMapper due to the lack of mag.**
5. Other training-related configurations such as weight coefficients, selection of loss functions (CRPS, cross-entropy,
   MSE), and specific CSV file reading can all be modified in train.py.

<!-- TOC --><a name="quick-start-of-the-structure-of-the-code"></a>

# Quick start of the structure of the code

1. model files:

   1.1 resModel.py: Backbone network of ResNet101

   1.2 ImprovedVitModel.py: Backbone network of Vision Transformer.

   1.3 model.py: The complete network structure integrates cross-modal contrastive learning methods and downstream task
   processing using image backbone networks.

2. util files:

   2.1 pytorchtools.py: Including a utility class for training strategies like early stopping.

   2.2 utils.py: Including a utility class for reading SDSS and WISE images and photometric data, as well as specific
   implementations of various loss functions.

3. other files:

   3.1 train.py: py files to train the model.

   3.2 SkyMapperDataset.py: A utility class for reading SkyMapper photometric data and images.

<!-- TOC --><a name="csv-files-enumerate"></a>

# Csv files enumerate

- MN.csv (Dataset I in paper [Hong S, Zou Z, Luo A L, et al. PhotoRedshift-MML: A multimodal machine learning method for
  estimating photometric redshifts of quasars[J]. Monthly Notices of the Royal Astronomical Society, 2023, 518(4):
  5049-5058.])
- YAONEW.csv (Dataset II in paper [Yao L, Qiu B, Luo A L, et al. Photometric redshift estimation of quasars with fused
  features from photometric data and images[J]. Monthly Notices of the Royal Astronomical Society, 2023, 523(4):
  5799-5811.])
- SKYSRIZ_NEW.csv (WISE and SKYMAPPER DATASET) [Dataset IV in our paper]
- UNIFY.csv (WISE and SDSS DATASET) -(Quasars $0 \leq z \leq 6$ in DR16Q) [Dataset V in our paper]

<!-- TOC --><a name="image-cropped-details-and-preprocessing"></a>

# Image cropped details and preprocessing

In this paper, the images will be cropped into 64 X 64 images according to the Ra and Dec coordinates of the quasar. The
images that are less than 64 X 64 will be padded with 0. The images processed by WISE are usually 64 x 64 matrices with
4 channels, and the images processed by SDSS are usually 64 X 64 matrices with 5 channels. They are usually named as
{name}-{ $z_{spec}$ }-{id}.mat or {name}-{ $z_{spec}$ }-{id}.mat. For SKYMAPPER images, we directly use the official API
to crop the 64*64 images as single fits file.

The preprocessed images can be accessed at https://nadc.china-vo.org/res/r101446/.

- The images (SDSS bands) in Dataset I are included in the SDSS.tar.bz2 file.
- The images (WISE and SDSS bands) used in Dataset II and Dataset VI are included in the unify.tar.bz2 file.
- The WISE and SKYMAPPER images used in Dataset V are included in the SkyFullImage.tar.bz2 file.

The total size of the image files is 202.48GB.

<!-- TOC --><a name="Notice"></a>

# Notice

Since there are a large number of extremely low redshift samples in the SKYMAPPER dataset, mixed double-precision
training cannot be enabled when training the model to avoid loss of accuracy.
