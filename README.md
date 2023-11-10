<h1 align="center">Leveraging Laryngograph Data for Robust Voicing Detection in Speech</h1>
<div align="center">

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
<!-- [![Downloads](https://static.pepy.tech/badge/torchcrepe)](https://pepy.tech/project/torchcrepe) -->

</div>

Code for robust voicing detection using the model described in:

Y. Zhang, H. Wang, and D.L. Wang, "Leveraging Laryngograph Data for Robust Voicing Detection in Speech", to be submitted, 2023

We kindly request that academic publications utilizing this repo cite the aforementioned paper.

## Description

Accurately detecting voiced intervals in speech signals is a critical step in pitch tracking and has numerous applications. While conventional signal processing methods and deep learning algorithms have been proposed for this task, their need to fine-tune threshold parameters for different datasets and limited generalization restrict their utility in real-world applications. To address these challenges, this repo provides a supervised voicing detection model that leverages recorded laryngograph data. The model is based on a densely-connected convolutional recurrent neural network (DC-CRN), and trained on data with reference voicing decisions extracted from laryngograph data. Pre-training is also investigated to improve the generalization ability of the model. 

This repository comprises mainly two parts:

1. The pretrained DC-CRN model that leverages laryngograph data for voicing detection. The provided model incorporates a pretraining strategy on the LibriSpeech dataset where the pseudo voicing labels are extracted using RAPT and then train on laryngograph datasets selected for this study: FDA [1], PTDB-TUG [2], KEELE [3], MochaTIMIT [4], and CMU Arctic [5]. We provide six pretrained models, all listed in './rvd/pretrained' directory:
   a. rvd_cfkm_weights.pth : trained on CMU Arctic, FDA, KEELE, MochaTIMIT.
   b. rvd_cfkp_weights.pth : trained on CMU Arctic, FDA, KEELE, PTDB-TUG.
   c. rvd_cfmp_weights.pth : trained on CMU Arctic, FDA, MochaTIMIT, PTDB-TUG.
   d. rvd_ckmp_weights.pth : trained on CMU Arctic, KEELE, MochaTIMIT, PTDB-TUG.
   e. rvd_fkmp_weights.pth : trained on FDA, KEELE, MochaTIMIT, PTDB-TUG.
   f. rvd_all_weights.pth : maximizes the use of available laryngograph data and is trained on all five laryngograph datasets.
   
<!-- The following results were obtained when evaluated on previously unseen test utterances.

<div align="center">
<table>
    <thead>
        <tr>
            <th> </th>
            <th><sub>PTDB</sub></th>
            <th><sub>Mocha TIMIT</sub></th>
            <th><sub>KEELE</sub></th>
            <th><sub>FDA</sub></th>
            <th><sub>CMU Arctic</sub></th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><sub>RAPT [6]</sub></td>
            <td><sub>3.47%</sub></td>
            <td><sub>10.41%</sub></td>
            <td><sub>5.75%</sub></td>
            <td><sub>4.61%</sub></td>
            <td><sub>6.46%</sub></td>
        </tr>       
        <tr>
            <td><sub>DC-CRN</sub></td>
            <td><sub></sub></td>
            <td><sub></sub></td>
            <td><sub></sub></td>
            <td><sub></sub></td>
            <td><sub></sub></td>
        </tr>        
    </tbody>
</table>
</div> -->


2. As noted in our paper, we identified inaccuracies in some labels within the Mocha-TIMIT dataset. This repository includes our manually corrected labels for specific utterances. Additionally, we have updated the file lists for the PTDB-TUG and Mocha-TIMIT datasets to exclude those entries with low-quality laryngograph waveforms.


## Installation

To install the package, clone the repo and execute the following command in the rvd root folder:

```bash
$ pip install .
```

<!-- The package will be made available on PyPI. To install it, execute the following command within your Python environment:

```bash
$ pip install rvd
``` -->


## Usage

### computing the voicing decision from audio

```python
import rvd


# Assign the path of the audio file
audio_filename = ...

# Specify the path to the pre-trained model file
model_file = ...

# Set the voicing threshold
voicing_threshold = ...

# Choose a device to use for inference
device = 'cuda:0'

# Create the model with the specified model file, voicing threshold, and device
model = rvd.Model(model_file, voicing_threshold, device)

# Compute voicing decision using first gpu
vd = model.predict(audio_filename)

```


## References

[1] P. C. Bagshaw, S. M. Hiller, and M. A. Jack, “Enhanced pitch tracking and the processing of F0 contours for computer aided intonation teaching,” in Proc. Eurospeech, 1993.

[2] G. Pirker, M. Wohlmayr, S. Petrik, and F. Pernkopf, “A pitch tracking corpus with evaluation on multipitch tracking scenario,” in Proc. Interspeech, 2011.

[3] F. Plante, G. Meyer, and W. Ainsworth, “A pitch extraction reference database,” in Proc. Eurospeech, 1995.

[4] https://data.cstr.ed.ac.uk/mocha/

[5] J. Kominek and A. W. Black, “The CMU arctic speech databases,” in Fifth ISCA workshop on speech synthesis, 2004.

[6] D. Talkin and W. B. Kleijn, “A robust algorithm for pitch tracking (RAPT),” Speech Coding and Synthesis, vol. 495, p. 518, 1995.
