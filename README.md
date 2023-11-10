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

1. The pretrained DC-CRN model that leverages laryngograph data for voicing detection. Initially, the model incorporates a pretraining strategy on the LibriSpeech dataset, utilizing pseudo voicing labels obtained through the RAPT algorithm. It is then fine-tuned on laryngograph datasets chosen for this study:

- FDA [1]
- PTDB-TUG [2]
- KEELE [3]
- Mocha-TIMIT [4]
- CMU Arctic [5]

Within the `./rvd/pretrained` directory, we provide six variants of the pretrained model, each finetuned with a unique combination of the aforementioned datasets:

- `rvd_cfkm_weights.pth`: Fine-tuned on CMU Arctic, FDA, KEELE, and Mocha-TIMIT.
- `rvd_cfkp_weights.pth`: Fine-tuned on CMU Arctic, FDA, KEELE, and PTDB-TUG.
- `rvd_cfmp_weights.pth`: Fine-tuned on CMU Arctic, FDA, Mocha-TIMIT, and PTDB-TUG.
- `rvd_ckmp_weights.pth`: Fine-tuned on CMU Arctic, KEELE, Mocha-TIMIT, and PTDB-TUG.
- `rvd_fkmp_weights.pth`: Fine-tuned on FDA, KEELE, Mocha-TIMIT, and PTDB-TUG.
- `rvd_all_weights.pth`: This comprehensive model maximizes the use of available laryngograph data, being fine-tuned on all five datasets for maximal coverage and performance.


   
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


2. As noted in the paper, we have identified and addressed two key issues: flawed laryngograph recordings within the PTDB-TUG dataset and inaccuracies in the Mocha-TIMIT dataset labels, attributed to noisy harmonic patterns during silent intervals in laryngograph recordings. To rectify these issues, we have preprocessed the data by removing the samples with low-quality laryngograph waveforms and correcting the labels for specific utterances. This repository contains updated lists of filenames for both the PTDB-TUG and Mocha-TIMIT datasets, from which we have removed all entries containing poor-quality laryngograph waveforms. Additionally, the repository encompasses manually corrected labels for select utterances from the Mocha-TIMIT dataset, accessible in the `./corrected_labels` directory.


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
