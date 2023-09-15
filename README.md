<h1 align="center">Leveraging Laryngograph Data for Robust Voicing Detection in Speech</h1>
<div align="center">

[![PyPI](https://img.shields.io/pypi/v/torchcrepe.svg)](https://pypi.python.org/pypi/torchcrepe)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://static.pepy.tech/badge/torchcrepe)](https://pepy.tech/project/torchcrepe)

</div>

Code for robust voicing detection using the model described in:

Y. Zhang, H. Wang, and D.L. Wang, "Leveraging Laryngograph Data for Robust Voicing Detection in Speech", to be submitted, 2023

We kindly request that academic publications utilizing this repo cite the aforementioned paper.

## Description

Accurately detecting voiced intervals in speech signals is a critical step in pitch tracking and has numerous applications. While conventional signal processing methods and deep learning algorithms have been proposed for this task, their need to fine-tune threshold parameters for different datasets and limited generalization restrict their utility in real-world applications. To address these challenges, this repo provides a supervised voicing detection model that leverages recorded laryngograph data. The model is based on a densely-connected convolutional recurrent neural network (DC-CRN), and trained on data with reference voicing decisions extracted from laryngograph data. Pre-training is also investigated to improve the generalization ability of the model. 

This repository comprises two main components:

1. The pretrained DC-CRN model that leverages laryngograph data for voicing detection. The provided pre-trained model maximizes the use of available laryngograph data and is trained on all five datasets selected for this study: FDA [1], PTDB [2], KEELE [3], MochaTIMIT [4], and CMU Arctic [5]. The following results were obtained when evaluated on previously unseen test utterances.

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
            <td><sub>1.17%</sub></td>
            <td><sub>2.22%</sub></td>
            <td><sub>0.5%</sub></td>
            <td><sub>1.66%</sub></td>
            <td><sub>2.35%</sub></td>
        </tr>        
    </tbody>
</table>
</div>


2. As noted in our paper, we identified inaccuracies in some labels within the Mocha-TIMIT dataset. This repository includes our manually corrected labels for specific utterances. 

## Installation

The package is available on PyPI. To install it, execute the following command within your Python environment:

```bash
$ pip install rvd
```

## Usage

### Computing the voicing decision from audio

```python
import py_vd


# Load audio
audio, sr = py_vd.load.audio( ... )

```


## References

[1] P. C. Bagshaw, S. M. Hiller, and M. A. Jack, “Enhanced pitch tracking and the processing of f0 contours for computer aided intonation teaching,” in Proc. Eurospeech, 1993.

[2] G. Pirker, M. Wohlmayr, S. Petrik, and F. Pernkopf, “A pitch tracking corpus with evaluation on multipitch tracking scenario,” in Proc. Interspeech, 2011.

[3] F. Plante, G. Meyer, and W. Ainsworth, “A pitch extraction reference database,” in Proc. Eurospeech, 1995.

[4] https://data.cstr.ed.ac.uk/mocha/

[5] J. Kominek and A. W. Black, “The CMU arctic speech databases,” in Fifth ISCA workshop on speech synthesis, 2004.

[6] D. Talkin and W. B. Kleijn, “A robust algorithm for pitch tracking (RAPT),” Speech Coding and Synthesis, vol. 495, p. 518, 1995.