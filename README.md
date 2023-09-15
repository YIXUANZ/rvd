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

1. The pretrained DC-CRN model that leverages laryngograph data for voicing detection.
2. As noted in our paper, we identified inaccuracies in some labels within the Mocha-TIMIT dataset. This repository includes our manually corrected labels for specific utterances.

## Installation

The package is available on PyPI. To install it, execute the following command within your Python environment:

```bash
$ pip install rvd
```

## Usage

# Computing the voicing decision from audio

```python
import py_vd


# Load audio
audio, sr = py_vd.load.audio( ... )

```


## References

