# Continual Learning Through Synaptic Intelligence

This repository contains code to reproduce the key findings of our path integral approach to prevent catastrophic forgetting in continual learning.

Zenke, F.<sup>1</sup>, Poole, B.<sup>1</sup>, and Ganguli, S. (2017). Continual Learning Through
Synaptic Intelligence. In Proceedings of the 34th International Conference on
Machine Learning, D. Precup, and Y.W. Teh, eds. (International Convention
Centre, Sydney, Australia: PMLR), pp. 3987–3995.

http://proceedings.mlr.press/v70/zenke17a.html

<sup>1</sup>) Equal contribution

## BibTeX
```
@InProceedings{pmlr-v70-zenke17a,
title = 	 {Continual Learning Through Synaptic Intelligence},
author = 	 {Friedemann Zenke and Ben Poole and Surya Ganguli},
booktitle = 	 {Proceedings of the 34th International Conference on Machine Learning},
pages = 	 {3987--3995},
year = 	 {2017},
editor = 	 {Doina Precup and Yee Whye Teh},
volume = 	 {70},
series = 	 {Proceedings of Machine Learning Research},
address = 	 {International Convention Centre, Sydney, Australia},
month = 	 {06--11 Aug},
publisher = 	 {PMLR},
pdf = 	 {http://proceedings.mlr.press/v70/zenke17a/zenke17a.pdf},
url = 	 {http://proceedings.mlr.press/v70/zenke17a.html},
}
```


## Requirements

We have tested this code with the following configuration:

* Python 3.5.2
* Jupyter 4.3.0
* Tensorflow 1.2.1
* Keras 2.0.5


**Update:** Since some recent Keras update seems to have broken our code, here is a quick fix as to how to run with the old compatible Keras version
using virtualenv (https://virtualenv.pypa.io).

```
virtualenv -p python3 env
source env/bin/activate
pip3 install -vI keras==2.0.5
pip3 install jupyter matplotlib numpy tensorflow-gpu tqdm seaborn
```
