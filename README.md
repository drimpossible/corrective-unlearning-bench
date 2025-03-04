# Corrective Unlearning
 
This repository contains simplified code for the paper:

**Corrective Machine Unlearning**  
[Shashwat Goel*](https://shash42.github.io/), [Ameya Prabhu*](https://drimpossible.github.io), [Philip Torr](https://www.robots.ox.ac.uk/~phst/), [Ponnurangam Kumaraguru](https://precog.iiit.ac.in/), [Amartya Sanyal](https://amartya18x.github.io/)

\* = Equal Contribution

[[PDF](https://openreview.net/pdf?id=v8enu4jP9B)] [[Slides](https://docs.google.com/presentation/d/1WzLnsdjy45Ror-BXDbrlG0ayLVvu6z39YU2gB0gt89M/edit#slide=id.g2dc8ccd0772_0_0)] [[Bibtex](https://github.com/drimpossible/corrective-unlearning-bench/#citation)]

<p align="center">
<a href="url"><img src="https://github.com/drimpossible/corrective-unlearning-bench/blob/main/corrective-diff.png" height="300" width="381" ></a>
</p>

<!-- ## Installation and Dependencies

* Install all requirements required to run the code on a Python 3.x by:
 ```	
# First, activate a new virtual environment
$ pip3 install -r requirements.txt
 ```
 
* Create two additional folders in the repository `data/` and `logs/` which will store the datasets and logs of experiments. Point `--data_dir` and `--log_dir` in `src/opts.py` to locations of these folders.

 * Select `Imagenet100` from Imagenet using [this link](https://github.com/wuyuebupt/LargeScaleIncrementalLearning/tree/master/dataImageNet100) and TinyImagenet from [this link](https://www.kaggle.com/competitions/tiny-imagenet/data) and convert them to `ImageFolder` format with `train` and `test` splits.   -->
 
## Usage

* The code first pretrains a model if one with the specified parameters is not alredy available. Otherwise it uses the stored model. It then performs the specified unlearning procedure.  
```
$ python main.py --dataset=CIFAR10 --num_classes=10 --model=resnet9 --pretrain_iters=4000 --dataset_method=labeltargeted --forget_set_size=500 --deletion_size=250 --unlearn_method=EU --unlearn_iters=4000 --k=-1
```
The above script trains a Resnet9 model on CIFAR10 with 500 samples manipulated using Interclass Confusion. It then picks 250 samples known to the unlearning procedure, here retraining the whole model from scratch without the deletion data.  
Arguments you can freely tweak given a dataset and model: 
  - Pretrain Iterations (`--pretrain_iters`)
  - Manipulation Type (`--dataset_method`)
  - No. of Manipulated Samples (`--forget_set_size`)
  - No. of unlearning samples known to developers (`--deletion_size`)
  - Unlearning method (`--unlearn_method`)
  - Method specific hyperparameters, which can be inferred from `src/opts.py` 

Adding functionality can be done as follows:
  - To add new evaluations, modify `src/datasets.py`, specifically the `manip_dataset` function and `DatasetWrapper` class.  
  - To add new unlearning methods, modify `src/methods.py`
  - To add new datasets, modify the `load_dataset` function in `src/datasets.py`
  - In general, any of these changes would require updates to `src/main.py`, `src/opts.py` and `src/visualize.py`

Additional details and default hyperparameters can be found in `src/opts.py` 
  
 * To replicate the complete set of experiments, run `scripts/CIFAR10_poisoning.sh` and similar for other datasets and manipulations. 

##### If you discover any bugs in the code please contact me, I will cross-check them with my nightmares.


## Citation

We hope you are excited by the potential for corrective unlearning! To cite our work:

```
@article{goel2024corrective,
      title={Corrective Machine Unlearning}, 
      author={Shashwat Goel and Ameya Prabhu and Philip Torr and Ponnurangam Kumaraguru and Amartya Sanyal},
      journal={Transactions on Machine Learning Research},
      year={2024},
      url={https://openreview.net/forum?id=v8enu4jP9B}
}
```
