# "Scalable and Order-robust Continual Learning with Additive Parameter Decomposition", ICLR 2020
+ Jaehong Yoon(KAIST), Saehoon Kim(AITRICS), Eunho Yang(KAIST, AITRICS), and Sung Ju Hwang(KAIST, AITRICS)

This project hosts the code for our [**ICLR 2020** paper](https://arxiv.org/abs/1902.09432).

While recent continual learning methods largely alleviate the catastrophic problem on toy-sized datasets, some issues remain to be tackled to apply them to real-world problem domains. First, a continual learning model should effectively handle catastrophic forgetting and be efficient to train even with a large number of tasks. Secondly, it needs to tackle the problem of order-sensitivity, where the performance of the tasks largely varies based on the order of the task arrival sequence, as it may cause serious problems where fairness plays a critical role (e.g. medical diagnosis). To tackle these practical challenges, we propose a novel continual learning method that is scalable as well as order-robust, which instead of learning a completely shared set of weights, represents the parameters for each task as a sum of task-shared and sparse task-adaptive parameters. With our Additive Parameter Decomposition (APD), the task-adaptive parameters for earlier tasks remain mostly unaffected, where we update them only to reflect the changes made to the task-shared parameters. This decomposition of parameters effectively prevents catastrophic forgetting and order-sensitivity, while being computation- and memory-efficient. Further, we can achieve even better scalability with APD using hierarchical knowledge consolidation, which clusters the task-adaptive parameters to obtain hierarchically shared parameters. We validate our network with APD, APD-Net, on multiple benchmark datasets against state-of-the-art continual learning methods, which it largely outperforms in accuracy, scalability, and order-robustness.

## Reference

If you use this code as part of any published research, please refer the following paper.

```
@inproceedings{yoon2020apd,
  title={Scalable and Order-robust Continual Learning with Additive Parameter Decomposition},
  author={Yoon, Jaehong and Kim, Saehoon and Yang, Eunho and Hwang, Sung Ju},
  year={2020},
  booktitle={ICLR}
}
```

## Prerequisites
- Python 3.x
- Tensorflow 1.14.0

## Dataset Preparation
We give example codes for CIFAR-100-Split / CIFAR-100-Superclass dataset.
Download CIFAR-100-python version dataset from [Toronto-CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html) to "your_directory".

## Model Run
- Run one of the followings.
- Default task order is the "orderA". you can modify "order_type" in the code.

__CIFAR-100-Split__ experiment
```
# 10 classes & 10 tasks, APD(2) which includes hierarchical knowledge consolidation
$ python cifar100_apd_bash.py
```

__CIFAR-100-Superclass__ experiment
```
# 5 classes & 20 tasks, APD(2) which includes hierarchical knowledge consolidation
$ python cifar100_sup_apd_bash.py
```


## Authors

[Jaehong Yoon](https://jaehongyoon93.wixsite.com/aboutme)<sup>1</sup>, [Saehoon Kim](https://saehoonkim.github.io/)<sup>2</sup>, [Eunho Yang](https://sites.google.com/site/yangeh/)<sup>1</sup><sup>2</sup>, and [Sung Ju Hwang](http://www.sungjuhwang.com/)<sup>1</sup><sup>2</sup>

<sup>1</sup>[KAIST](http://www.kaist.edu/) @ School of Computing, KAIST, Daejeon, South Korea

<sup>2</sup>[AITRICS](https://www.aitrics.com/) @ Seoul, South Korea
