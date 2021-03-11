# Pytorch Implementation of SSD300 

We redesign and fix the bug in original implementation which considers `pytorch 0.4`.

This code supports `pytorch 1.0 >` in `python 3.6`.


# Objective

**To build a model that can detect and localize specific objects in images.**

<p align="center">
<img src="./img/baseball.gif">
</p>

We will be implementing the [Single Shot Multibox Detector (SSD)](https://arxiv.org/abs/1512.02325), a popular, powerful, and especially nimble network for this task. The authors' original implementation can be found [here](https://github.com/weiliu89/caffe/tree/ssd).

Here are some examples of object detection in images not seen during training –

# Usage

## Create Data List

Before you train the model, you need to preprocess the data.

`python create_data_list.py`
`python train.py`
