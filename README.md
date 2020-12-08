# Toy Dialog System

This dialog system is based on a very basic Transformer Seq2Seq architecture
(see https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf for the
original Transformer architecture). Much of the code was adapted from
https://pytorch.org/tutorials/beginner/transformer_tutorial.html

## Install Instructions

These instructions assume Python 3 and a Linux operating system. You may run
into issues in other environments.

### Virtual Environment

To avoid dependency issues, it is recommended that you first create a new
virtual environment with `python -m venv ENV`. If python complains about
not having the venv module, you first need to `pip install venv`. Activate
the ENV with `. ENV/bin/activate`.

### Install Dependencies

`pip install -r requirements.txt`

## Runtime Instructions

You can use the pretrained model in Github, which will likely say some
variation on "wire willing there able implies implies formally genesis
genesis genesis formally formally agricultural agricultural with"
because it doesn't seem to be training right.

To train another model, type `python train.py`. This will generate two files,
`model.pickle` and `TEXT.pickle`. Both of these files are necessary for
`talk.py` to function.

To talk with the model, type `python talk.py`.

## Other Information

The data used is from http://yanran.li/dailydialog


