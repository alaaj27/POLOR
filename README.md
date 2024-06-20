# POLOR

We propose **POLOR**: leveraging contrastive learning to detect the **POL**itical **OR**ientation of opinion pieces in news media. POLOR exploits several contrastive learning objectives where each sentence is contrasted to a set of sentences from different and similar sources. We adopt different objective functions to generate additional features that focus on textual cues related to the political bias of articles, instead of the style of the news media. POLOR produces sentence-specific and article-specific labels based on the training data derived from sources.

## Requirements and Installation

Run the following command to install all dependencies:

    conda env create --file condaenv.yml
    conda activate condaenv

## Training details
Training the model with default parameters:

    cd src
    python run.py

Train the model with specific parameters:

    cd src
    python run.py --batch_size batch_size --lr learning_rate
For example:

> python run.py --batch_size 80 --lr 2e-5

Available parameters:

 -  batch_size , default= 80
  - triplet_size, default= 5
  - epochs, default=5
  - maxlen, default=80
  - lr, default=2e-5
  - pooling,  default="mean"
  - loss_objective, default="MMA"
  - alpha, default=0.25
  - beta, default=0.25
  - distance_norm, default=2
  - margin, default=1.0
  - train_path, default="../data/train.csv"
  - test_path, default="../data/test.csv"

## Citation
Please cite our paper if you find it helpful.
```
@inproceedings{jararweh2024polor,
  title={POLOR: Leveraging Contrastive Learning to Detect Political Orientation of Opinion in News Media},
  author={Jararweh, Ala and Mueen, Abdullah},
  booktitle={The International FLAIRS Conference Proceedings},
  volume={37},
  year={2024}
}

```
