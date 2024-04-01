#### Requirements and Installation

Run the following command to install all dependencies:

    conda env create --file condaenv.yml
    conda activate condaenv

#### Training details
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

