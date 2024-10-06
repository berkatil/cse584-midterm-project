# cse587-midterm-project
<!-- I will modify the readme better when we're done with all experiments - janice- -->
This folder contains codes and results with augmented datasets for training the classifiers. Dataset visualization has been provided in the directory named "data visualization."
Owned by Jihyun Janice Ahn, Bucky Park, and Berk Atil


## Data Augmentation 
We extracted some portions of the dataset from 6 dataset resources and halved them to make X_i set. About 28k unique X_is were generated and each x_i has a label indicating where an original sentence was from; [1](https://www.kaggle.com/datasets/thedevastator/hellaswag-a-new-commonsense-nli-dataset), [2](https://www.kaggle.com/datasets/stanfordu/stanford-natural-language-inference-corpus/data), [3](https://github.com/tylin/coco-caption/blob/master/results/captions_val2014_fakecap_results.json), [5](https://cims.nyu.edu/~sbowman/multinli/), [6](https://www.kaggle.com/datasets/nltkdata/brown-corpus/data?select=brown.csv ), [7](https://www.kaggle.com/datasets/oswinrh/bible?select=t_asv.csv ). Then our X_i set was fed to 7 Large Language Models that we selected to use to generate X_js. Generated X_js with X_i were stored in CSV files in the `dataset` directory, with the model's name. All datasets were combined in the file name `combined_data.csv` which was used for training our classifiers. Visualization of it is stored in the `data_visualization` folder. <!-- maybe I can put two images here as well if needed. (don't think so cuz we're going to write a report with it right -->

## Training Classifiers
Two classifiers (SBert and Luar) were selected for our experiment and were trained with our generated dataset (`combined_data.csv`). By testing in different environments and parameter settings, we found our best scenarios for each classifiers to be trianed. Training results for each model were represented in the confusion matrix image in corresponding directories. The best model files are stored in the [Google Drive](https://drive.google.com/drive/folders/11MlZ6gF0aj2P9RJ6787FZF5eV6k9cPjN?usp=drive_link) due to the size.

You need to run `python main.py --dataset_file <path to combined data file> --architecture_type <model architecture either single or dual> --encoder_type <encoder model>` to train the desired classifiers. 
For the main experiment, make sure to use `combined_dataset` as a dataset file. 
When execution is finished, you will be able to get the best-performance model file with the confusion matrix result as an image. Also, all the results including accuracy will be shown on the terminal.

