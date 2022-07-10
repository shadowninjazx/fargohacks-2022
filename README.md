# Wells Fargo Campus Analytics Challenge 2022 Solution Report

Solution by Zhizhuo Zhou

## Reproducing the Paper

### Installing Dependencies

We recommend creating a conda environment and installing the required prerequisites there. 


To install miniconda:

- Visit [miniconda](https://docs.conda.io/en/latest/miniconda.html) website
- Download the corresponding .sh file for your system
- Linux:
    - ```chmod +x {Miniconda3-latest-Linux-x86_64.sh}```
    - ```./ {Miniconda3-latest-Linux-x86_64.sh}```
    - ```export PATH="/home/{username}/miniconda/bin:$PATH"```
    - ```source ~/.zshrc```

Make sure to replace the file names in {} with the right ones for your installation. Verify the installation of conda by typing "conda -V" in the command prompt, which should show the conda version installed. 

Create a new conda environment:

- ```conda create --name fargohacks python=3.8```
- ```conda activate fargohacks```


We require the installation of the following dependencies from their respective websites:

- PyTorch (https://pytorch.org/get-started/locally/)

After installing PyTorch, you'll need to install a few more dependencies.

- ```pip install -r requirements.txt```

### Downloading pre-extracted features

To download pre-extracted features to reproduce the paper, run ```download_weights.py```

- ```python download_weights.py```

Alternatively, download the entire code archive along with the weights from Google drive https://drive.google.com/drive/folders/13SaqCq_e-QbhvLWdslIOAiWNJUCbHDTW?usp=sharing and saving the files to ```data/train_features.pt``` and ```data/test_features.pt```. 

### Training and Predicting Test XLSX

Verify that ```data/train_features.pt``` and ```data/test_features.pt``` exist. 

To train the classifier, run:

```python classifier.py -f data/train_features.pt```

To train the classifier and predict the test CSV, run:

```python classifier.py -f data/train_features.pt -t data/test_feature.pt```

***
## Running on New Data


### Running the Feature Extractor

To run the feature extractor on new data:

```python feature_extractor.py -f data/train.xlsx -t data/test.xlsx```

Notice the syntax -f <train.xlsx> and -t <test.xlsx>

### Running the Classifier 

Run the classifier with the following syntax:
- -f [train features] address of saved train features
- -t [test features] address of saved test features
- -c [test xlsx] address of saved test xlsx
- -o [output] address of the desired output xlsx

```python classifier.py -f data/train_features.pt -t data/test_feature.pt -c data/test.xlsx -o pred.xlsx```

***
## Modifying the Code

Every class and function are well documented. To experiment and modify the code, I recommend looking through both of the files. 


