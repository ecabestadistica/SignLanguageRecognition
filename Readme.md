# SignLanguageRecognition

Pre-print: https://arxiv.org/abs/2404.07211 
Web application: https://signlanguagerecognition.aprendeconeli.com/

## Folder structure

### Notebooks

    notebooks for training the models

### Datasets

    datasets for training the models

    Datasets currently at: https://drive.google.com/drive/folders/1oWIGEy2vciJK8Wfl79Xg64JffZbjmzfY

### TrainedModels

    pretrained models just for testing

    Trained models currently at: https://drive.google.com/drive/folders/1u_H4_2Wc2-plryjFQnZ9C52HyTXHGn-5

### testApp

    the apps/scripts for testing the models

### testApp/Signos XX,AlphabetASL,Signos Numeros

    Example images for testing app

### testWeb

    simple web server for testing the models from anywhere without need to install the environment

### Note if you are in google colab pipenv is mostly not needed

### Note About the dependencies, tensorflow 2.14.1 works with cuda 11 if you have cuda 12 you need to install tensorflow 2.15.X also you need to take care of tensorrt to make sure is compatible with cudnn

## How to run the app

This folder constains Pipenv file for installing the dependencies. To install the dependencies, run the following command:

```bash
    pip install pipenv
    pipenv install
```

To run the app, run the following command:

```bash
    pipenv run python testApp/test_app.py
```

## How to train the models

This folder constains Pipenv file for installing the dependencies. To install the dependencies, run the following command:

```bash
    pip install pipenv
    pipenv install
```

Configure your jupyter to use the pipenv kernel and then run the notebooks in the notebooks folder. Follow the instructions in the notebooks to train the models.
