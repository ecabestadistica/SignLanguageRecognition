# SignLanguageRecognition

## Folder structure

### Notebooks

    notebooks for training the models

### Datasets

    datasets for training the models

    Datasets currently at: https://drive.google.com/drive/folders/1-qBEfke0XmfN13_zaccMZa5YH0vfvNJa?usp=drive_link

### TrainedModels

    pretrained models just for testing

    Trained models currently at: https://drive.google.com/drive/folders/1-qBEfke0XmfN13_zaccMZa5YH0vfvNJa?usp=drive_link

### testApp

    the apps/scripts for testing the models

### testApp/Signos XX

    Example images for testing app



### Note if you are in google colab pipenv is mostly not needed

### Note About the dependencies, tensorflow 2.14.1 works with cuda 11 if you have cuda 12 you need to install tensorflow 2.15.X

## How to run the app

This folder constains Pipenv file for installing the dependencies. To install the dependencies, run the following command:

```bash
    pip install pipenv
    pipenv install
```

To run the app, run the following command:

```bash
    pipenv run python testApp/test_app_numbers.py
    or
    pipenv run python testApp/test_app_letters.py
```

## How to train the models

This folder constains Pipenv file for installing the dependencies. To install the dependencies, run the following command:

```bash
    pip install pipenv
    pipenv install
```

Configure your jupyter to use the pipenv kernel and then run the notebooks in the notebooks folder. Follow the instructions in the notebooks to train the models.