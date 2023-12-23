# SignLanguageRecognition

## Folder structure

### Notebooks

    notebooks for training the models

### Datasets

    datasets for training the models

### TrainedModels

    pretrained models just for testing

### testApp

    the apps/scripts for testing the models


## How to run the app

This folder constains Pipenv file for installing the dependencies. To install the dependencies, run the following command:

```bash
    pip install pipenv
    pipenv install
```

To run the app, run the following command:

```bash
    pipenv run testApp\\test_app_numbers.py
    or
    pipenv run testApp\\test_app_letters.py
```

## How to train the models

This folder constains Pipenv file for installing the dependencies. To install the dependencies, run the following command:

```bash
    pip install pipenv
    pipenv install
```

Configure your jupyter to use the pipenv kernel and then run the notebooks in the notebooks folder. Follow the instructions in the notebooks to train the models.