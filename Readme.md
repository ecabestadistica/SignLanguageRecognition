# SignLanguageRecognition

Please cite this paper:

Cabana, E. (2025). Advancing Accessible AI: A Comprehensive Dataset and Neural Models for Real-Time American Sign Language Alphabet Classification. In: Arai, K. (eds) Intelligent Systems and Applications. IntelliSys 2025. Lecture Notes in Networks and Systems, vol 1567. Springer, Cham. 
https://doi.org/10.1007/978-3-032-00071-2_15

Web application: https://signlanguagerecognition.aprendeconeli.com/

Licence Attribution-NonCommercial-ShareAlike 4.0 International: https://github.com/ecabestadistica/SignLanguageRecognition/tree/master?tab=License-1-ov-file#readme

Datasets for training and testing the models can be found at Mendeley Data under CC BY 4.0 license: https://data.mendeley.com/datasets/jdyksv2jhh/1

Please cite the dataset as well if re-used:

Cabana Garceran del Vall, Elisa (2025), “American Sign Language Alphabet Dataset”, Mendeley Data, V1, doi: 10.17632/jdyksv2jhh.1


## Folder structure

### Notebooks

    notebooks for training the models

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
