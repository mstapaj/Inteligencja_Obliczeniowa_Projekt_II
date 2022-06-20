## Alzheimer's disease severity classification

A project that uses convolutional networks to classify the severity of Alzheimer's disease on the basis of images taken
magnetic resonance imaging. The program was written in python using the keras and tensorflow packages. The project was
made for a Computer Intelligence course at the University of Gdańsk.

## Project Status

Project completed on 8 June 2022

## Project Report

[PDF with project report (Polish language)](./Projekt_2___Inteligencja_obliczeniowa.pdf)

## Technologies Used

- keras
- numpy
- sklearn
- tensorflow

## Installation and Setup Instructions

#### Example:

Clone down this repository.

Installation:

`pip install -r requirements.txt`

To Prepare Data:

`python prepareData.py`

To Run Classification:

`python classification.py`

## Functionalities

- The project allows you to classify the severity of alzheimer's disease.
- Possibility to set options:
  - save option - train and save new model
  - load option - load ready model from file
- You can set the option to copy wrongly classified images to the wrong_predicted folder.
- You can display statistics:
    - train time
    - evaluation time of one image
    - evaluation time of test set
    - accuracy
    - error matrix
