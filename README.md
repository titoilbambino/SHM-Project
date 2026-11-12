# CGN P11-12 – PCA + TCN strain prediction

Jupyter notebook `CGN_P11_12_PCA_first.ipynb`.
It predicts the strain channel `SB01_x` from 75 acceleration channels:

1. Loads the undamaged sweep recordings from `data_100Hz.h5` (100 Hz).
2. Reduces the acceleration channels with PCA.
3. Cuts each recording into sliding windows and splits them into train, validation and test.
4. Trains a causal temporal convolutional network (Keras, PyTorch backend).
5. Reports R² and RMSE and plots the predictions.

## Dataset

The dataset is not in this repository.
Put `data_100Hz.h5` in a folder named `data` next to the notebook:

```
CGN_P11_12_PCA_first.ipynb
data/
    data_100Hz.h5
```

## Requirements

Python 3.10 or later. Install the packages with:

```
pip install -r requirements.txt
```

Then open the notebook in Jupyter or VS Code and run all cells.
