# Strain Prediction Using PCA and Temporal Convolutional Networks

**Notebook:** `CGN_P11_12_PCA_v2.ipynb`

## 1. Objective

Predict the strain response at sensor `SB05_x` using acceleration measurements and a causal Temporal Convolutional Network (TCN). PCA reduces the dimensionality of the acceleration signals before model training.

## 2. Dataset and Recording Split

- Dataset: `data_100Hz.h5`
- Sampling frequency: 100 Hz
- Input: 75 acceleration channels (selected AL channels and AG y/z channels)
- Target: `SB05_x`
- Structural condition: Undamaged (UDS)
- Excitation: Sweep mode (SM)
- Shaker positions: P1 and P2
- Directions: Y and Z
- Noise-mode recordings: Excluded (`ADD_NOISE_TO_TRAINING = False`)

The split is performed at the **recording level**, rather than randomly splitting overlapping windows.

| Set | Recordings | Selection |
| --- | ---: | --- |
| Training | 6 | All remaining undamaged sweep recordings |
| Validation | 1 | `MVS_P1_UDS_SM_Z_02` |
| Testing | 1 | `MVS_P2_UDS_SM_Y_02` |
| Total | 8 | Undamaged sweep recordings |

The training set covers all four shaker-position/direction combinations. Validation and test recordings remain separate from training.

## 3. Data Preprocessing

1. Load acceleration channels and target strain from each HDF5 recording.
2. Construct a time index using the sampling frequency.
3. Fit a `StandardScaler` on the training acceleration measurements only.
4. Fit PCA on the standardized training acceleration measurements.
5. Transform training, validation and test recordings using the training-fitted scaler and PCA.
6. Standardize the principal components using another scaler fitted only on training PCs.
7. Generate sliding windows separately within each recording.
8. Standardize target strain using the training-window target mean and standard deviation.

The number of acceleration features is reduced from 75 to 25. The explained variance is printed during execution; no numerical result is available in the supplied code.

## 4. Sliding Window Configuration

| Parameter | Value |
| --- | --- |
| Sampling frequency | 100 Hz |
| Principal components | 25 |
| Window size | 500 samples |
| Window duration | 5 seconds |
| Overlap | 50% |
| Stride | 250 samples |
| Time between window starts | 2.5 seconds |
| Input shape per window | `(500, 25)` |
| Target | Strain at the final sample of each window |

```python
LONG_WIN = 500
OVERLAP = 0.50
STRIDE = max(1, int(round(LONG_WIN * (1 - OVERLAP))))
```

Windows never cross recording boundaries. The task is contemporaneous strain estimation from current and preceding acceleration samples, not future strain forecasting.

## 5. Model Architecture

The model is a causal, dilated, residual 1D CNN (Temporal Convolutional Network).

```text
Acceleration signals (75 channels)
    -> Standardization
    -> PCA (25 components)
    -> PC standardization
    -> Sliding window (500 x 25)
    -> Conv1D input projection (32 filters, kernel size 1)
    -> 7 residual temporal convolution blocks
    -> Last time-step feature vector (32 features)
    -> Dense(1)
    -> Predicted strain
```

Each residual block contains a causal dilated Conv1D, layer normalization, ReLU, spatial dropout and residual addition.

| Architecture parameter | Value |
| --- | --- |
| Convolution filters | 32 |
| Temporal kernel size | 5 |
| Dilation rates | 1, 2, 4, 8, 16, 32, 64 |
| Residual blocks | 7 |
| Padding | Causal |
| Normalization | Layer normalization |
| Activation | ReLU |
| Spatial dropout | 0.25 |
| Output | One scalar |

The theoretical receptive field is `1 + (5 - 1) * (1 + 2 + 4 + 8 + 16 + 32 + 64) = 509` samples. With 500-sample windows, the earliest part of this receptive field uses causal padding.

## 6. Training Configuration

| Hyperparameter | Value |
| --- | --- |
| Framework | Keras with PyTorch backend |
| Optimizer | Adam |
| Initial learning rate | 0.0005 |
| Loss | Mean squared error (MSE) |
| Maximum epochs | 200 |
| Batch size | 32 |
| Early stopping patience | 25 epochs |
| Early stopping monitor | Validation loss |
| Restore best weights | True |
| LR reduction factor | 0.5 |
| LR reduction patience | 8 epochs |
| Minimum learning rate | 0.00001 |
| Shuffle training windows | True |
| Random seed | 0 |
| Target standardization | Enabled |

GPU (CUDA) is used when available; otherwise, CPU is used. The seed is set for reproducibility, although deterministic execution is not guaranteed.

## 7. Evaluation

The best validation-loss checkpoint is evaluated on the training, validation and test sets. Predictions are converted back to physical units before calculating R² and RMSE.

Reported metrics:

- MSE in standardized target space
- Coefficient of determination (R²)
- Root mean squared error (RMSE) in the original target unit
- Target standard deviation
- RMSE / target standard deviation

| Set | MSE (scaled) | R² | RMSE | RMSE / std |
| --- | --- | --- | --- | --- |
| Training | — | — | — | — |
| Validation | — | — | — | — |
| Testing | — | — | — | — |

Results are not included in the supplied source code. Populate this table after running the notebook.

The notebook plots training, validation and test loss, the best epoch, and a constant-prediction validation baseline. Test loss is logged every epoch but is not used directly by early stopping or learning-rate scheduling. For an unbiased final assessment, avoid using these test curves to guide model selection or hyperparameter tuning.

## 8. Dense Strain Prediction at 100 Hz

After training, the same model is applied to the held-out test recording with a smaller stride:

```python
TEST_STRIDE = 1
```

- Window size remains 500 samples (5 seconds).
- Inference stride is 1 sample (0.01 seconds).
- Prediction frequency is 100 Hz.
- No retraining is required.
- The first prediction corresponds to sample index 499 (4.99 seconds).

Actual and predicted strain are plotted together, including a zoomed interval from 280 to 350 seconds. Dense predictions are visualized but separate dense-inference metrics are not calculated in the supplied code.

## 9. Configuration Notes

- The `SPLIT = (0.70, 0.15, 0.15)` setting is declared but is not used to construct the recording-level split.
- The 7/2/1 split comment describes the optional noise-enabled configuration, not the active 6/1/1 split.
- The stride comment says 500, but the actual calculated stride is 250.
- Principal components are standardized despite a later comment stating otherwise.
- The experiment label `TCN, standardized PCs + noise` is misleading because noise recordings are disabled.
- The target unit is read from HDF5 metadata; some plot labels hard-code MPa. Verify the metadata before interpreting the physical units.

## 10. Future Work

- Explore a hybrid 1D CNN and Transformer architecture (listed as the next experiment in the notebook; not implemented in the supplied code).
- Compare models using the same recording-level split and evaluation metrics.
- Evaluate dense predictions quantitatively in addition to visual comparison.

