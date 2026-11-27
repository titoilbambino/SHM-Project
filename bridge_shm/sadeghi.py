"""Endpoint adaptation of Sadeghi Eshkevari's CNN and attention network (Fig. 2.2)."""

import os

os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import layers


class TimeEmbedding(layers.Layer):
    def build(self, input_shape):
        self.positions = self.add_weight(
            name="positions", shape=input_shape[1:], initializer="random_normal"
        )

    def call(self, inputs):
        return inputs + self.positions


def build_sadeghi(n_features, architecture, learning_rate):
    inputs = keras.Input(shape=(architecture["window_samples"], n_features))
    channels = architecture["cnn_channels"]
    kernel = architecture["kernel_size"]
    x = layers.Dense(channels, activation="relu", name="embedding_hidden")(inputs)
    x = layers.Dense(channels, name="embedding")(x)
    features = [x]
    for i in range(3):
        h = layers.Conv1D(channels, kernel, padding="same", activation="relu",
                          name=f"cnn_{i + 1}")(x)
        x = layers.Add()([x, layers.LayerNormalization()(h)])
        features.append(x)
    features.append(layers.Conv1D(channels, kernel, padding="same", name="cnn_4")(x))
    x = TimeEmbedding(name="time_embedding")(layers.Concatenate()(features))

    width = len(features) * channels
    heads = architecture["attention_heads"]
    if width % heads:
        raise ValueError("Attention heads must evenly divide the concatenated feature width")
    for i in range(2):
        h = layers.Dense(width, activation="relu", name=f"attention_mlp_{i + 1}")(x)
        # The final readout only uses the last query. Both blocks attend to every key and value.
        query = h if i == 0 else h[:, -1:, :]
        x = layers.MultiHeadAttention(num_heads=heads, key_dim=width // heads,
                                      name=f"attention_{i + 1}")(query, h)
        x = layers.LayerNormalization()(x)
    x = layers.Dense(architecture["decoder_width"], activation="relu",
                     name="decoder_hidden")(x[:, -1, :])
    x = layers.Dense(architecture["decoder_width"], activation="relu", name="decoder")(x)
    model = keras.Model(inputs, layers.Dense(1, name="strain")(x), name="sadeghi_endpoint")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss="mse")
    return model
