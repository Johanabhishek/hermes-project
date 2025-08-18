# src/hermes/evaluation.py

from . import augmentations
from . import policy
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

def get_augmentation_function(name):
    func_map = {
        "jittering": augmentations.jitter,
        "time_warping": augmentations.time_warping,
        "scaling": augmentations.scaling,
        "permutation": augmentations.permutation,
    }
    if name in func_map:
        return func_map[name]
    else:
        raise ValueError(f"Unknown augmentation technique: {name}")

def apply_policy(X_data, policy):
    X_augmented = np.copy(X_data)
    num_samples, _, num_features = X_data.shape
    for i in range(num_samples):
        for tech in policy:
            tech_name = tech['name']
            tech_params = tech['params']
            aug_func = get_augmentation_function(tech_name)
            for j in range(num_features):
                X_augmented[i, :, j] = aug_func(X_augmented[i, :, j], **tech_params)
    return X_augmented

def build_baseline_lstm(input_shape):
    model = Sequential([
        LSTM(50, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def evaluate_model(X_train_data, y_train_data, X_test_data, y_test_data):
    y_train_cat = to_categorical(y_train_data, num_classes=2)
    y_test_cat = to_categorical(y_test_data, num_classes=2)
    input_shape = (X_train_data.shape[1], X_train_data.shape[2])
    model = build_baseline_lstm(input_shape)
    model.fit(X_train_data, y_train_cat, epochs=5, batch_size=128, validation_split=0.1, verbose=0)
    _, accuracy = model.evaluate(X_test_data, y_test_cat, verbose=0)
    return accuracy

def evaluate_policy(policy, X_train, y_train, X_test, y_test):
    print(f"-> Evaluating Policy: {[(p['name']) for p in policy]}")
    X_train_augmented = apply_policy(X_train, policy)
    accuracy = evaluate_model(X_train_augmented, y_train, X_test, y_test)
    print(f"--> Policy Score (Accuracy): {accuracy:.4f}")
    return accuracy
