from __future__ import print_function

import os

import cv2
import numpy as np
import tensorflow as tf
from config import imshape
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import (
    Conv2D,
    Conv2DTranspose,
    Dropout,
    Input,
    Lambda,
    MaxPooling2D,
    SeparableConv2D,
    concatenate,
)
from keras.models import Model
from keras.optimizers import Adam


def weighted_cross_entropy(beta):
    def loss(y_true, y_pred):
        weight_a = beta * tf.cast(y_true, tf.float32)
        weight_b = 1 - tf.cast(y_true, tf.float32)

        o = (tf.math.log1p(tf.exp(-tf.abs(y_pred))) + tf.nn.relu(-y_pred)) * (
            weight_a + weight_b
        ) + y_pred * weight_b
        return tf.reduce_mean(o)

    return loss


def dice(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


def iou_coef(y_true, y_pred, smooth=1):
    # arreglo=tf.make_ndarray(y_pred)
    # np.save('predictedEval',arreglo)
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3]) - intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou


def dice_loss(y_true, y_pred):
    # y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.sigmoid(y_pred)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)

    return 1 - numerator / denominator


def modelUnet():
    IMG_HEIGHT = imshape[0]
    IMG_WIDTH = imshape[1]
    IMG_CHANNELS = imshape[2]
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    # s = Lambda(lambda x: x / 255) (inputs)

    # c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
    c1 = SeparableConv2D(
        16, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(inputs)
    c1 = Dropout(0.1)(c1)
    c1 = SeparableConv2D(
        16, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = SeparableConv2D(
        32, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(p1)
    c2 = Dropout(0.1)(c2)
    c2 = SeparableConv2D(
        32, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = SeparableConv2D(
        64, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(p2)
    c3 = Dropout(0.2)(c3)
    c3 = SeparableConv2D(
        64, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = SeparableConv2D(
        128, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(p3)
    c4 = Dropout(0.2)(c4)
    c4 = SeparableConv2D(
        128, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = SeparableConv2D(
        256, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(p4)
    c5 = Dropout(0.3)(c5)
    c5 = SeparableConv2D(
        256, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c5)
    u6 = concatenate([u6, c4])
    c6 = SeparableConv2D(
        128, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(u6)
    c6 = Dropout(0.2)(c6)
    c6 = SeparableConv2D(
        128, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c6)
    u7 = concatenate([u7, c3])
    c7 = SeparableConv2D(
        64, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(u7)
    c7 = Dropout(0.2)(c7)
    c7 = SeparableConv2D(
        64, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(c7)
    u8 = concatenate([u8, c2])
    c8 = SeparableConv2D(
        32, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(u8)
    c8 = Dropout(0.1)(c8)
    c8 = SeparableConv2D(
        32, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding="same")(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = SeparableConv2D(
        16, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(u9)
    c9 = Dropout(0.1)(c9)
    c9 = SeparableConv2D(
        16, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(c9)

    outputs = Conv2D(2, (1, 1), activation="softmax")(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False, name="unet"),
        metrics=[dice, iou_coef],
    )
    # model.compile(optimizer='sgd', loss="mean_absolute_error", metrics=[dice])
    model.summary()
    return model
