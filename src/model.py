import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(num_classes=4):
    augment = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.1), layers.RandomZoom(0.1)])
    base = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet")
    base.trainable = False
    inputs = layers.Input(shape=(224, 224, 3))
    x = augment(inputs)
    x = layers.Rescaling(1./127.5, offset=-1)(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def train_model(model, train_ds, val_ds, epochs=10, class_weight=None):
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                          restore_best_weights=True)
    return model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                     class_weight=class_weight, callbacks=[es])

def retrain(model_path, train_ds, val_ds, epochs=3, save_path=None):
    """Load the saved model, train it further on new data, and save it again."""
    model = tf.keras.models.load_model(model_path)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    model.save(save_path or model_path)
    return history
