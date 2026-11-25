import json
import numpy as np
import tensorflow as tf

def load_model(model_path="models/maize_model.h5", classes_path="models/class_names.json"):
    model = tf.keras.models.load_model(model_path)
    class_names = json.load(open(classes_path))
    return model, class_names

def predict_image(model, class_names, image_path):
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    arr = tf.keras.utils.img_to_array(img)[None]
    probs = model.predict(arr, verbose=0)[0]
    i = int(np.argmax(probs))
    return {"class": class_names[i], "confidence": float(probs[i]),
            "probabilities": {c: float(p) for c, p in zip(class_names, probs)}}
