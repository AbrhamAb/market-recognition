import numpy as np
import tensorflow as tf
import os
from typing import Any, cast

MODEL_PATH = os.path.abspath('model/run_10epochs/product_classifier')
CONF_THRESHOLD = 0.5

print('Loading model from', MODEL_PATH)
m: Any = None
try:
    # Prefer tf.keras loader at runtime
    if hasattr(tf, 'keras') and hasattr(tf.keras, 'models'):
        m = tf.keras.models.load_model(MODEL_PATH)
    else:
        m = None
except Exception as e:
    print('load_model failed:', e)
    # try TFSMLayer fallback
    try:
        keras = getattr(tf, 'keras', None)
        if keras is not None and hasattr(keras, 'layers') and hasattr(keras.layers, 'TFSMLayer'):
            m = keras.layers.TFSMLayer(
                MODEL_PATH, call_endpoint='serving_default')
        else:
            raise RuntimeError('No TFSMLayer available')
    except Exception as e2:
        print('TFSMLayer fallback failed:', e2)
        raise

for i in range(5):
    x = np.random.rand(1, 224, 224, 3).astype('float32')
    preds = cast(Any, m).predict(x)
    if isinstance(preds, dict):
        preds = list(preds.values())[0]
    preds = np.array(preds)
    if preds.ndim == 2:
        p = preds[0]
    else:
        p = preds
    top_prob = float(np.max(p))
    print(
        f'sample {i}: top_prob={top_prob:.6f} below_threshold={top_prob < CONF_THRESHOLD}')
