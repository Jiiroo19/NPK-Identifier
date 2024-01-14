
import tensorflow as tf
from tensorflow import lite


# 0 = OM, 1 = P, 2 = K
for target_what in range(0,3):
    if target_what == 0:
        model_dir = "./models/final_regression_model_OM.h5"
        convert_lite = "./converted/final_regression_model_OM.tflite"

    elif target_what == 1:
        var_name = "Phosphorus (P), ppm"
        model_dir = "./models/final_regression_model_P.h5"
        convert_lite = "./converted/final_regression_model_P.tflite"

    else:
        var_name = "Potassium [K], ppm"
        model_dir = "./models/final_regression_model_K.h5"
        convert_lite = "./converted/final_regression_model_K.tflite"

    # Convert H5 file model to tflite
    model_cnn = tf.keras.models.load_model(model_dir)
    converter = lite.TFLiteConverter.from_keras_model(model_cnn)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Optional optimizations
    tflite_model = converter.convert()

    # Specify the desired filename for the TensorFlow Lite model
    with open(convert_lite, "wb") as f:
        f.write(tflite_model)
