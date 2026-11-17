
import tensorflow as tf

model_path = "models/stroke_classifier_210_videos.keras"

model = tf.keras.models.load_model(model_path, compile=False)

print("Model input shape:", model.input_shape)
print("Model output shape:", model.output_shape)
model.summary()
