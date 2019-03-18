# coding: utf-8

from tensorflow import lite
import numpy as np


def convert_to_tflite(kers_model_path, new_tflite_model):
    converter = lite.TFLiteConverter.from_keras_model_file(kers_model_path)
    tflite_model = converter.convert()
    open(new_tflite_model, "wb").write(tflite_model)


def test_tflite_model(tflite_model):

    # Load TFLite model and allocate tensors.
    interpreter = lite.Interpreter(model_path=tflite_model)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Print input and output 'name' and 'shape'
    print(f'Model-Input > Name: {input_details[0]["name"]}, Shape: {input_details[0]["shape"]}')
    print(f'Model-Output > Name: {output_details[0]["name"]}, Shape: {output_details[0]["shape"]}\n')

    # Test model on random input data.
    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)

