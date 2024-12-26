import onnx
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType


img = cv2.imread("1.png")
img = cv2.resize(img, dsize=(256, 256))
img = np.transpose(img, (2, 0, 1)) 
img = np.expand_dims(img, axis=0) # shape of img is (1, 3, 256, 256)

# Define calibration data reader
class DataReader(CalibrationDataReader):
    def __init__(self, input_name):
        self.data = iter([
            {input_name: (img).astype(np.float32)}
        ])

    def get_next(self):
        return next(self.data, None)


# Load the model
model_path = "OD.onnx"
input_name = "images"  # Replace with your actual input layer name
data_reader = DataReader(input_name)

# Perform quantization
quantized_model_path = "QInt8_model.onnx"
quantize_static(
    model_input=model_path,
    model_output=quantized_model_path,
    calibration_data_reader=data_reader,
    quant_format=QuantType.QInt8
)

print(f"Quantized model saved to {quantized_model_path}")

