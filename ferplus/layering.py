import onnx
from onnx import helper, TensorProto
from add_intermediate_output import add_intermediate_output

# Path to your ONNX model
onnx_path = "emotion-ferplus-8_new.onnx"
output_path = "emotion-ferplus-multi-output.onnx"

# List of layer output names to add
layer_outputs = [
    "ReLU384_Output_0",   # Early
    "ReLU496_Output_0",   # Middle
    "ReLU636_Output_0",   # Late
    "ReLU670_Output_0",   # Penultimate
    "Plus692_Output_0"    # Final
]

# Start with the original model
current_input = onnx_path

for i, layer_name in enumerate(layer_outputs):
    # Always write to the same output file
    add_intermediate_output(current_input, output_path, layer_name)
    current_input = output_path  # Use the updated model for the next layer

print(f"Final model with all outputs saved as: {output_path}")
