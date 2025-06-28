import onnx
from onnx import helper, TensorProto

def add_intermediate_output(onnx_path, output_path, layer_name):
    # Load the existing ONNX model
    model = onnx.load(onnx_path)

    # Get the graph
    graph = model.graph

    # Check if the desired layer exists in the model's intermediate outputs
    layer_found = False
    for node in graph.node:
        for output in node.output:
            if output == layer_name:
                layer_found = True
                break

    if not layer_found:
        raise ValueError(f"Layer '{layer_name}' not found in the model.")

    # Check if it's already in graph output
    for output in graph.output:
        if output.name == layer_name:
            print(f"Layer '{layer_name}' is already an output.")
            return

    # Get the value_info from intermediate layers if available
    value_info = None
    for vi in graph.value_info:
        if vi.name == layer_name:
            value_info = vi
            break

    # If not found in value_info, create a generic one with unknown shape
    if value_info is None:
        value_info = helper.make_tensor_value_info(
            layer_name,
            TensorProto.FLOAT,
            None  # Shape is unknown
        )

    # Add it to the graph's outputs
    graph.output.append(value_info)

    # Save the modified model
    onnx.save(model, output_path)
    print(f"Modified model saved to '{output_path}' with '{layer_name}' as additional output.")
