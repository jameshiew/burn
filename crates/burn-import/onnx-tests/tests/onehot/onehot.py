#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/onehot/onehot.onnx

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, num_classes: int = 5):
        super(Model, self).__init__()
        self._num_classes = num_classes

    def forward(self, indices):
        y = torch.nn.functional.one_hot(indices.long(), num_classes=self._num_classes)
        return y

def main():
    # Export to onnx
    model = Model(num_classes=5)
    model.eval()
    device = torch.device("cpu")
    onnx_name = "onehot.onnx"
    
    dummy_input = torch.tensor([0, 2, 3], device=device)
    
    torch.onnx.export(model, dummy_input, onnx_name,
                     verbose=False, opset_version=16)
    
    print(f"Finished exporting model to {onnx_name}")

    # Output some test data
    test_input = torch.tensor([1, 4, 2], device=device)
    print(f"Test input data shape: {test_input.shape}")
    output = model.forward(test_input)
    print(f"Test output data shape: {output.shape}")

if __name__ == '__main__':
    main()