import torch
from stuperml.model import MeanBaseModel, SimpleMLP

def test_simple_mlp_output_shape():
    input_size = 10
    batch_size = 8
    model = SimpleMLP(input_size=input_size)
    x = torch.randn(batch_size, input_size)
    output = model(x)
    assert output.shape == (batch_size, 1)

def test_mean_base_model_logic():
    target = torch.tensor([10.0, 20.0, 30.0, 40.0])
    model = MeanBaseModel(target_tensor=target)
    x_input = torch.randn(3, 5)
    output = model(x_input)
    assert output.shape == (3, 1)
    assert torch.allclose(output, torch.tensor([[25.0], [25.0], [25.0]]))

def test_mlp_gradient_flow():
    model = SimpleMLP(input_size=4)
    x = torch.randn(1, 4)
    output = model(x)
    output.sum().backward()
    assert model.fc1.weight.grad is not None