import torch
import torch.onnx
from MST_Plus_Plus import MST_Plus_Plus  # This assumes that MST_Plus_Pl


# Define your model (adjust parameters as needed)
model = MST_Plus_Plus(in_channels=3, out_channels=31, n_feat=31, stage=3).cuda()

# Load the checkpoint
pretrained_model_path = r'C:\Users\joeli\Dropbox\Code\Python Projects\MST-plus-plus\exp\mst_plus_plus\2024_03_11_19_33_47\epoch_3_train_loss0.07470536977052689_test_loss_0.06304560353358586.pth'  # Update this path
checkpoint = torch.load(pretrained_model_path)

# Load the model weights
model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()}, strict=False)

# Set the model to inference mode
model.eval()

# Define a dummy input according to the input size your model expects
dummy_input = torch.randn(1, 3, 512, 512, device='cuda')  # Adjust size (224x224) as needed

# Specify the path for the ONNX model
onnx_model_path = "path_to_save_your_model.onnx"

# Export the model
torch.onnx.export(model, dummy_input, onnx_model_path, export_params=True,
                  do_constant_folding=True, input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
