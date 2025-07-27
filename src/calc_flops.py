import torch
from calflops import calculate_flops
from vit_pytorch.vit import ViT
from utils.AF_mobilenet import create_mobilenet_1d



# Create a main function
def main():
    compute_ViT_flops()
    compute_Mobilenet_flops()


def compute_Mobilenet_flops():
# 1. Create the model using the function
    model = create_mobilenet_1d(num_classes=5)

    # 2. Define the input shape for a single sample
    #    (batch_size, channels, length)
    batch_size = 1
    input_channels = 1
    input_length = 1000
    input_shape = (batch_size, input_channels, input_length)

    # 3. Calculate FLOPs for Inference (Forward Pass)
    flops_fwd, macs_fwd, params = calculate_flops(
        model=model,
        input_shape=input_shape,
        output_as_string=True,
        output_precision=4
    )
    print("--- PyTorch MobileNet-1D Analysis ---")
    print(f"Model Parameters: {params}")
    print(f"Inference | MACs: {macs_fwd} | FLOPs: {flops_fwd}")

    # 4. Calculate FLOPs for Training (Forward + Backward Pass)
    flops_train, macs_train, _ = calculate_flops(
        model=model,
        input_shape=input_shape,
        include_backPropagation=True,
        output_as_string=True,
        output_precision=4
    )
    print(f"Training  | MACs: {macs_train} | FLOPs: {flops_train}")
    print("---------------------------------------")



def compute_ViT_flops():
    # 1. Initialize the Vision Transformer model
    #    This configuration matches the init_vit function in your script.
    model = ViT(
        image_size=(3200, 15),
        patch_size=(80, 5),
        num_classes=2,
        dim=16,
        depth=4,
        heads=4,
        mlp_dim=4,
        pool='cls',
        channels=1,
        dim_head=4,
        dropout=0.2,
        emb_dropout=0.2
    )

    # 2. Define the input shape for the model
    #    Based on your training script, the input is reshaped to:
    #    (batch_size, channels, sequence_length, feature_dimension)
    #    Here, we use a batch size of 1 for a single-sample calculation.
    batch_size = 1
    input_shape = (batch_size, 1, 3200, 15)


    # 3. Calculate the FLOPs for the FORWARD pass (Inference)
    #    This measures the cost of a single prediction.
    flops_fwd, macs_fwd, params = calculate_flops(
        model=model,
        input_shape=input_shape,
        output_as_string=True,
        output_precision=4
    )
    print(f"ViT Model (Inference) | FLOPs: {flops_fwd} | MACs: {macs_fwd} | Params: {params}")


    # 4. Calculate the FLOPs for the TRAINING step (Forward + Backward Pass)
    #    This is achieved by setting 'include_backPropagation=True'.
    flops_train, macs_train, params_train = calculate_flops(
        model=model,
        input_shape=input_shape,
        include_backPropagation=True, # Set to True to calculate training cost
        output_as_string=True,
        output_precision=4
    )
    print(f"ViT Model (Training)   | Fwd+Bwd FLOPs: {flops_train} | Fwd+Bwd MACs: {macs_train} | Params: {params_train}")



if __name__ == "__main__":
    main()