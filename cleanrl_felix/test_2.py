import torch
import typing

if __name__ == "__main__":
    #import torch

    print("CUDA available:", torch.cuda.is_available())
    print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA device found")


    def set_cuda_configuration(gpu: typing.Any) -> torch.device:
        """Set up the device for the desired GPU or all GPUs."""

        if gpu is None or gpu == -1 or gpu is False:
            device = torch.device("cpu")
        elif isinstance(gpu, int):
            assert gpu <= torch.cuda.device_count(), "Invalid CUDA index specified."
            device = torch.device(f"cuda:{gpu}")
        else:
            device = torch.device("cuda")

        return device


    #device = torch.device("cpu")
    #print('torch.device("cpu")')
    #device1 = torch.device(f"cuda:10000")
    #print('torch.device(f"cuda:10000")')
    print(torch.cuda.device_count())


    print(torch.version.cuda)  # Should return the CUDA version if CUDA is installed with PyTorch
    print(torch.backends.cudnn.enabled)  # Should return True if CUDA is supported

    print(torch.__version__)


    print("PyTorch Version:", torch.__version__)  # Should print '2.5.1' or similar
    print("CUDA Version:", torch.version.cuda)  # Should print '11.8'
    print("CUDA Available:", torch.cuda.is_available())  # Should print 'True'
    print("CUDA Device Count:", torch.cuda.device_count())  # Number of GPUs
    print("Current CUDA Device:", torch.cuda.current_device())  # Current GPU ID
    print("Device Name:", torch.cuda.get_device_name(torch.cuda.current_device()))  # GPU Name






