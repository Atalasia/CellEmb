import downstream.train_1d_dose as train_1d_dose
import downstream.train_2d as train_2d
import torch

import argparse
import importlib

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

torch.cuda.set_device(0)
torch.cuda.empty_cache()

torch.set_num_interop_threads(15)
torch.set_num_threads(15)

VALID_SCRIPTS = ["single_dose", "combination"]

def main():
    parser = argparse.ArgumentParser(description="Running the downstream task")
    parser.add_argument("mode", type=str, help="Name of the mode you wish to run.")
    parser.add_argument("data", type=str, help="Filepath to the data.")

    args = parser.parse_args()

    mode = -1
    if args.mode == "single_dose" :
    	mode = "train_1d_dose"
    elif args.mode == "combination":
    	mode= "train_2d"
    else:
    	raise ValueError(f"Invalid input: '{args.mode}'. Valid options are: {', '.join(VALID_SCRIPTS)}")

    # Dynamically import the module
    module = importlib.import_module(args.script)
    module.main()


if __name__ == "__main__":
    main()
