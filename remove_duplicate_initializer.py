import argparse
from tqdm import tqdm
import onnx


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="input model")
    parser.add_argument("--output", required=True, help="output model")
    args = parser.parse_args()
    return args


def remove_duplicate_initializer(model: onnx.ModelProto) -> bool:
    if model.ir_version < 4:
        print("Model with ir_version below 4 requires to include initializer in graph input")
        return False

    modified = False
    for initializer in tqdm(model.graph.initializer):
        if len(initializer.raw_data) and len(initializer.float_data):
            if len(initializer.raw_data) / len(initializer.float_data) == 4:
                while(len(initializer.float_data)):
                    initializer.float_data[:] = []

    return modified


if __name__ == "__main__":
    args = get_args()
    model = onnx.load(args.input)
    remove_duplicate_initializer(model)
    onnx.save(model, args.output)
