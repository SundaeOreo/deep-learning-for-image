# -*-coding:utf-8-*-
import io
import torch
import torch.onnx
from model import MobileNetV2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test():
    model = MobileNetV2()
    pthfile = '/home/pateo/JUL/CODE/deep-learning-for-image-processing-master/pytorch_classification/Test6_mobilenet/MobileNetV2.pth'
    loaded_model = torch.load(pthfile, map_location='cpu')
    # try:
    #   loaded_model.eval()
    # except AttributeError as error:
    #   print(error)

   # model.load_state_dict(loaded_model['state_dict'])
   # model = model.to(device)

    # data type nchw
    dummy_input1 = torch.randn(1, 3, 64, 64)
    # dummy_input2 = torch.randn(1, 3, 64, 64)
    # dummy_input3 = torch.randn(1, 3, 64, 64)
    input_names = ["actual_input_1"]
    output_names = ["output1"]
    # torch.onnx.export(model, (dummy_input1, dummy_input2, dummy_input3), "C3AE.onnx", verbose=True, input_names=input_names, output_names=output_names)
    torch.onnx.export(model, dummy_input1, "MobileNetV2.onnx", verbose=True, input_names=input_names,
                      output_names=output_names)

if __name__ == "__main__":
    test()