# LRQ_MindSpore


# Neural Networks23(CCF-B) - Long-range zero-shot generative deep network quantization [paper](https://arxiv.org/pdf/2211.06816v1)

## Requirements

```
conda create -n mindspore python=3.8
pip install mindspore
pip install -r requirement.txt
```



## Evaluate pre-trained models

The pre-trained models and corresponding logs can be downloaded [here](https://drive.google.com/drive/folders/1wk0WNxHhJiUky2ymEYJBg4o6oXLM15e4?usp=sharing) 

Please make sure the "qw" and "qa" in *.hocon, *.hocon, "--model_name" and "--model_path" are correct.

For cifar10
```
python test.py --model_name resnet20_cifar10 --model_path path_to_pre-trained model --conf_path cifar10_resnet20.hocon
or
python test.py --model_name resnet20_cifar100 --model_path path_to_pre-trained model --conf_path cifar100_resnet20.hocon
```

For ImageNet
```
python test.py --model_name resnet18/mobilenet_w1/mobilenetv2_w1 --model_path path_to_pre-trained model --conf_path imagenet.hocon
```

Results of pre-trained models are shown below:

| Model     | Bit-width| Dataset  | Top-1 Acc.  |
| --------- | -------- | -------- | ----------- | 
| [resnet18](https://drive.google.com/file/d/1cMmYk9h7WhTkNRqataD0GWVCB3Dg0wPm/view?usp=sharing)  | W4A4 | ImageNet | 66.47%    | 
| [resnet18](https://drive.google.com/file/d/1k8Sh30Ftl0vkmttQYxy29XBuGRLf9H2m/view?usp=sharing)  | W5A5 | ImageNet | 69.94%    | 
| [mobilenetv1](https://drive.google.com/file/d/19Nzp6PyQcqRnAw9ZkNNpIPC8-IARfdBP/view?usp=sharing)  | W4A4 | ImageNet | 51.36%    |
| [mobilenetv1](https://drive.google.com/file/d/1KVGzD4K4qYzD-6KTtqGMYJ23YpJyuD2V/view?usp=sharing)  | W5A5 | ImageNet | 68.17%    | 
| [mobilenetv2](https://drive.google.com/file/d/16qfhgPsnORUq8EzMacoWNMNQ9npygM5h/view?usp=sharing)  | W4A4 | ImageNet | 65.10%    | 
| [mobilenetv2](https://drive.google.com/file/d/1PhsHcPLmpfcAUAxMAkWOr4-J1yJMWoXo/view?usp=sharing)  | W5A5 | ImageNet | 71.28%    |
| [resnet-20](https://drive.google.com/file/d/1MCA2bOiXnTJ3143oQW2l1c6cNsmtpmNC/view?usp=sharing)  | W3A3 | cifar10 | 77.07%    | 
| [resnet-20](https://drive.google.com/file/d/10RrZ9-weZ5Gq-g9XnVvEEh9esvyQ5_w6/view?usp=sharing)  | W4A4 | cifar10 | 91.49%    | 
| [resnet-20](https://drive.google.com/file/d/1-OJB6WJd-VyyR9JY2M6u32UBXB663nDn/view?usp=sharing)  | W3A3 | cifar100 | 64.98%    | 
| [resnet-20](https://drive.google.com/file/d/1M07Dvk747N1_CS9Tfl_EiBRV78aW7jC3/view?usp=sharing)  | W4A4 | cifar100 | 48.25%    | 

