## CPSC-DFKD
This repository is the core code of 'Conditional pseudo-supervised contrast for data-Free knowledge distillation', published in Journal of Pattern Recognition.


### Train the teacher on Cifar100 dataset.

```
python train_teacher.py  --gpu_id 0 --batch_size 32 --dataset cifar100 --model resnet34 --num_classes 200 --data_root /datasets/cifar100
```
