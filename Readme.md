## CPSC-DFKD
This repository is the core code of 'Conditional pseudo-supervised contrast for data-Free knowledge distillation', published in Journal of Pattern Recognition. (Updating...)


### Step 1. Train the teacher on Cifar100 dataset.

```python
python train_teacher.py  --gpu_id 0 --batch_size 32 --dataset cifar100 --model resnet34 --num_classes 100 --data_root /xxx/cifar100

```

### Step 2. Train the student on Cifar100 dataset by distillation.
```python
python CPSC_DFKD.py --epochs 600  --gpu_id 0 --batch_size  512  --G_steps 6 --loss_IKD --co_alpha 5  --loss_BN  --co_beta 1 --loss_SCL --co_gamma 0.7  --loss_CE --co_eta 0.7 --data_root /xxx/cifar100
```



## Citation
```
@article{shao2023conditional,
  title={Conditional pseudo-supervised contrast for data-Free knowledge distillation},
  author={Shao, Renrong and Zhang, Wei and Wang, Jun},
  journal={Pattern Recognition},
  volume={143},
  pages={109781},
  year={2023},
  publisher={Elsevier}
}
```