1. 更换镜像源为官方源（清华源有些时候没有特定包）
```
conda config --set channel_alias https://conda.anaconda.org
```

2. linux cuda
```
conda install python==3.10
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install qtdm
```