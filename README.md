# SelfTMO
This is the code of EG 22 paper: Learning a self-supervised tone mapping operator via feature contrast masking loss

## Environment Configuration
```bash
conda env create -f environment.yaml
```

## Online Training & Testing
Download the pre-trained VGG
- download VGG weights [VGG](https://drive.google.com/file/d/1C4VJTAyNjDcc2tQwRUkZIwLPKEiClVVl/view?usp=sharing)
```bash
python selftmo.py 
```
You can change the 'data_dir' to your own testing images path

## Offline Testing (Fast version)
```bash
python selftmo_offline.py
```
You can change the 'im_dir' to your own testing images path
