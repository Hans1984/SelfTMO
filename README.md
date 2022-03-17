# SelfTMO
This is the code of EG 22 paper: Learning a self-supervised tone mapping operator via feature contrast masking loss

## Environment Configuration
conda env create -f environment.yaml

## online training & testing
python selftmo.py 
** You can change the 'data_dir' to your own testing images path

## offline testing (Fast version)
python selftmo_offline.py

** You can change the 'im_dir' to your wown testing images path
