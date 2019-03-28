# PSL-DL
Data process:

fasta --> txt --> npy

Program running process:

python tok2id.py data/wiki/ch/


python pretrain_lm.py data/wiki/ch/ -1 --lr 1e-3 --cl 1


python finetune_lm.py data/wiki/ch/ data/wiki/ch/ -1 10 --lm-id pretrain_ 

(only for your main data that classfication when running finetune_lm.py)

 
python train_clas.py data/wiki/ch/ -1 --lm-id pretrain_ --clas-id pretrain_ --cl 10 


python eval_clas.py data/wiki/ch/ -1 --lm-id pretrain_ --clas-id pretrain_
