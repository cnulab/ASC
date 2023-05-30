python train_backbone.py --loss infonce --backbone resnet101 --optimizer sgd  --batch_size 32 --temperature 0.08 --image_size 112
python train_backbone.py --loss infonce --backbone resnet50  --optimizer sgd  --batch_size 32  --temperature 0.1  --image_size 224

python train_backbone.py --loss softmax --backbone resnet101 --optimizer adam  --batch_size 32 --image_size 112
python train_backbone.py --loss softmax --backbone resnet50 --optimizer adam  --batch_size 32  --image_size 224

python train_backbone.py --loss arcface --backbone resnet101  --optimizer adam  --batch_size 32  --s 10.0 --m 0.4 --image_size 112
python train_backbone.py --loss arcface --backbone resnet50  --optimizer adam  --batch_size 32 --s 10.0 --m 0.4  --image_size 224

python train_backbone.py --loss triplet --backbone resnet101  --optimizer adam  --batch_size 32 --margin 0.25 --image_size 112
python train_backbone.py --loss triplet --backbone resnet50  --optimizer adam  --batch_size 32 --margin 0.15 --image_size 224


python calibration.py  --backbone resnet101  --model_root arcface_resnet101 --image_size 112
python calibration.py  --backbone resnet50  --model_root arcface_resnet50 --image_size 224


python train_uncertainty.py  --backbone resnet101  --model_root arcface_resnet101 --image_size 112
python train_uncertainty.py  --backbone resnet50  --model_root arcface_resnet50 --image_size 224


python eval_uncertainty.py  --backbone resnet101  --model_root arcface_resnet101 --image_size 112 --head_checkpoint_path 'heads/epoch_60_head.pkl'
python eval_uncertainty.py  --backbone resnet50  --model_root arcface_resnet50 --image_size 224 --head_checkpoint_path 'heads/epoch_60_head.pkl'