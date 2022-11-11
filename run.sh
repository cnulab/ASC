python main.py --loss infonce --backbone resnet101 --optimizer sgd  --batch_size 25 --tau 0.08
python main.py --loss infonce --backbone resnet50  --optimizer sgd  --batch_size 25  --tau 0.1

python main.py --loss softmax --backbone resnet101 --optimizer adam  --batch_size 50
python main.py --loss softmax --backbone resnet50 --optimizer adam  --batch_size 50

python main.py --loss arcface --backbone resnet101  --optimizer adam  --batch_size 50  --s 10.0 --m 0.4
python main.py --loss arcface --backbone resnet50  --optimizer adam  --batch_size 50  --s 10.0 --m 0.4

python main.py --loss triplet --backbone resnet101  --optimizer adam  --batch_size 45 --margin 0.25
python main.py --loss triplet --backbone resnet50  --optimizer adam  --batch_size 45 --margin 0.25
