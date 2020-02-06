CUDA_VISIBLE_DEVICES=0 tsp python scripts/robustness/train.py --asrc-path ./built_data/asrc --saved-preprocessor preprocessor --gpu-num 1 --models bert --exp dropout $1
CUDA_VISIBLE_DEVICES=1 tsp  python scripts/robustness/train.py --asrc-path ./built_data/asrc --saved-preprocessor preprocessor --gpu-num 1 --models bert --exp weight_decay $1
CUDA_VISIBLE_DEVICES=2 tsp  python scripts/robustness/train.py --asrc-path ./built_data/asrc --saved-preprocessor preprocessor --gpu-num 1 --models others $1
