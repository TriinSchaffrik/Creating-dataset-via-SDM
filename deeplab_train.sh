#from base directory
cd deeplabv3_pytorch-ade20k
python main.py --data_root ./datasets/data \
    --dataset ade20k \
    --num_classes 151 \
    --gpu 0 \
    --lr 0.001 \
    --model deeplabv3plus_resnet101 \
    --batch_size 12 \
    --val_batch_size 12 \
    --total_itrs 150000

# predict
#python predict.py --input datasets/data/ade20k/ADEChallengeData2016/images/test/ \
#    --dataset ade20k \
#    --model deeplabv3plus_resnet101 \
#    --ckpt checkpoints/best_deeplabv3plus_resnet101_ade20k_os16.pth \
#    --save_val_results_to test_results
