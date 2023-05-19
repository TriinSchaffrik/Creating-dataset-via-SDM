# Samples

python image_sample.py --num_samples 2 \
     --no_instance True \
     --num_classes 151  \
     --data_dir ./data/ade20k \
     --dataset_mode ade20k \
     --attention_resolutions 32,16,8 \ 
     --diffusion_steps 1000 \
     --image_size 256 \
     --learn_sigma True \ 
     --noise_schedule linear \ 
     --num_channels 256  \
     --num_head_channels 64  \
     --num_res_blocks 2 \
     --resblock_updown True \
     --use_fp16 True \
     --use_scale_shift_norm True \
     --class_cond True \
     --s 1.5 \
     --model_path ema_0.9999_best.pt