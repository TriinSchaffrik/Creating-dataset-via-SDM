"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
from datetime import date

import torch as th
import torch.distributed as dist
import torchvision as tv

from guided_diffusion.image_datasets import load_data

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    dir = date.today()
    args = create_argparser().parse_args("""--num_samples 2
     --no_instance True
     --num_classes 151 
     --data_dir ./data/ade20k 
     --dataset_mode ade20k
     --attention_resolutions 32,16,8 
     --diffusion_steps 1000 
     --image_size 256 
     --learn_sigma True 
     --noise_schedule linear 
     --num_channels 256 
     --num_head_channels 64 
     --num_res_blocks 2 
     --resblock_updown True 
     --use_fp16 True 
     --use_scale_shift_norm True 
     --class_cond True 
     --s 1.5 
     --is_train True
     --model_path ema_0.9999_best.pt
     --results_path RESULTS/{dir}""".format(dir=dir).split())

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu"), strict=False
    )
    model.to(dist_util.dev())

    logger.log("creating data loader...")
    data = load_data(
        dataset_mode=args.dataset_mode,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=True,
        random_crop=False,
        random_flip=False,
        is_train=args.is_train
    )

    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    len_data_dir = len(os.listdir(args.data_dir + "/annotations/training"))
    image_path = os.path.join(args.results_path, 'images')
    os.makedirs(image_path, exist_ok=True)
    label_path = os.path.join(args.results_path, 'labels')
    os.makedirs(label_path, exist_ok=True)
    sample_path = os.path.join(args.results_path, 'samples')
    os.makedirs(sample_path, exist_ok=True)
    logger.log("sampling...")
    all_samples = []
    for i, (batch, cond) in enumerate(data):
        if i < 13010 or i % 25 != 0:
            print(cond['path'])
            continue
        image = ((batch + 1.0) / 2.0).cuda()
        label = (cond['label_ori'].float() / 255.0).cuda()
        model_kwargs = preprocess_input(cond, num_classes=args.num_classes)
        # set hyperparameter
        model_kwargs['s'] = args.s
        count = args.num_samples #  // len_data_dir if args.num_samples // len_data_dir >= 1 else 1
        for l in range(count):
            sample_fn = (
                diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            )
            sample = sample_fn(
                model,
                (args.batch_size, 3, image.shape[2], image.shape[3]),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                progress=True
            )
            sample = (sample + 1) / 2.0

            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            all_samples.extend([sample.cpu().numpy() for sample in gathered_samples])

            for j in range(sample.shape[0]):
                tv.utils.save_image(image[j], os.path.join(image_path,
                                                           cond['path'][j].split('/')[-1].split('.')[0] + "_" + str(
                                                               l) + '.png'))
                tv.utils.save_image(sample[j], os.path.join(sample_path,
                                                            cond['path'][j].split('/')[-1].split('.')[0] + "_" + str(
                                                                l) + '.png'))
                tv.utils.save_image(label[j], os.path.join(label_path,
                                                           cond['path'][j].split('/')[-1].split('.')[0] + "_" + str(
                                                               l) + '.png'))

            logger.log(f"created {len(all_samples) * args.batch_size} samples: {cond['path'][0].split('/')[-1].split('.')[0]}")
        os.remove(os.path.join(args.data_dir, "annotations/training",cond['path'][0].split('/')[-1].split('.')[0] + '.png'))
        os.remove(os.path.join(args.data_dir, "images/training",cond['path'][0].split('/')[-1].split('.')[0] + '.jpg'))
        logger.log(args.num_samples*len_data_dir)
        if len(all_samples) * args.batch_size > (args.num_samples*len_data_dir/30):
            break

    dist.barrier()
    logger.log("sampling complete")


def preprocess_input(data, num_classes):
    # move to GPU and change data types
    data['label'] = data['label'].long()

    # create one-hot label map
    label_map = data['label']
    bs, _, h, w = label_map.size()
    input_label = th.FloatTensor(bs, num_classes, h, w).zero_()
    input_semantics = input_label.scatter_(1, label_map, 1.0)

    # concatenate instance map if it exists
    if 'instance' in data:
        inst_map = data['instance']
        instance_edge_map = get_edges(inst_map)
        input_semantics = th.cat((input_semantics, instance_edge_map), dim=1)
    return {'y': input_semantics}


def get_edges(t):
    edge = th.ByteTensor(t.size()).zero_()
    edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    return edge.float()


def create_argparser():
    defaults = dict(
        data_dir="",
        dataset_mode="",
        clip_denoised=True,
        num_samples=10000,
        batch_size=1,
        use_ddim=False,
        model_path="",
        results_path="",
        is_train=True,
        s=1.0
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
