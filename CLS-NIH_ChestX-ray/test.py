import pdb
import os
import argparse

parser = argparse.ArgumentParser('MaCo pre-training', add_help=False)
parser.add_argument('--model', default='', type=str)
parser.add_argument('--last', default='maco', type=str)
parser.add_argument('--gpu', default=4, type=int)
parser.add_argument('--path', default="/path/to/CLS-NIH_ChestX-ray/finetuning_outputs/", type=str)

if __name__ == '__main__':
    args = parser.parse_args()
    gpu = str(args.gpu)
    name = "test"
    name_list = [str(args.model)]
    print('name_list:' + args.model)

    for model_name in name_list:
        pretrained_path = args.path + model_name + '/'
        pretrained_path += (args.last + '.bin')
        model_type = "ViT-B_16"
        print(os.system('CUDA_VISIBLE_DEVICES=' + gpu +' python3 train.py --name ' + name + ' --stage test --model_type ' + model_type +' --model vit_base_patch16 --num_classes 14 --pretrained_path ' + pretrained_path +' --eval_batch_size 512 --img_size 224 --fp16 --fp16_opt_level O2'))

