import pdb
import os
import argparse

parser = argparse.ArgumentParser('MaCo pre-training', add_help=False)
parser.add_argument('--model', default='', type=str)
parser.add_argument('--gpu', default=1, type=int)

if __name__ == '__main__':
    args = parser.parse_args()
    gpu = str(args.gpu)
    name = "test"
    list = ['30']
    name_list = [args.model]
    print('name_list:' + args.model)

    for model_name in name_list:
        for i in list:
            pretrained_path = "./CLS-NIH_ChestX-ray/finetuning_outputs/" + model_name + '/'
            pretrained_path += (i + '.bin')
            model_type = "ViT-B_16"
            print(os.system('CUDA_VISIBLE_DEVICES=' + gpu +' python3 train.py --name ' + name + ' --stage test --model_type ' + model_type +' --model vit_base_patch16 --num_classes 14 --pretrained_path ' + pretrained_path +' --eval_batch_size 512 --img_size 224 --fp16 --fp16_opt_level O2'))

