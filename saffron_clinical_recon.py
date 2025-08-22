


import os
import sys


import torch

from saffron.svr.inference import svorm_predict
from saffron.tool import load_stack,Stack, Slice, Volume
import argparse
from argparse import Namespace
from typing import Dict, Tuple, Any,List
import os
import re


import logging

import pysitk.python_helper as ph


def inputs(args: Namespace,input_stacks,stack_masks,device):
    input_dict: Dict[str, Any] = dict()
    # if getattr(args, "input_stacks", None) is not None:
    input_dict["input_stacks"] = []
    # input_stacks = []
    # slices = []
    for i, f in enumerate(input_stacks):

        stack = load_stack(
            f,
            # stack_masks[i] if args.stack_masks is not None else None,
            stack_masks[i],
            # None,
            device=device,
        )

        input_dict["input_stacks"].append(stack)


        # idx_nonempty = stack.mask.flatten(1).any(1)
        # # print("size",stack.slices.size())
        # # stack.slices /= torch.max(stack.slices) * 0.98
        # stack.slices /= torch.quantile(stack.slices[stack.mask], 0.99)
        # slices.extend(stack[idx_nonempty])

    return input_dict, args

def main() -> Namespace:
    parser = argparse.ArgumentParser()



    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--debug", action="store_true", help="Debug mode.")
    parser.add_argument("--mask-threshold", type=float, default=1.0)
    parser.add_argument('--gpu', default='0', help='GPU ID number(s), comma-separated (default: 3)')
    args = parser.parse_args()
    return args


# 文件包含多种协议，方向为协议后三个
def find_files3(file_dir,seg_dir,out_dir):
    # logging.basicConfig(filename=file_dir + '/log.log',
    #                     format='%(asctime)s - %(name)s - %(levelname)s -%(module)s:  %(message)s',
    #                     datefmt='%Y-%m-%d %H:%M:%S %p',
    #                     level=10)
    ph.create_directory(os.path.join(file_dir + "/out"))
    # ph.create_directory(os.path.join(file_dir + "/out/slices"))
    log = ""
    files = {}
    files['files'] = []
    # files['mask_output'] = "/home/data/"+file_dir + "/seg/"
    files['mask_output'] = seg_dir
    files['masked_recon'] = file_dir + "/masked_recon/"
    # files["out"] = file_dir + "/out"
    files["out"] = out_dir
    files["crop"] = file_dir + "/out/crop"
    # ph.create_directory(files["crop"])
    # ph.create_directory(os.path.dirname(files['masked_recon']))
    imgs_dict = {}
    global global_path
    g = os.walk(file_dir)
    for path, dir_list, file_list in g:
        global_path = path

        # if "seg" in path or "out" in path or "masked_recon" in path  or "bestData" in path or "results" in path or "ini" in path:
        if "seg" in path or "out" in path or "masked_recon" in path  or "bestData" in path or "results" in path :
            continue
        for file_name in file_list:
            if file_name.endswith(".nii.gz") :
                # names = file_name.split('t2haste')
                # names = re.findall(r"\d+\_\d+",file_name)
                names = re.findall(r"20\d+", file_name)
                # print(file_name)
                # name = names[0]
                name = names[-1]
                if not name in imgs_dict.keys():
                    imgs = []
                    imgs.append(os.path.join(path,file_name))
                    # imgs.append(os.path.join("/home/data/"+path,file_name))
                    imgs_dict[name] = imgs
                else:
                    imgs_dict[name].append(os.path.join(path,file_name))
                    # imgs_dict[name].append(os.path.join("/home/data/"+path,file_name))
    for name,file_list in imgs_dict.items():
        print(name)
        print(file_list)
        file = {}
        file['img'] = []
        file['mask'] = []
        for file_name in file_list:
            name = file_name.split('/')[-1]
            # names = re.findall(r"\d+\_\d+",file_name)
            names = re.findall(r"20\d+", file_name)
            # file['name'] = names[0]
            file['name'] = names[-1]
            file['img'].append(file_name)
            file['mask'].append(os.path.join(files['mask_output'], name))
        print(file['img'])
        print(file['mask'])

        files['files'].append(file)
    log += file_dir + "共有" + str(len(files['files'])) + "个subject"
    print("共有" + str(len(files['files'])) + "个文件")
    print("log\n"+log)
    logging.info(log)
    return files

def reconstruct_batch(files,device,args):
    logging.basicConfig(filename=files["out"] + 'svorm.log',
                        format='%(asctime)s - %(name)s - %(levelname)s -%(module)s:  %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S %p',
                        level=10)
    print(files["out"] + 'svorm.log')
    time_start = ph.start_timing()
    total = len(files['files'])
    logs = ""
    svrtk_commands = []
    for i, f in enumerate(files['files']):
        # if i <= 40 or i >50:
        #     continue
        log = ""
        print("重建%s/%s " % (i + 1, total))
        log += "重建%s/%s" % (i + 1, total) + "\n"

        filenames = ""
        masks = ""
        filenames += " ".join(f['img'])
        masks += " ".join(f['mask'])


        out_dir = os.path.join(files["out"], "saffron")

        os.makedirs(out_dir, exist_ok=True)
        fname = os.path.join(out_dir,f['name'])
        input_stacks = f["img"]
        input_stacks_masks = f["mask"]

        input_dict, args = inputs(args, input_stacks=input_stacks, stack_masks=input_stacks_masks, device=device)
        stacks_out, volume = svorm_predict(dataset=input_dict["input_stacks"], checkpoint=args.checkpoint,
                                            device=device)

        volume.save(fname+".nii.gz")



if __name__ == "__main__":
    # parameters
    args = main()



    args.file_dir = "data/test"
    args.seg_dir = "data/test/seg"
    args.output = "data/test/out"


    args.checkpoint = "checkpoint/saffron_GZ_GFY.pt"


    args.gpu = "0"
    device = torch.device('cuda:{}'.format(args.gpu) if args.gpu is not None else 'cpu')
    files = find_files3(args.file_dir,args.seg_dir,args.output)
    reconstruct_batch(files,device,args)