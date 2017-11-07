# python script to run GAN algorithms.

import os, sys
import json
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--type', default='pggan', help='Progressive-growing GAN')
parser.add_argument('--multi', default=False, help='true: if you want to prevent memory allocation across all GPUs')
args = parser.parse_args()
params = vars(args)
print json.dumps(params, indent = 4)



if params['multi'] : 
    os.system('CUDA_VISIBLE_DEVICES=4 th script/main.lua')
else:
    os.system('th script/main.lua')


