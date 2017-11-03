# progressive-growing-torch
__YOUR CONTRIBUTION IS INVALUABLE FOR THIS PROJECT :)__   
Torch implementation of [PROGRESSIVE GROWING OF GANS FOR IMPROVED QUALITY, STABILITY, AND VARIATION](http://research.nvidia.com/sites/default/files/pubs/2017-10_Progressive-Growing-of//karras2017gan-paper.pdf)

![image](https://puu.sh/ydG0E/e0f32b0d92.png)


## To-Do List
- [X] Equalized learning rate (weight normalization)
- [X] Pixel-wise norm
- [X] Support WGAN-GP loss


## Prerequisites
+ [Torch7](http://torch.ch/docs/getting-started.html#_)
+ [display](https://github.com/szym/display)
+ [nninit](https://github.com/Kaixhin/nninit)


## How to use?

__[step 1.] Preapre dataset__ 

CelebA-HQ dataset is not available yet, so I used 100,000 generated PNGs of CelebA-HQ released by the author.   
The quality of the generated image was extremely good enough for training and verifying the preformance of the code. If the CelebA-HQ dataset is releasted in then near future, I will update the experimental result.  
[[download]](https://drive.google.com/open?id=0B4qLcYyJmiz0MUVMVFEyclJnRmc)

__[step 2.] Run training__ 
  + edit script/opts.lua to change training parameter. (don't forget to change path to training images)
  + run and enjoy!  (Multi-threaded dataloading is supported.)
  `$ python run.py`

__[step 3.] Visualization__
  
  + to start display server:  
  `$ th server.lua`
  
  + to check images during training procudure:  
  `$ <server_ip>:<port> at your browser`


## Experimental results
results so far (just started training. I will keep updating the result.)  
![image](https://puu.sh/ydFYx/46cb67da75.png)

## Compatability
+ cuda v8.0
+ Tesla P40 (you may need more than 12GB Memory. If not, please adjust the batch_table in `pggan.lua`)
+ python 2.7 / Torch7

## Acknowledgement
+ [tkarras/progressive_growing_of_gans](https://github.com/tkarras/progressive_growing_of_gans)
+ [soumith/dcgan.torch](https://github.com/soumith/dcgan.torch)

## Author
MinchulShin, [@nashory](https://github.com/nashory)
