# Progressive Growing of GANs for Improved Quality, Stability, and Variation


-------

[NOTE]  This project was not goint well, so I made PyTorch implementation here. :fire: [[pggan-pytorch]](https://github.com/nashory/pggan-pytorch)

-------


  
Torch implementation of [PROGRESSIVE GROWING OF GANS FOR IMPROVED QUALITY, STABILITY, AND VARIATION](http://research.nvidia.com/sites/default/files/pubs/2017-10_Progressive-Growing-of//karras2017gan-paper.pdf)   
__YOUR CONTRIBUTION IS INVALUABLE FOR THIS PROJECT :)__ 

![image](https://puu.sh/ydG0E/e0f32b0d92.png)

## NEED HELP

[ ] (1) Implementing Pixel-wise normalization layer  
[ ] (2) Implementing pre-layer normalization (for equalized learning rate)    
__(I have tried both, but failed to converge. Anyone can help implementing those two custom layers?)__

## Prerequisites
+ [Torch7](http://torch.ch/docs/getting-started.html#_)
+ [display](https://github.com/szym/display)
+ [nninit](https://github.com/Kaixhin/nninit)


## How to use?

__[step 1.] Prepare dataset__   
CelebA-HQ dataset is not available yet, so I used 100,000 generated PNGs of CelebA-HQ released by the author.   
The quality of the generated image was good enough for training and verifying the preformance of the code.  
If the CelebA-HQ dataset is releasted in then near future, I will update the experimental result.  
[[download]](https://drive.google.com/open?id=0B4qLcYyJmiz0MUVMVFEyclJnRmc)

+ CAUTION: loading 1024 x 1024 image and resizing every forward process makes training slow. I recommend you to use normal CelebA dataset until the output resolution converges to 256x256.

~~~
---------------------------------------------
The training data folder should look like : 
<train_data_root>
                |--classA
                        |--image1A
                        |--image2A ...
                |--classB
                        |--image1B
                        |--image2B ...
---------------------------------------------
~~~

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
![image](https://puu.sh/ydFYx/46cb67da75.png)  

### Transition experiment: (having trouble with transition from 8x8 -> 16x16 yet.)
<img src="https://github.com/nashory/gifs/blob/progressive/resl_4.gif?raw=true" width="430" height="430"> <img src="https://github.com/nashory/gifs/blob/progressive/resl_4to8.gif?raw=true" width="430" height="430">

## What does the printed log mean?
~~~
(example)
[E:0][T:91][ 91872/202599]    errD(real): 0.2820 | errD(fake): 0.1557 | errG: 0.3838    [Res:   4][Trn(G):0.0%][Trn(D):0.0%][Elp(hr):0.2008]
~~~
+ E: epoch / T: ticks (1tick = 1000imgs) / errD,errG: loss of discrminator and generator
+ Res: current resolution of output
+ Trn: transition progress (if 100%, in training phase. if less than 100%, in transition phase using fade-in layer.)
  + first Trn : Transition of fade-in layer in generator.
  + second Trn : Transition of fade-in layer in discriminator.
+ Elp(hr): Elapsed Time (Hour)



## To-Do List (will be implemented soon)
- [X] Equalized learning rate (weight normalization)
- [X] Support WGAN-GP loss

## Compatability
+ cuda v8.0
+ Tesla P40 (you may need more than 12GB Memory. If not, please adjust the batch_table in `pggan.lua`)
+ python 2.7 / Torch7

## Acknowledgement
+ [tkarras/progressive_growing_of_gans](https://github.com/tkarras/progressive_growing_of_gans)
+ [soumith/dcgan.torch](https://github.com/soumith/dcgan.torch)

## Author
MinchulShin, [@nashory](https://github.com/nashory)  
![image](https://camo.githubusercontent.com/e053bc3e1b63635239e8a44574e819e62ab3e3f4/687474703a2f2f692e67697068792e636f6d2f49634a366e36564a4e6a524e532e676966)
