# RepVGG_demo
This repository provides the code for implementation of Reparametrized VGG (RepVGG) network as proposed in [RepVGG: Making VGG Style ConvNets Great Again](https://openaccess.thecvf.com/content/CVPR2021/papers/Ding_RepVGG_Making_VGG-Style_ConvNets_Great_Again_CVPR_2021_paper.pdf).  

The RepVGG performs uptp 108x faster computations in comparison to the conventional VGG multibranch networks. The code provides a simple demo for comparing the conventional and reparametrized version of VGG style network. 

The same technique can be ported to other architectures as well. 

The code runs the model on RTX 3060 with different batch sizes and following is the result.  


![time](https://user-images.githubusercontent.com/26203136/183696318-563bd1c1-6ef7-4579-9c9a-1ecf30a44d06.png)
![vram](https://user-images.githubusercontent.com/26203136/183696344-0a89e3dd-5007-4086-8278-a2bc4d1f2cc8.png)

You can visualize the difference of the reparametrized version in comparison to its vanilla counterpart. The implementation used pytorch so make sure to install the required packages, accordingly.

