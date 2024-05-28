# What is this

Basiclly a image editor, writen in C++ using Qt, OpenCV, OpenVINO, with a lot of AI models.
The goal is to make it usable on Intel CPU, all development is made on Linux.
Motivated by [dingboard](https://dingboard.com/) from [yacine](https://yacine.ca/).

## Progress

Right now the app is cappable of:

    - Loading image
    - Making full segmentation with [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM)
    - Segment object on click with [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM)
    - Fancy remove background
    - Depth map with [DepthAnything](https://github.com/LiheYoung/Depth-Anything)
    - 3D scanning effect on 2D image using depth map

Things to add:

    - Inpaint with [Lama](https://github.com/advimman/lama)
    - Saving images
    - Image upscale
    - Drag and drop
    - Drag and draw box then segmentation 
    - Removing objects
    - Adding objects with blending in
    - Multiple clicked masks
    - Diffrent rotation like mirror and stuff
    - Load multiple images 

## Demo

![Alt text](utils/demo.gif)



 
