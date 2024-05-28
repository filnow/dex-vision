# What is this

Basically a image editor, written in C++ using Qt, OpenCV, OpenVINO, with a lot of AI models.
The goal is to make it usable on Intel CPU, all development is made on Linux.
Motivated by [dingboard](https://dingboard.com/) from [yacine](https://yacine.ca/).

## Progress

Right now the app is capable of:

    - Loading image
    - Making full segmentation with FastSAM
    - Segmenting object on click with FastSAM
    - Fancy remove background
    - Depth map with DepthAnything
    - 3D scanning effect on 2D image using depth map

Things to add:

    - Inpainting with Lama
    - Saving images
    - Image upscaling
    - Drag and drop
    - Drag and draw box then segmentation 
    - Removing objects
    - Adding objects with blending in
    - Multiple clicked masks
    - Different rotation like mirror and stuff
    - Loading multiple images

## Demo

![Alt text](utils/demo.gif)



 
