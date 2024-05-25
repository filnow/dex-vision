#ifndef OVLOAD_H
#define OVLOAD_H

#include <string>
#include <opencv2/opencv.hpp>

#include <opencv2/core.hpp>
#include <openvino/openvino.hpp>


class ovload
{
public:
    ovload();

    bool Initialize(const std::string& xml_path, float conf, float iou);
    void Infer(const std::string &image_path);

    cv::Mat Render();
};

#endif // OVLOAD_H
