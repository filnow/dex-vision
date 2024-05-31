#ifndef DEPTH_H
#define DEPTH_H

#include "ovload.h"
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>


class Depth : public OVload
{
public:
    Depth() : OVload() {};
    ~Depth() {};

    bool Initialize(const std::string& xml_path, int input_number, int output_number) {
        return OVload::Initialize(xml_path, input_number, output_number);
    }

    void Infer(const std::string &image_path) {
        OVload::Infer(image_path);
    }

    cv::Mat RenderDepth() {
        return Render();
    }

protected:
    ov::Tensor Preprocess(cv::Mat& image) override {
        cv::resize(image, image, cv::Size(518, 518));

        if(!ConvertLayout(image)) {
            return ov::Tensor();
        }
        return BuildTensor();
    }

    std::vector<cv::Mat> Postprocess(cv::Mat& image) override {
        auto output = BuildOutput();
        cv::Mat depth = output[0];
        cv::resize(depth, depth, cv::Size(image.cols, image.rows));

        double minVal, maxVal;
        cv::minMaxLoc(depth, &minVal, &maxVal);

        depth = (depth - minVal) / (maxVal - minVal) * 255.0;
        depth.convertTo(depth, CV_8U);

        cv::Mat depth_color;
        cv::applyColorMap(depth, depth_color, cv::COLORMAP_INFERNO);

        return {depth_color};
    }
};

#endif // DEPTH_H
