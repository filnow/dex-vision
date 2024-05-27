#ifndef OVLOAD_H
#define OVLOAD_H

#include <string>
#include <opencv2/opencv.hpp>

#include <opencv2/core.hpp>
#include <openvino/openvino.hpp>


class OVload
{
public:
    OVload();

    cv::Mat Render();

    bool Initialize(const std::string& xml_path, float conf, float iou);
    void Infer(const std::string &image_path);

private:
    cv::Mat BuildOutput();

    ov::Tensor BuildTensor();
    ov::Tensor Preprocess(cv::Mat& image);

    bool ConvertSize(cv::Mat& image);
    bool ConvertLayout(cv::Mat& image);
    bool ParseArgs();
    bool BuildProcessor();

private:
    cv::Mat result;
    cv::Mat image;

    ov::Core m_core;
    ov::InferRequest m_request;
    ov::Shape model_input_shape;
    ov::Shape model_output_shape;
    ov::CompiledModel m_compiled_model;

    std::vector<float> input_data;
    std::shared_ptr<ov::Model> m_model;
    std::shared_ptr<ov::preprocess::PrePostProcessor> m_ppp;

    int mw = 160;
    int mh = 160;
    int input_width = 0;
    int input_height = 0;
    int input_channel = 3;
};

#endif // OVLOAD_H
