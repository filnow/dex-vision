#ifndef DEPTH_H
#define DEPTH_H

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>


class Depth
{
public:
    Depth() {};
    ~Depth(){};

    bool Initialize(const std::string& xml_path);
    void Infer(const std::string &image_path);

    cv::Mat RenderDepth();

private:
    bool ParseArgs();
    bool BuildProcessor();
    bool ConvertLayout(cv::Mat& image);

    ov::Tensor Preprocess(cv::Mat& image);
    std::vector<cv::Mat> Postprocess(const cv::Mat& oriImage);

    cv::Mat BuildOutput();
    ov::Tensor BuildTensor();

private:
    ov::Core m_core;
    ov::CompiledModel m_compiled_model;
    ov::InferRequest m_request;
    ov::Shape model_input_shape;
    ov::Shape model_output_shape;

    std::shared_ptr<ov::preprocess::PrePostProcessor> m_ppp;
    std::shared_ptr<ov::Model> m_model;
    std::vector<float> input_data;
    cv::Mat result;

    cv::Mat image;

    int input_width = 0;
    int input_height = 0;
    int input_channel = 3;
    int mw = 518;
    int mh = 518;
};

#endif // DEPTH_H
