#ifndef INPAINT_H
#define INPAINT_H

#include <string>
#include <opencv2/opencv.hpp>

#include <opencv2/core.hpp>
#include <openvino/openvino.hpp>


class Inpaint
{
public:
    Inpaint();

    cv::Mat Render();
    std::vector<cv::Mat> GetResult();

    void Infer();
    bool Initialize(const std::string &xml_path, int input_number, int output_number);

protected:
    ov::Tensor BuildTensor();

    std::vector<cv::Mat> BuildOutput();

    ov::Tensor Preprocess(cv::Mat &image);
    std::vector<cv::Mat> Postprocess(cv::Mat &image);

    bool ConvertSize(cv::Mat& image);
    bool ConvertLayout(cv::Mat& image);
    bool ParseArgs(int input_number, int output_number);
    bool BuildProcessor();

protected:
    cv::Mat image;
    std::vector<cv::Mat> result;

    ov::Core m_core;
    ov::InferRequest m_request;
    ov::CompiledModel m_compiled_model;

    std::vector<float> input_data;
    std::shared_ptr<ov::Model> m_model;
    std::vector<ov::Shape> model_input_shape;
    std::vector<ov::Shape> model_output_shape;
    std::shared_ptr<ov::preprocess::PrePostProcessor> m_ppp;

    int mw = 160;
    int mh = 160;
    int input_width = 0;
    int input_height = 0;
    int input_channel = 3;
};

#endif // INPAINT_H
