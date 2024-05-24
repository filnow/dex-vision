#pragma once
#include <string>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>


class FastSAM
{
public:
    FastSAM() {};
    ~FastSAM(){};

    bool Initialize(const std::string& xml_path, float conf, float iou, bool useGpu);
    void Infer(const std::string &image_path, std::vector<cv::Point2f> cords = std::vector<cv::Point2f>());

    cv::Mat Render();
    cv::Mat RenderSingleMask(std::vector<cv::Point2f> cords);


private:
    std::vector<cv::Mat> Postprocess(const cv::Mat& oriImage);

    cv::Mat BuildOutput0();
    cv::Mat BuildOutput1();

    void ScaleBoxes(cv::Mat& box, const cv::Size& oriSize);

    std::vector<cv::Mat> ProcessMaskNative(const cv::Mat& oriImage, cv::Mat& protos, cv::Mat& masks_in, cv::Mat& bboxes, cv::Size shape);
    std::vector<cv::Mat> NMS(cv::Mat& prediction, int max_det = 300);

    void xywh2xyxy(cv::Mat &box);

    void ColorMask(const cv::Mat& mask, cv::Mat& rendered);

private:
    ov::Tensor Preprocess(cv::Mat& image);

    bool ConvertSize(cv::Mat& image);
    bool ConvertLayout(cv::Mat& image);

    ov::Tensor BuildTensor();

private:
    bool ParseArgs();
    bool BuildProcessor();
    bool IsGpuAvaliable(const ov::Core& core);

private:
    std::shared_ptr<ov::Model> m_model;
    ov::CompiledModel m_compiled_model;

    ov::Core m_core;
    ov::InferRequest m_request;
    std::shared_ptr<ov::preprocess::PrePostProcessor> m_ppp;

    float m_conf;
    float m_iou;

    std::vector<float> input_data;

    int input_width = 0;
    int input_height = 0;
    int input_channel = 3;
    ov::Shape model_input_shape;
    ov::Shape model_output0_shape;
    ov::Shape model_output1_shape;

    float ratio = 1.0f;
    float dw = 0.f;
    float dh = 0.f;

    int mw = 160;
    int mh = 160;

    cv::Mat m_image;
    std::vector<cv::Mat> result;
    cv::Mat image;
};
