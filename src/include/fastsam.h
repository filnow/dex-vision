#ifndef FASTSAM_H
#define FASTSAM_H

#include "ovload.h"
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>


class FastSAM : public OVload
{
public:
    FastSAM() : OVload() {};
    ~FastSAM() {};

    bool Initialize(const std::string& xml_path, int input_number, int output_number) {
        return OVload::Initialize(xml_path, input_number, output_number);
    }

    void Infer(const std::string &image_path) {
        OVload::Infer(image_path);
    }

protected:
    void xywh2xyxy(cv::Mat &box);
    void ScaleBoxes(cv::Mat& box, const cv::Size& oriSize);

    bool ConvertSize(cv::Mat& image);

    std::vector<cv::Mat> NMS(cv::Mat& prediction, int max_det = 300);
    std::vector<cv::Mat> ProcessMaskNative(const cv::Mat& oriImage, cv::Mat& protos, cv::Mat& masks_in, cv::Mat& bboxes, cv::Size shape);

    ov::Tensor Preprocess(cv::Mat& image) override {
        if(!ConvertSize(image)) {
            return ov::Tensor();
        }

        if(!ConvertLayout(image)) {
            return ov::Tensor();
        }

        return BuildTensor();
    }

    std::vector<cv::Mat> Postprocess(cv::Mat& image) override {
        std::vector<cv::Mat> builded_output = BuildOutput();
        cv::Mat prediction = builded_output[0];
        cv::Mat proto = builded_output[1];

        std::vector<cv::Mat> remat = NMS(prediction, 100);

        if(remat.size() < 2) {
            return std::vector<cv::Mat>();
        }

        cv::Mat box = remat[0];
        cv::Mat mask = remat[1];
        ScaleBoxes(box, image.size());

        return ProcessMaskNative(image, proto, mask, box, image.size());
    }

protected:
    float dw = 0.f;
    float dh = 0.f;
    float ratio = 1.0f;
};

#endif // FASTSAM_H
