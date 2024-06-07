#include <QDebug>
#include <algorithm>

#include "fastsam.h"


void FastSAM::ScaleBoxes(cv::Mat &box, const cv::Size& oriSize)
{
    float oriWidth = static_cast<float>(oriSize.width);
    float oriHeight = static_cast<float>(oriSize.height);
    float *pxvec = box.ptr<float>(0);

    for (int i = 0; i < box.rows; i++) {
        pxvec = box.ptr<float>(i);
        pxvec[0] -= this->dw;
        pxvec[0] = std::clamp(pxvec[0] * this->ratio, 0.f, oriWidth);
        pxvec[1] -= this->dh;
        pxvec[1] = std::clamp(pxvec[1] * this->ratio, 0.f, oriHeight);
        pxvec[2] -= this->dw;
        pxvec[2] = std::clamp(pxvec[2] * this->ratio, 0.f, oriWidth);
        pxvec[3] -= this->dh;
        pxvec[3] = std::clamp(pxvec[3] * this->ratio, 0.f, oriHeight);
    }
}


std::vector<cv::Mat> FastSAM::ProcessMaskNative(const cv::Mat &image, cv::Mat &protos, cv::Mat &masks_in, cv::Mat &bboxes, cv::Size shape)
{
    std::vector<cv::Mat> result;
    //result.push_back(bboxes);  //

    cv::Mat matmulRes = (masks_in * protos).t();

    cv::Mat maskMat = matmulRes.reshape(bboxes.rows, {mh, mw});  // shape [bboxes.rows, 160, 160]

    std::vector<cv::Mat> maskChannels;
    cv::split(maskMat, maskChannels);
    int scale_dw = this->dw / input_width * mw;
    int scale_dh = this->dh / input_height * mh;
    cv::Rect roi(scale_dw, scale_dh, mw - 2 * scale_dw, mh - 2 * scale_dh);
    float *pxvec = bboxes.ptr<float>(0);
    for (int i = 0; i < bboxes.rows; i++) {
        pxvec = bboxes.ptr<float>(i);
        cv::Mat dest, mask;
        cv::exp(-maskChannels[i], dest);
        dest = 1.0 / (1.0 + dest);
        dest = dest(roi);
        cv::resize(dest, mask, image.size(), cv::INTER_LINEAR);
        cv::Rect roi(pxvec[0], pxvec[1], pxvec[2] - pxvec[0], pxvec[3] - pxvec[1]);
        cv::Mat temmask = mask(roi);
        cv::Mat boxMask = cv::Mat(image.size(), mask.type(), cv::Scalar(0.0));
        float rx = std::max(pxvec[0], 0.0f);
        float ry = std::max(pxvec[1], 0.0f);
        for (int y = ry, my = 0; my < temmask.rows; y++, my++) {
            float *ptemmask = temmask.ptr<float>(my);
            float *pboxmask = boxMask.ptr<float>(y);
            for (int x = rx, mx = 0; mx < temmask.cols; x++, mx++) {
                pboxmask[x] = ptemmask[mx] > 0.5 ? 1.0 : 0.0;
            }
        }
        result.push_back(boxMask);
    }

    return result;
}


std::vector<cv::Mat> FastSAM::NMS(cv::Mat &prediction, int max_det)
{
    float m_conf = 0.6;
    float m_iou = 0.9;

    std::vector<cv::Mat> vreMat;
    cv::Mat temData = cv::Mat();
    prediction = prediction.t(); // [37, 8400] --> [rows:8400, cols:37]
    float *pxvec = prediction.ptr<float>(0);

    for (int i = 0; i < prediction.rows; i++) {
        pxvec = prediction.ptr<float>(i);
        if (pxvec[4] > m_conf) {
            temData.push_back(prediction.rowRange(i, i + 1).clone());
        }
    }

    if (temData.rows == 0) {
        return vreMat;
    }

    cv::Mat box = temData.colRange(0, 4).clone();
    cv::Mat cls = temData.colRange(4, 5).clone();
    cv::Mat mask = temData.colRange(5, temData.cols).clone();

    cv::Mat j = cv::Mat::zeros(cls.size(), CV_32F);
    cv::Mat dst;
    cv::hconcat(box, cls, dst);
    cv::hconcat(dst, j, dst);
    cv::hconcat(dst, mask, dst); // dst = [box class j mask]

    std::vector<float> scores;
    std::vector<cv::Rect> boxes;
    pxvec = dst.ptr<float>(0);
    for (int i = 0; i < dst.rows; i++) {
        pxvec = dst.ptr<float>(i);
        boxes.push_back(cv::Rect(pxvec[0], pxvec[1], pxvec[2], pxvec[3]));
        scores.push_back(pxvec[4]);
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, m_conf, m_iou, indices);
    cv::Mat reMat;
    for (int i = 0; i < indices.size() && i < max_det; i++) {
        int index = indices[i];
        reMat.push_back(dst.rowRange(index, index + 1).clone());
    }
    box = reMat.colRange(0, 6).clone();
    xywh2xyxy(box);
    mask = reMat.colRange(6, reMat.cols).clone();

    vreMat.push_back(box);
    vreMat.push_back(mask);

    return vreMat;
}


void FastSAM::xywh2xyxy(cv::Mat &box)
{
    float *pxvec = box.ptr<float>(0);
    for (int i = 0; i < box.rows; i++) {
        pxvec = box.ptr<float>(i);
        float w = pxvec[2];
        float h = pxvec[3];
        float cx = pxvec[0];
        float cy = pxvec[1];
        pxvec[0] = cx - w / 2;
        pxvec[1] = cy - h / 2;
        pxvec[2] = cx + w / 2;
        pxvec[3] = cy + h / 2;
    }
}


bool FastSAM::ConvertSize(cv::Mat &image)
{
    float height = static_cast<float>(image.rows);
    float width = static_cast<float>(image.cols);

    float r = std::min(input_height / height, input_width / width);
    int padw = static_cast<int>(std::round(width * r));
    int padh = static_cast<int>(std::round(height * r));

    if((int)width != padw || (int)height != padh)
        cv::resize(image, image, cv::Size(padw, padh));

    float _dw = (input_width - padw) / 2.f;
    float _dh = (input_height - padh) / 2.f;

    int top =  static_cast<int>(std::round(_dh - 0.1f));
    int bottom = static_cast<int>(std::round(_dh + 0.1f));
    int left = static_cast<int>(std::round(_dw - 0.1f));
    int right = static_cast<int>(std::round(_dw + 0.1f));
    cv::copyMakeBorder(image, image, top, bottom, left, right, cv::BORDER_CONSTANT,
                       cv::Scalar(114, 114, 114));

    this->ratio = 1 / r;
    this->dw = _dw;
    this->dh = _dh;

    return true;
}





