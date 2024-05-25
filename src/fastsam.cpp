#include <algorithm>
#include <filesystem>

#include <QDebug>

#include "fastsam.h"


cv::Scalar RandomColor()
{
    int b = rand() % 256;
    int g = rand() % 256;
    int r = rand() % 256;
    return cv::Scalar(b, g, r);
}

bool FastSAM::Initialize(const std::string &xml_path, float conf, float iou)
{
    m_conf = conf;
    m_iou = iou;

    if(!std::filesystem::exists(xml_path))
        return false;

    m_model = m_core.read_model(xml_path);

    if(!ParseArgs())
        return false;


    if(!BuildProcessor())
        return false;

    m_compiled_model = m_core.compile_model(m_model, "CPU");

    m_request = m_compiled_model.create_infer_request();

    qDebug() << "Init sucessfull\n";

    return true;
}


void FastSAM::Infer(const std::string &image_path)
{
    try {
        image = cv::imread(image_path);
        cv::Mat processedImg = image.clone();
        ov::Tensor input_tensor = Preprocess(processedImg);

        assert(input_tensor.get_size() != 0);

        m_request.set_input_tensor(input_tensor);
        m_request.infer();

        result =  Postprocess(image);
    }
    catch (std::exception& e) {
        qDebug() << "Failed to Infer! ec: " << e.what() << '\n';
    }
}

bool FastSAM::ParseArgs()
{
    try {
        model_input_shape = m_model->input().get_shape();
        model_output0_shape = m_model->output(0).get_shape();
        model_output1_shape = m_model->output(1).get_shape();

        qDebug()  << "xml input shape:" << model_input_shape << "\n";
        qDebug() << "xml output shape 0:" << model_output0_shape << "\n";
        qDebug() << "xml output shape 1:" << model_output1_shape << "\n";

        // [1, 3, 640, 640]
        input_channel = model_input_shape[1];
        input_height = model_input_shape[2];
        input_width = model_input_shape[3];

        this->input_data.resize(input_channel * input_height * input_height);

        qDebug() << "model input height:" << input_height << " input width:" << input_width << "\n";

        // output0 = [1,37,8400]
        // output1 = [1,32,160,160]
        mh = model_output1_shape[2];
        mw = model_output1_shape[3];

        qDebug() << "model output mh:" << mh << " output mw:" << mw << "\n";

        return true;
    }
    catch(const std::exception& e) {
        qDebug() << "Failed to Parse Args. "<< e.what() << '\n';
        return false;
    }

}

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

cv::Mat FastSAM::Render()
{
    cv::Mat rendered = image.clone();

    for (const auto& mask : result) {
        ColorMask(mask, rendered);
    }

    return rendered;
}

cv::Mat FastSAM::RenderSingleMask(std::vector<cv::Point2f> cords)
{
    cv::Mat rendered = image.clone();

    for (const auto& mask: result) {
        for (int j = 0; j < cords.size(); j++) {
            if (mask.at<float>(cords[j].y, cords[j].x) == 1.0) {
                ColorMask(mask, rendered);
            }
        }

    }

    return rendered;
}

void FastSAM::ColorMask(const cv::Mat& mask, cv::Mat& rendered) {
    auto color = RandomColor();
    for (int y = 0; y < mask.rows; y++) {
        const float* mp = mask.ptr<float>(y);
        uchar* p = rendered.ptr<uchar>(y);
        for (int x = 0; x < mask.cols; x++) {
            if (mp[x] == 1.0) {
                p[0] = cv::saturate_cast<uchar>(p[0] * 0.5 + color[0] * 0.5);
                p[1] = cv::saturate_cast<uchar>(p[1] * 0.5 + color[1] * 0.5);
                p[2] = cv::saturate_cast<uchar>(p[2] * 0.5 + color[2] * 0.5);
            }
            p += 3;
        }
    }
}

ov::Tensor FastSAM::Preprocess(cv::Mat &image)
{
    if(!ConvertSize(image)) {
        qDebug() << "failed to Convert Size!\n";
        return ov::Tensor();
    }

    if(!ConvertLayout(image)) {
        qDebug() << "Failed to Convert Layout!\n";
        return ov::Tensor();
    }

    return BuildTensor();
}

std::vector<cv::Mat> FastSAM::Postprocess(const cv::Mat& oriImage)
{
    cv::Mat prediction = BuildOutput0();
    cv::Mat proto = BuildOutput1();

    std::vector<cv::Mat> remat = NMS(prediction, 100);

    if(remat.size() < 2) {
        qDebug() << "Empty data after nms!\n";
        return std::vector<cv::Mat>();
    }

    cv::Mat box = remat[0];
    cv::Mat mask = remat[1];
    ScaleBoxes(box, oriImage.size());

    return ProcessMaskNative(oriImage, proto, mask, box, oriImage.size());
}

cv::Mat FastSAM::BuildOutput0()
{
    auto* ptr = m_request.get_output_tensor(0).data();
    return cv::Mat(model_output0_shape[1], model_output0_shape[2], CV_32F, ptr);
}

cv::Mat FastSAM::BuildOutput1()
{
    auto* ptr = m_request.get_output_tensor(1).data();
    return cv::Mat(model_output1_shape[1], model_output1_shape[2] * model_output1_shape[3], CV_32F, ptr);
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

bool FastSAM::ConvertLayout(cv::Mat &image)
{
    int row = image.rows;
    int col = image.cols;
    int channels = image.channels();

    if(channels != 3)
        return false;

    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    for (int c = 0; c < channels; c++) {
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                float pix = image.at<cv::Vec3b>(i, j)[c];
                input_data[c * row * col + i * col + j] = pix / 255.0;

            }
        }
    }

    return true;
}

ov::Tensor FastSAM::BuildTensor()
{
    ov::Shape shape = {1, static_cast<unsigned long>(input_channel), static_cast<unsigned long>(input_height), static_cast<unsigned long>(input_width)};

    return ov::Tensor(ov::element::f32, shape, input_data.data());;
}

bool FastSAM::BuildProcessor()
{
    try
    {
        m_ppp = std::make_shared<ov::preprocess::PrePostProcessor>(m_model);
        m_ppp->input().tensor()
            .set_shape({1, input_channel, input_height, input_width})
            .set_element_type(ov::element::f32)
            .set_layout("NCHW")
            .set_color_format(ov::preprocess::ColorFormat::RGB);

        m_model = m_ppp->build();
    }
    catch(const std::exception& e)
    {
        std::cerr << "Failed to build the model processor!\n" << e.what() << '\n';
        return false;
    }

    qDebug() << "Build successfully!\n";
    return true;
}




