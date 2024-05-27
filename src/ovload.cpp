#include "ovload.h"
#include "QDebug"

OVload::OVload() {}



bool OVload::ParseArgs()
{
    try {
        model_input_shape = m_model->input().get_shape();
        model_output_shape = m_model->output(0).get_shape();

        qDebug()  << "xml input shape:" << model_input_shape << "\n";
        qDebug() << "xml output shape 0:" << model_output_shape << "\n";

        // [1, 3, 518, 518]
        input_channel = model_input_shape[1];
        input_height = model_input_shape[2];
        input_width = model_input_shape[3];

        this->input_data.resize(input_channel * input_height * input_height);

        qDebug() << "model input height:" << input_height << " input width:" << input_width << "\n";

        // output = [1,518,518]
        mh = model_output_shape[1];
        mw = model_output_shape[2];

        qDebug() << "model output mh:" << mh << " output mw:" << mw << "\n";

        return true;
    }
    catch(const std::exception& e) {
        qDebug() << "Failed to Parse Args. "<< e.what() << '\n';
        return false;
    }
}

cv::Mat OVload::BuildOutput()
{
    auto* ptr = m_request.get_output_tensor(0).data();
    return cv::Mat(model_output_shape[1], model_output_shape[2], CV_32F, ptr);
}

ov::Tensor OVload::BuildTensor()
{
    ov::Shape shape = {1, static_cast<unsigned long>(input_channel), static_cast<unsigned long>(input_height), static_cast<unsigned long>(input_width)};

    return ov::Tensor(ov::element::f32, shape, input_data.data());;
}
