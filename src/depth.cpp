#include "depth.h"
#include <filesystem>

#include <QDebug>


bool Depth::Initialize(const std::string &xml_path)
{
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

void Depth::Infer(const std::string &image_path)
{
    try {
        image = cv::imread(image_path);
        float h = image.rows;
        float w = image.cols;
        cv::Mat processedImg = image.clone();
        ov::Tensor input_tensor = Preprocess(processedImg);

        assert(input_tensor.get_size() != 0);

        m_request.set_input_tensor(input_tensor);
        m_request.infer();

        cv::Mat depth = BuildOutput();

        cv::resize(depth, depth, cv::Size(w, h));

        // Normalize the depth values to the range [0, 255]
        double minVal, maxVal;
        cv::minMaxLoc(depth, &minVal, &maxVal);
        depth = (depth - minVal) / (maxVal - minVal) * 255.0;
        depth.convertTo(depth, CV_8U);

        // Apply a colormap to the depth map
        cv::Mat depth_color;
        cv::applyColorMap(depth, depth_color, cv::COLORMAP_INFERNO);

        result = depth_color;
    }
    catch (std::exception& e) {
        qDebug() << "Failed to Infer! ec: " << e.what() << '\n';
    }
}

cv::Mat Depth::RenderDepth()
{
    return result;
}


bool Depth::ParseArgs()
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

ov::Tensor Depth::Preprocess(cv::Mat &image)
{
    cv::resize(image, image, cv::Size(518, 518));

    if(!ConvertLayout(image)) {
        qDebug() << "Failed to Convert Layout!\n";
        return ov::Tensor();
    }

    return BuildTensor();
}

bool Depth::BuildProcessor()
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

cv::Mat Depth::BuildOutput()
{
    auto* ptr = m_request.get_output_tensor(0).data();
    return cv::Mat(model_output_shape[1], model_output_shape[2], CV_32F, ptr);
}

bool Depth::ConvertLayout(cv::Mat &image)
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

ov::Tensor Depth::BuildTensor()
{
    ov::Shape shape = {1, static_cast<unsigned long>(input_channel), static_cast<unsigned long>(input_height), static_cast<unsigned long>(input_width)};

    return ov::Tensor(ov::element::f32, shape, input_data.data());;
}
