#include "ovload.h"
#include "QDebug"
#include <filesystem>


OVload::OVload() {}


bool OVload::Initialize(const std::string &xml_path, int input_number, int output_number)
{
    if(!std::filesystem::exists(xml_path))
        return false;

    m_model = m_core.read_model(xml_path);

    if(!ParseArgs(input_number, output_number))
        return false;


    if(!BuildProcessor())
        return false;

    m_compiled_model = m_core.compile_model(m_model, "CPU");

    m_request = m_compiled_model.create_infer_request();

    qDebug() << "Init sucessfull\n";

    return true;
}


void OVload::Infer(const std::string &image_path)
{
    try {
        image = cv::imread(image_path);
        cv::Mat processedImg = image.clone();
        ov::Tensor input_tensor = Preprocess(processedImg);

        assert(input_tensor.get_size() != 0);

        m_request.set_input_tensor(input_tensor);
        m_request.infer();

        result = Postprocess(image);
    }
    catch (std::exception& e) {
        qDebug() << "Failed to Infer! ec: " << e.what() << '\n';
    }
}


std::vector<cv::Mat> OVload::GetResult()
{
    return result;
}


bool OVload::ParseArgs(int input_number, int output_number)
{
    try {
        size_t num_inputs = m_model->inputs().size();
        size_t num_outputs = m_model->outputs().size();

        for (size_t i = 0; i < num_inputs; i++) {
            model_input_shape.push_back(m_model->input(i).get_shape());
        }

        for (size_t i = 0; i < num_outputs; i++) {
            model_output_shape.push_back(m_model->output(i).get_shape());
        }

        input_channel = model_input_shape[input_number-1][1];
        input_height = model_input_shape[input_number-1][2];
        input_width = model_input_shape[input_number-1][3];

        this->input_data.resize(input_channel * input_height * input_height);

        qDebug() << "model input height:" << input_height << " input width:" << input_width << "\n";

        mh = model_output_shape[output_number-1][-2];
        mw = model_output_shape[output_number-1][-1];

        qDebug() << "model output mh:" << mh << " output mw:" << mw << "\n";

        return true;
    }
    catch(const std::exception& e) {
        qDebug() << "Failed to Parse Args. "<< e.what() << '\n';
        return false;
    }

}


bool OVload::BuildProcessor()
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


std::vector<cv::Mat> OVload::BuildOutput()
{
    size_t num_outputs = m_model->outputs().size();
    std::vector<cv::Mat> build_outputs;

    for (size_t i = 0; i < num_outputs; i++) {
        auto* ptr = m_request.get_output_tensor(i).data();
        if (i == 1) {
            build_outputs.push_back(cv::Mat(model_output_shape[i][1], model_output_shape[i][2] * model_output_shape[i][3], CV_32F, ptr));
        } else {
            build_outputs.push_back(cv::Mat(model_output_shape[i][1], model_output_shape[i][2], CV_32F, ptr));
        }
    }

    return build_outputs;
}


ov::Tensor OVload::BuildTensor()
{
    ov::Shape shape = {1, static_cast<unsigned long>(input_channel), static_cast<unsigned long>(input_height), static_cast<unsigned long>(input_width)};

    return ov::Tensor(ov::element::f32, shape, input_data.data());;
}


bool OVload::ConvertLayout(cv::Mat &image)
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

