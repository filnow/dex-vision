#include "inpaint.h"
#include "QDebug"
#include <filesystem>


Inpaint::Inpaint() {}


bool Inpaint::Initialize(const std::string &xml_path, int input_number, int output_number)
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

std::vector<cv::Mat> Inpaint::Postprocess(cv::Mat &image) {
    auto output = BuildOutput();
    cv::Mat depth = output[0];
    return depth;
}

ov::Tensor Inpaint::Preprocess(cv::Mat &image) {
    cv::resize(image, image, cv::Size(518, 518));

    if(!ConvertLayout(image)) {
        return ov::Tensor();
    }
    return BuildTensor();
}

void Inpaint::Infer()
{
    try {
        cv::Mat image_to_inapint = cv::imread("/home/filnow/Desktop/image.jpg");
        cv::Mat mask = cv::imread("/home/filnow/Desktop/mask.jpg", cv::IMREAD_GRAYSCALE);

        cv::resize(image_to_inapint, image_to_inapint, cv::Size(518, 518));
        cv::resize(mask, mask, cv::Size(518, 518));


        cv::cvtColor(image_to_inapint, image_to_inapint, cv::COLOR_BGR2RGB);


        std::vector<float> image_data;
        image_data.resize(3 * 512 * 512);

        for (int c = 0; c < 3; c++) {
            for (int i = 0; i < 512; i++) {
                for (int j = 0; j < 512; j++) {
                    float pix = image_to_inapint.at<cv::Vec3b>(i, j)[c];
                    image_data[c * 512 * 512 + i * 512 + j] = pix;

                }
            }
        }

        ov::Shape image_shape = {1, 3, 512, 512};
        ov::Tensor image_tensor =  ov::Tensor(ov::element::f32, image_shape, image_data.data());;

        std::vector<float> mask_data;

        mask_data.resize(1 * 512 * 512);

        for (int c = 0; c < 1; c++) {
            for (int i = 0; i < 512; i++) {
                for (int j = 0; j < 512; j++) {
                    float pix = mask.at<cv::Vec3b>(i, j)[c];
                    mask_data[c * 512 * 512 + i * 512 + j] = pix;

                }
            }
        }

        ov::Shape shape_mask = {1, 1, 512, 512};

        ov::Tensor mask_tensor =  ov::Tensor(ov::element::f32, shape_mask, mask_data.data());;

        std::vector<ov::Tensor> tensors = {image_tensor, mask_tensor};

        m_request.set_tensor(m_model->input(0), image_tensor);
        m_request.set_tensor(m_model->input(1), mask_tensor);

        m_request.infer();

        auto output = BuildOutput();
        cv::Mat image_output = output[0];

        cv::imshow("inapitn image", image_output);
        cv::waitKey(0);
    }
    catch (std::exception& e) {
        qDebug() << "Failed to Infer! ec: " << e.what() << '\n';
    }
}


std::vector<cv::Mat> Inpaint::GetResult()
{
    return result;
}


bool Inpaint::ParseArgs(int input_number, int output_number)
{
    try {
        std::map<ov::Output<ov::Node>, ov::PartialShape> port_to_shape;
        for (const ov::Output<ov::Node>& input : m_model->inputs()) {
            ov::PartialShape shape = input.get_partial_shape();
            shape[0] = 1;
            port_to_shape[input] = shape;
        }
        m_model->reshape(port_to_shape);

        size_t num_inputs = m_model->inputs().size();
        size_t num_outputs = m_model->outputs().size();

        for (size_t i = 0; i < num_inputs; i++) {
            model_input_shape.push_back(m_model->input(i).get_shape());
        }

        for (size_t i = 0; i < num_outputs; i++) {
            model_output_shape.push_back(m_model->output(i).get_shape());
        }

        input_channel = model_input_shape[0][1];
        input_height = model_input_shape[0][2];
        input_width = model_input_shape[0][3];

        this->input_data.resize(input_channel * input_height * input_height);

        qDebug() << "model input height:" << input_height << " input width:" << input_width << "\n";

        qDebug() << "model channgels: " << input_channel << "\n";

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


bool Inpaint::BuildProcessor()
{
    try
    {
        m_ppp = std::make_shared<ov::preprocess::PrePostProcessor>(m_model);
        m_ppp->input(0).tensor()
            .set_shape({1, 3, input_height, input_width})
            .set_element_type(ov::element::f32)
            .set_layout("NCHW")
            .set_color_format(ov::preprocess::ColorFormat::RGB);

        m_ppp->input(1).tensor()
            .set_shape({1, 1, input_height, input_width})
            .set_element_type(ov::element::f32)
            .set_layout("NCHW")
            .set_color_format(ov::preprocess::ColorFormat::GRAY);

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


std::vector<cv::Mat> Inpaint::BuildOutput()
{
    size_t num_outputs = m_model->outputs().size();
    std::vector<cv::Mat> build_outputs;

    for (size_t i = 0; i < num_outputs; i++) {
        auto* ptr = m_request.get_output_tensor(i).data();

        build_outputs.push_back(cv::Mat(model_output_shape[i][2], model_output_shape[i][3], CV_32FC3, ptr));
    }

    return build_outputs;
}


ov::Tensor Inpaint::BuildTensor()
{
    ov::Shape shape = {1, static_cast<unsigned long>(input_channel), static_cast<unsigned long>(input_height), static_cast<unsigned long>(input_width)};

    return ov::Tensor(ov::element::f32, shape, input_data.data());;
}


bool Inpaint::ConvertLayout(cv::Mat &image)
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

