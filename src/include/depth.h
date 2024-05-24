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

private:
    ov::Core m_core;
    std::shared_ptr<ov::Model> m_model;
    ov::CompiledModel m_compiled_model;
    ov::InferRequest m_request;
};

#endif // DEPTH_H
