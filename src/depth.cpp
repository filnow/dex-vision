#include "depth.h"
#include <filesystem>

#include <QDebug>



bool Depth::Initialize(const std::string &xml_path)
{
    if(!std::filesystem::exists(xml_path))
        return false;

    m_model = m_core.read_model(xml_path);

    //if(!ParseArgs())
      //  return false;


    //if(!BuildProcessor())
      //  return false;

    m_compiled_model = m_core.compile_model(m_model, "CPU");

    m_request = m_compiled_model.create_infer_request();

    qDebug() << "Init sucessfull\n";

    return true;
}
