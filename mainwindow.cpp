#include "mainwindow.h"
#include "./ui_mainwindow.h"

#include <QFileDialog>
#include <QMessageBox>
#include <QDir>
#include <QPixmap>
#include <QMouseEvent>
#include <QDebug>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <openvino/openvino.hpp>

#include <cppsam/SAMModel.h>
#include <vino_executor/ONNXVinoExecutor.h>


cv::Mat applyMask(const cv::Mat& image, const cv::Mat& mask) {
    cv::Mat result = cv::Mat::zeros(image.size(), image.type());
    image.copyTo(result, mask);
    return result;
}

cv::Mat getMask(cv::Mat& image, std::vector<float> input_coordinates) {
    std::vector<float> input_labels = { 1 };
    //std::vector<float> input_coordinates = { 926, 926 };
    std::vector<cv::Point2f> input_points;

    for (size_t i = 0; i < input_coordinates.size(); i = i + 2)
        input_points.emplace_back(input_coordinates[i], input_coordinates[i + 1]);

    if (input_points.size() != input_labels.size())
        throw std::runtime_error("The number of points and labels must coincide");

    cv::cvtColor(image, image, cv::COLOR_BGR2RGB); // SAM awaits an image in RGB format

    // Setting up and running the model
    cppsam::SAMModel model(std::make_shared<vino_executor::ONNXVinoExecutor>(ov::Core(), "/home/filnow/fun/dex-vision/models/image_encoder.onnx",
                                                                             "/home/filnow/fun/dex-vision/models/the_rest.onnx", "CPU"));

    // Making inference
    model.setInput(image);
    cv::Mat result = model.predict(input_points, input_labels);

    //preparing for showing the results
    cv::resize(result, result, cv::Size(), 0.4, 0.4); // resize for convenient representation
    cv::resize(image, image, cv::Size(), 0.4, 0.4); // resize for convenient representation
    cv::cvtColor(image, image, cv::COLOR_RGB2BGR); // converting back into BGR format for presenting
    cv::Mat masked = applyMask(image, result);

    return masked;

}

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    // Set label background color to grey
    ui->label_pic->setStyleSheet("QLabel { background-color : grey;}");

}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_pushButton_clicked()
{
    file_name = QFileDialog::getOpenFileName(this, "Open a file", QDir::homePath(),
                            tr("Image Files (*.png *.jpg *.bmp *.tif);;"));

    // Check if file name empty
    if (file_name != "") {
        // Display image on label
        QPixmap img(file_name);
        ui->label_pic->setPixmap(img);
        ui->label_pic->setScaledContents( true );
        ui->label_pic->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);

        // MessageBox with path to file
        QMessageBox::information(this, "Displayed image from path:", file_name);
    }

}

void MainWindow::mousePressEvent(QMouseEvent *event)
{
    // Check if a file has been loaded
    if (!ui->label_pic->pixmap().isNull() && event->button() == Qt::LeftButton && ui->label_pic->geometry().contains(event->pos()))
    {
        // Get the size of the displayed image
        QPixmap pixmap = ui->label_pic->pixmap();
        QSize imageSize = pixmap.size();

        // Get the size of the label widget
        QSize labelSize = ui->label_pic->size();

        // Calculate the scaling factors
        double scaleX = static_cast<double>(labelSize.width()) / imageSize.width();
        double scaleY = static_cast<double>(labelSize.height()) / imageSize.height();

        // Calculate the scaled mouse coordinates
        int scaledX = static_cast<int>((event->pos().x() - ui->label_pic->x()) / scaleX);
        int scaledY = static_cast<int>((event->pos().y() - ui->label_pic->y()) / scaleY);

        qDebug() << "Scaled mouse position:" << scaledX << ", " << scaledY;

        // Call the getMask function and pass the coordinates and the cv_image
        std::vector<float> input_coordinates = {static_cast<float>(scaledX), static_cast<float>(scaledY)};

        cv::Mat cv_image = cv::imread(file_name.toStdString(), cv::IMREAD_COLOR);
        cv::Mat masked_image = getMask(cv_image, input_coordinates);

        // Display the masked image on the label
        QImage qimage(masked_image.data, masked_image.cols, masked_image.rows, masked_image.step, QImage::Format_RGB888);
        ui->label_pic->setPixmap(QPixmap::fromImage(qimage));
    }
}


