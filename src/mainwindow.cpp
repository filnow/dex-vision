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

#include "FastSAM.h"



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

    QDir appDir(QCoreApplication::applicationDirPath());
    appDir.cdUp(); appDir.cdUp();
    QString imagePath = appDir.absolutePath() + "/images";

    file_name = QFileDialog::getOpenFileName(this, "Open a file", imagePath,
                            tr("Image Files (*.png *.jpg *.bmp *.tif);;"));

    image = cv::imread(file_name.toStdString(), cv::IMREAD_COLOR);

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

        std::vector<cv::Point2f> cords;
        cords.push_back(cv::Point2f(scaledX, scaledY));

        QDir appDir(QCoreApplication::applicationDirPath());
        appDir.cdUp(); appDir.cdUp();

        QString modelPath = appDir.filePath("models/FastSAM-x.xml");

        FastSAM fastsam;

        if(fastsam.Initialize(modelPath.toStdString(), 0.6, 0.9, true)) {
            cv::Mat mask = fastsam.Infer(file_name.toStdString(), cords);
            QImage qimage(mask.data, mask.cols, mask.rows, mask.step, QImage::Format_BGR888);
            ui->label_pic->setPixmap(QPixmap::fromImage(qimage));
        }


    }
}


void MainWindow::on_pushButton_2_clicked()
{

    QDir appDir(QCoreApplication::applicationDirPath());
    appDir.cdUp(); appDir.cdUp();

    QString modelPath = appDir.filePath("models/FastSAM-x.xml");

    if (file_name != "") {
        FastSAM fastsam;

        if(fastsam.Initialize(modelPath.toStdString(), 0.6, 0.9, true)) {
            cv::Mat mask = fastsam.Infer(file_name.toStdString());
            QImage qimage(mask.data, mask.cols, mask.rows, mask.step, QImage::Format_BGR888);
            ui->label_pic->setPixmap(QPixmap::fromImage(qimage));
        }
    }

}

void MainWindow::on_pushButton_3_clicked()
{
    QImage qimage(image.data, image.cols, image.rows, image.step, QImage::Format_BGR888);
    ui->label_pic->setPixmap(QPixmap::fromImage(qimage));
}

