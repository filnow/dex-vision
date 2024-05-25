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


MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
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

    QString file_name = QFileDialog::getOpenFileName(this, "Open a file", imagePath,
                            tr("Image Files (*.png *.jpg *.bmp *.tif);;"));

    if (file_name != "") {
        QImage img(file_name);
        this->ui->label_pic->SetImage(img, file_name);
    }
}

void MainWindow::on_pushButton_2_clicked()
{
    this->ui->label_pic->SegmentAll();
}

void MainWindow::on_pushButton_3_clicked()
{
    this->ui->label_pic->SetOrginalImage();
}

void MainWindow::on_pushButton_4_clicked()
{
    this->ui->label_pic->ShowDepth();
}

