#include "mainwindow.h"
#include "./ui_mainwindow.h"

#include <QFileDialog>
#include <QMessageBox>
#include <QDir>
#include <QPixmap>
#include <QMouseEvent>
#include <QDebug>
#include <QMovie>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    this->ui->label_pic->ModelInit();
}


MainWindow::~MainWindow()
{
    delete ui;
}


QDialog* MainWindow::make_gif_dialog(QString path_to_gif)
{
    QDialog* dialog = new QDialog(this, Qt::FramelessWindowHint);
    dialog->setAttribute(Qt::WA_TranslucentBackground);

    QLabel* gifLabel = new QLabel(dialog);
    QMovie* movie = new QMovie(path_to_gif);
    gifLabel->setMovie(movie);
    movie->start();

    dialog->setFixedSize(movie->frameRect().size());

    QVBoxLayout* layout = new QVBoxLayout(dialog);
    layout->addWidget(gifLabel);
    layout->setAlignment(Qt::AlignCenter);

    return dialog;
}

//FILE

void MainWindow::on_actionLoad_triggered()
{
    QDir appDir(QCoreApplication::applicationDirPath());
    appDir.cdUp(); appDir.cdUp();

    QString imagePath = appDir.absolutePath() + "/images";

    QDialog* gifBox = make_gif_dialog(":/loading.gif");

    file_name = QFileDialog::getOpenFileName(this, "Open a file", imagePath,
                                             tr("Image Files (*.png *.jpg *.bmp *.tif);;"));

    if (file_name != "") {
        QImage img(file_name);

        connect(this->ui->label_pic, &customLabel::processingStarted, this, [gifBox]() {
            gifBox->show();
        });

        connect(this->ui->label_pic, &customLabel::processingFinished, this, [gifBox]() {
            gifBox->hide();
        });

        this->ui->label_pic->SetImage(img, file_name);
    }
}


void MainWindow::on_actionSave_triggered()
{
    QImage img = this->ui->label_pic->SaveImage();
    QString fileName = QFileDialog::getSaveFileName(this, tr("Save Image File"),
                                                    QString(),
                                                    tr("Images (*.png *.jpg *.bmp *.tif)"));
    if (!fileName.isEmpty())
    {
        img.save(fileName);
    }

}


void MainWindow::on_actionClear_triggered()
{
    this->ui->label_pic->SetOrginalImage();
}

//EDIT

void MainWindow::on_actionSegment_All_triggered()
{
    this->ui->label_pic->SegmentAll();
}


void MainWindow::on_actionRemove_Background_triggered()
{
    this->ui->label_pic->RemoveBackground();
}


void MainWindow::on_actionDepth_triggered()
{
    this->ui->label_pic->ShowDepth();
}

//EFFECTS

void MainWindow::on_actionScanner_triggered()
{
    this->ui->label_pic->ScanImage();
}



