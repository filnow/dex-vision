#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include <QFileDialog>
#include <QMessageBox>
#include <QDir>
#include <QPixmap>

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
    QString file_name = QFileDialog::getOpenFileName(this, "Open a file", QDir::homePath(),
                            tr("Image Files (*.png *.jpg *.bmp *.tif);;"));

    // Check if file name empty
    if (file_name != "") {
        // Display image on label
        QPixmap img(file_name);
        ui->label_pic->setPixmap(img);

        // MessageBox with path to file
        QMessageBox::information(this, "Displayed image from path:", file_name);
    }
}

