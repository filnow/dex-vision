#include "customlabel.h"
#include "QFileDialog"
#include <QtConcurrent/QtConcurrent>
#include <QFuture>
#include <QFutureWatcher>


customLabel::customLabel(QWidget *parent) : QLabel(parent)
{
}

void customLabel::ModelInit()
{
    QDir appDir(QCoreApplication::applicationDirPath());
    appDir.cdUp(); appDir.cdUp();
    QString samPath = appDir.filePath("models/FastSAM-x.xml");
    QString depthPath = appDir.filePath("models/depth_anything_vitb14.xml");

    if (fastsam.Initialize(samPath.toStdString(), 0.6, 0.9)) {
        sam_init = true;
    }

    if (depth.Initialize(depthPath.toStdString())) {
        depth_init = true;
    }
}

void customLabel::SetImage(QImage image, QString file_name)
{
    emit processingStarted();

    img = QImage();
    cords.clear();
    repaint();

    cv_img = cv::imread(file_name.toStdString(), cv::IMREAD_COLOR);

    auto fastsamInitFunction = [this, file_name, image]() {
        if (sam_init) {
            fastsam_results = fastsam.Infer(file_name.toStdString());
        }
    };

    auto depthInitFunction = [this, file_name, image]() {
        if (depth_init) {
            depth.Infer(file_name.toStdString());
            depth_map = depth.RenderDepth();
        }

        orginal_img = image;
        img = image;
        repaint();

        emit processingFinished();
    };

    QFuture<void> fastsamFuture = QtConcurrent::run(fastsamInitFunction);
    QFuture<void> depthFuture = QtConcurrent::run(depthInitFunction);

    QFutureWatcher<void> *fastsamWatcher = new QFutureWatcher<void>();
    QFutureWatcher<void> *depthWatcher = new QFutureWatcher<void>();

    connect(fastsamWatcher, &QFutureWatcher<void>::finished, this, [this, fastsamWatcher]() {
        fastsamWatcher->deleteLater();
    });

    connect(depthWatcher, &QFutureWatcher<void>::finished, this, [this, depthWatcher, image]() {
        depthWatcher->deleteLater();
    });

    fastsamWatcher->setFuture(fastsamFuture);
    depthWatcher->setFuture(depthFuture);
}

QImage customLabel::SaveImage()
{
    return img;
}

void customLabel::ShowDepth()
{
    QImage qimage(depth_map.data, depth_map.cols, depth_map.rows, depth_map.step, QImage::Format_BGR888);
    img = qimage;
    repaint();
}


std::vector<std::vector<cv::Point>> customLabel::drawLine(cv::Mat mask, cv::Mat image, int line_size, bool draw)
{
    std::vector<cv::Vec4i> hierarchy;
    std::vector<std::vector<cv::Point>> contours;

    cv::findContours(mask, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    if (draw == true)
    {
        cv::drawContours(image, contours, -1, cv::Scalar(0, 255, 43), line_size, cv::LINE_AA, hierarchy, 0);
    }

    return contours;
}

void customLabel::RemoveBackground()
{
    cv::Mat copy_img = cv_img.clone();

    cv::Mat gray_depth;
    cv::cvtColor(depth_map, gray_depth, cv::COLOR_BGR2GRAY);

    cv::Mat mask_8UC1;
    clicked_mask.convertTo(mask_8UC1, CV_8UC1);

    //NOTE: set pixels in mask to 255 and iterate z to 255 this way mask will be untouched
    cv::Mat gray_depth_mask = gray_depth.clone();
    gray_depth_mask.setTo(cv::Scalar(255), mask_8UC1);

    std::vector<std::vector<cv::Point>> line_mask = drawLine(mask_8UC1, copy_img, 2, false);

    for (int z = 0; z < 255;  ++z) {
        cv::Mat frame = cv_img.clone();

        cv::Mat plane_mask = cv::Mat::zeros(gray_depth_mask.size(), CV_8UC1);
        plane_mask.setTo(255, gray_depth_mask <= z);

        cv::Mat plane_color;
        cv::cvtColor(plane_mask, plane_color, cv::COLOR_GRAY2BGR);
        cv::addWeighted(frame, 1.0, plane_color, 1, 0, frame);

        QImage qimage(frame.data, frame.cols, frame.rows, frame.step, QImage::Format_BGR888);
        img = qimage;
        repaint();

        cv::waitKey(5);
    }
}


void customLabel::ScanImage()
{
    cv::Mat gray_depth;
    cv::cvtColor(depth_map, gray_depth, cv::COLOR_BGR2GRAY);

    for (int z = 0; z < 256;  ++z) {
        cv::Mat frame = cv_img.clone();

        cv::Mat plane_mask = cv::Mat::zeros(gray_depth.size(), CV_8UC1);
        plane_mask.setTo(255, gray_depth == z);

        std::vector<std::vector<cv::Point>> lines = drawLine(plane_mask, frame, 3, true);

        QImage qimage(frame.data, frame.cols, frame.rows, frame.step, QImage::Format_BGR888);
        img = qimage;
        repaint();

        cv::waitKey(8);
    }

    img = orginal_img;
    repaint();
}

void customLabel::SetOrginalImage()
{
    img = orginal_img;
    cords.clear();
    repaint();
}

void customLabel::SegmentAll()
{
    cv::Mat mask = fastsam.Render();
    QImage qimage(mask.data, mask.cols, mask.rows, mask.step, QImage::Format_BGR888);
    img = qimage;
    repaint();
}

void customLabel::mousePressEvent(QMouseEvent *e)
{
    if (e->button() == Qt::RightButton)
    {
    }
    else if (e->button() == Qt::LeftButton)
    {
        QPoint qcords = getTransformedPoint(this->size(), img.size(), e->pos(), true);

        if (std::find(cords.begin(), cords.end(), cv::Point2f(qcords.x(), qcords.y())) == cords.end()) {
            cords.push_back(cv::Point2f(qcords.x(), qcords.y()));
        } else {
            cords.erase(std::remove(cords.begin(), cords.end(), cv::Point2f(qcords.x(), qcords.y())), cords.end());
        }

        std::tie(image_with_masks, clicked_mask) = fastsam.RenderSingleMask(cords);

        QImage qimage(image_with_masks.data, image_with_masks.cols, image_with_masks.rows, image_with_masks.step, QImage::Format_BGR888);

        img = qimage;
        repaint();
    }
}

QRect customLabel::getTargetRect(QImage img)
{
    return QRect(QPoint(getTransformedPoint(this->size(), img.size(), {0, 0}, false)), QPoint(getTransformedPoint(this->size(), img.size(), {img.width(), img.height()}, false)));
}

void customLabel::paintEvent(QPaintEvent *event)
{
    QPainter p(this);
    p.drawImage(getTargetRect(img), img);
}

QPoint customLabel::getTransformedPoint(QSize window, QSize img, QPoint pt, bool isSourcePoint)
{
    float letterbox_rows = window.height();
    float letterbox_cols = window.width();
    float scale_letterbox;
    int resize_rows;
    int resize_cols;

    if ((letterbox_rows * 1.0 / img.height()) < (letterbox_cols * 1.0 / img.width()))
    {
        scale_letterbox = (float)letterbox_rows * 1.0f / (float)img.height();
    }
    else
    {
        scale_letterbox = (float)letterbox_cols * 1.0f / (float)img.width();
    }

    resize_cols = int(scale_letterbox * (float)img.width());
    resize_rows = int(scale_letterbox * (float)img.height());

    int tmp_h = (letterbox_rows - resize_rows) / 2;
    int tmp_w = (letterbox_cols - resize_cols) / 2;

    float ratio_x = (float)img.height() / resize_rows;
    float ratio_y = (float)img.width() / resize_cols;

    auto x0 = pt.x();
    auto y0 = pt.y();

    if (isSourcePoint)
    {
        x0 = (x0 - tmp_w) * ratio_x;
        y0 = (y0 - tmp_h) * ratio_y;
    }
    else
    {
        x0 = x0 / ratio_x + tmp_w;
        y0 = y0 / ratio_y + tmp_h;
    }

    return QPoint(x0, y0);
}
