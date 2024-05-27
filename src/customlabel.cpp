#include "customlabel.h"


customLabel::customLabel(QWidget *parent) : QLabel(parent)
{
}

void customLabel::SetImage(QImage image, QString file_name)
{
    orginal_img = image;
    img = image;
    repaint();

    QMessageBox::information(this, "Displayed image from path:", file_name);

    QDir appDir(QCoreApplication::applicationDirPath());
    appDir.cdUp(); appDir.cdUp();

    QString samPath = appDir.filePath("models/FastSAM-x.xml");
    QString depthPath = appDir.filePath("models/depth_anything_vitb14.xml");

    if(fastsam.Initialize(samPath.toStdString(), 0.6, 0.9)) {
        fastsam.Infer(file_name.toStdString());
    }

    if (depth.Initialize(depthPath.toStdString())) {
        depth.Infer(file_name.toStdString());
    }
}

void customLabel::ShowDepth()
{
    depth_map = depth.RenderDepth();
    QImage qimage(depth_map.data, depth_map.cols, depth_map.rows, depth_map.step, QImage::Format_BGR888);
    img = qimage;
    repaint();
}

void customLabel::ScanImage(QString file_name)
{
    depth_map = depth.RenderDepth();

    cv::Mat gray_depth;
    cv::cvtColor(depth_map, gray_depth, cv::COLOR_BGR2GRAY);

    cv::Mat th;
    cv::threshold(gray_depth, th, 127, 255, cv::THRESH_BINARY);

    cv::Mat kernel = cv::Mat::ones(15, 15, CV_8U);

    cv::Mat dilate;
    cv::morphologyEx(th, dilate, cv::MORPH_CLOSE, kernel, cv::Point(-1,-1), 3);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(dilate, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat img1 = cv::imread(file_name.toStdString(), cv::IMREAD_COLOR);

    for (int z = 0; z < 256;  ++z) {
        cv::Mat frame = img1.clone();

        cv::drawContours(frame, contours, -1, cv::Scalar(0, 255, 0), 3);

        cv::Mat plane_mask = cv::Mat::zeros(gray_depth.size(), CV_8UC1);
        plane_mask.setTo(255, gray_depth <= z);

        cv::Mat plane_color;
        cv::cvtColor(plane_mask, plane_color, cv::COLOR_GRAY2BGR);
        cv::addWeighted(frame, 1.0, plane_color, 0.5, 0, frame);

        QImage qimage(frame.data, frame.cols, frame.rows, frame.step, QImage::Format_BGR888);
        img = qimage;
        repaint();

        cv::waitKey(10);
    }
}

void customLabel::SetOrginalImage()
{
    img = orginal_img;
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

        std::vector<cv::Point2f> cords;
        cords.push_back(cv::Point2f(qcords.x(), qcords.y()));

        cv::Mat mask = fastsam.RenderSingleMask(cords);
        QImage qimage(mask.data, mask.cols, mask.rows, mask.step, QImage::Format_BGR888);

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
