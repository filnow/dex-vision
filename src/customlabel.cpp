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

    QString modelPath = appDir.filePath("models/FastSAM-x.xml");

    if(fastsam.Initialize(modelPath.toStdString(), 0.6, 0.9, true)) {
        fastsam.Infer(file_name.toStdString());
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
