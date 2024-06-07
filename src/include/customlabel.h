#ifndef CUSTOMLABEL_H
#define CUSTOMLABEL_H

#include <QLabel>
#include <QPainter>
#include <qimage.h>
#include <QMouseEvent>
#include "QDialog"
#include "QMimeData"
#include "QProgressBar"
#include <QDir>
#include <QMessageBox>

#include <opencv2/core.hpp>
#include "fastsam.h"
#include "depth.h"


class customLabel : public QLabel
{
    Q_OBJECT

public:
    customLabel(QWidget *parent = nullptr);

    static QPoint getTransformedPoint(QSize window, QSize img, QPoint pt, bool isSourcePoint);

    void SetImage(QImage image, QString file_name);
    void SetOrginalImage();
    void SegmentAll();
    void ShowDepth();
    void ScanImage();
    void RemoveBackground();
    void ModelInit();
    QImage SaveImage();

signals:
    void processingStarted();
    void processingFinished();

private:
    QImage orginal_img;
    QImage img;
    cv::Mat cv_img;
    cv::Mat depth_map;
    QRect getTargetRect(QImage img);

    FastSAM fastsam;
    Depth depth;

    std::vector<cv::Mat> fastsam_results;
    cv::Mat clicked_mask;
    cv::Mat image_with_masks;
    std::vector<cv::Point2f> cords;

    bool sam_init = false;
    bool depth_init = false;

private:
    void mousePressEvent(QMouseEvent *e) override;
    void paintEvent(QPaintEvent *event) override;

    void ColorMask(const cv::Mat& mask, cv::Mat& rendered, bool multi_color=true);

    cv::Scalar RandomColor();
    std::vector<std::vector<cv::Point>> drawLine(cv::Mat mask, cv::Mat image, int line_size, bool draw);
};

#endif // CUSTOMLABEL_H
