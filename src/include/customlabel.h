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

private:
    QImage orginal_img;
    QImage img;
    QRect getTargetRect(QImage img);

    FastSAM fastsam;
    Depth depth;

    void mousePressEvent(QMouseEvent *e) override;
    void paintEvent(QPaintEvent *event) override;

public:
    customLabel(QWidget *parent = nullptr);

    static QPoint getTransformedPoint(QSize window, QSize img, QPoint pt, bool isSourcePoint);

    void SetImage(QImage image, QString file_name);
    void SetOrginalImage();
    void SegmentAll();
    void ShowDepth();
};

#endif // CUSTOMLABEL_H
