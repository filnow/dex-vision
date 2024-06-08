#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <opencv2/core.hpp>


QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    QDialog* make_gif_dialog(QString path_to_gif);

    void on_actionLoad_triggered();

    void on_actionSave_triggered();

    void on_actionSegment_All_triggered();

    void on_actionRemove_Background_triggered();

    void on_actionDepth_triggered();

    void on_actionScanner_triggered();

    void on_actionClear_triggered();

    void on_actionInpaint_triggered();

private:
    Ui::MainWindow *ui;
    QString file_name;
};

#endif // MAINWINDOW_H
