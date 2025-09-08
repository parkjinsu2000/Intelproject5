#ifndef WIDGET_H
#define WIDGET_H

#include <QWidget>
#include <QProcess>
#include <QGraphicsView>    // graphicsView 위젯을 쓸 때
#include <QGraphicsScene>   // scene 객체를 쓸 때
#include <QPixmap>          // 이미지 렌더링용
#include <QImage>           // 이미지 디코딩용

QT_BEGIN_NAMESPACE
namespace Ui {
class Widget;
}
QT_END_NAMESPACE

class Widget : public QWidget
{
    Q_OBJECT

public:
    Widget(QWidget *parent = nullptr);
    ~Widget();

private slots:
    void on_pushButton_clicked();
    void onPythonStdout();
    void onPythonStderr();
    void onPythonFinished(int exitCode, QProcess::ExitStatus status);

private:
    Ui::Widget *ui;
    QProcess* pyProc = nullptr;   // 파이썬 프로세스 유지
    QGraphicsScene *scene ;

};
#endif // WIDGET_H
