#ifndef TEST_2_H
#define TEST_2_H

#pragma once
#include <QWidget>
#include <QProcess>

QT_BEGIN_NAMESPACE
namespace Ui {
class test_2;
}
QT_END_NAMESPACE

class test_2 : public QWidget
{
    Q_OBJECT

public:
    test_2(QWidget *parent = nullptr);
    ~test_2();

private slots:
    void on_pushButton_clicked();
    void onPythonStdout();
    void onPythonStderr();
    void onPythonFinished(int exitCode, QProcess::ExitStatus status);

private:
    Ui::test_2 *ui;
    QProcess* pyProc = nullptr;   // 파이썬 프로세스 유지
};
#endif // TEST_2_H
