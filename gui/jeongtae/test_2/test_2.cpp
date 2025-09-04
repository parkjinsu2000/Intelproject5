#include <QDebug>
#include <QDir>
#include "test_2.h"
#include "ui_test_2.h"

test_2::test_2(QWidget *parent)
    : QWidget(parent), ui(new Ui::test_2)
{
    ui->setupUi(this);

    pyProc = new QProcess(this);
    pyProc->setProcessChannelMode(QProcess::SeparateChannels);

    connect(pyProc, &QProcess::readyReadStandardOutput, this, &test_2::onPythonStdout);
    connect(pyProc, &QProcess::readyReadStandardError,  this, &test_2::onPythonStderr);
    connect(pyProc, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            this, &test_2::onPythonFinished);
}

test_2::~test_2()
{
    if (pyProc && pyProc->state() != QProcess::NotRunning) {
        pyProc->terminate();              // 정상 종료 유도
        if (!pyProc->waitForFinished(1500))
            pyProc->kill();               // 강제 종료
        pyProc->waitForFinished(1000);
    }
    delete ui;
}

void test_2::on_pushButton_clicked()
{
    qDebug() << "test2 Button Clicked!!";

    if (pyProc->state() != QProcess::NotRunning) {
        qDebug() << "Python already running.";
        return;
    }

    // 경로 및 인자 설정
    const QString workDir = "/home/ubuntu/Qt/py_code";
    // const QString python  = "python3"; // venv 쓰면 절대경로 지정
    const QString python  = "/home/ubuntu/workspace_intel/yolov8pose-venv/bin/python3"; // venv 쓰면 절대경로 지정
    const QString script  = workDir + "/realtime_pose_score.py";
    const QString ref     = workDir + "/ref.mp4"; // ref.mp4 위치가 다르면 변경

    QStringList args;
    args << "-u"          // unbuffered: stdout 즉시 전달
         << script
         << ref
         << "--cam" << "0"
         << "--start" << "3";

    pyProc->setWorkingDirectory(workDir);
    pyProc->start(python, args);

    if (!pyProc->waitForStarted(3000)) {
        qWarning() << "Failed to start python:" << pyProc->errorString();
    }
}

void test_2::onPythonStdout()
{
    const QByteArray out = pyProc->readAllStandardOutput();
    qDebug().noquote() << QString::fromUtf8(out);   // 파이썬 표준 출력
}

void test_2::onPythonStderr()
{
    const QByteArray err = pyProc->readAllStandardError();
    qWarning().noquote() << QString::fromUtf8(err); // 파이썬 에러 출력
}

void test_2::onPythonFinished(int exitCode, QProcess::ExitStatus status)
{
    qDebug() << "Python finished. exitCode=" << exitCode << "status=" << status;
}
