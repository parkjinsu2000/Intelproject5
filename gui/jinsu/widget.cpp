#include "widget.h"
#include "./ui_widget.h"
#include <QProcess>
#include <QGraphicsScene>
#include <QGraphicsPixmapItem>
#include <QImage>
#include <QPixmap>
#include <QDebug>

Widget::Widget(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Widget)
{
    ui->setupUi(this);

    pyProc = new QProcess(this);
    pyProc->setProcessChannelMode(QProcess::SeparateChannels);

    scene = new QGraphicsScene(this);
    ui->graphicsView->setScene(scene);

    connect(pyProc, &QProcess::readyReadStandardOutput, this, &Widget::onPythonStdout);
    connect(pyProc, &QProcess::readyReadStandardError, this, &Widget::onPythonStderr);
    connect(pyProc, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            this, &Widget::onPythonFinished);
}

Widget::~Widget()
{
    if (pyProc && pyProc->state() != QProcess::NotRunning) {
        pyProc->terminate();
        if (!pyProc->waitForFinished(1500))
            pyProc->kill();
        pyProc->waitForFinished(1000);
    }
    delete ui;
}

void Widget::on_pushButton_clicked()
{
    if (pyProc->state() != QProcess::NotRunning) {
        qDebug() << "Python already running.";
        return;
    }

    const QString workDir = "/home/ubuntu/Qt/py_code";
    const QString python  = "/home/ubuntu/myproject/venv2/bin/python3";
    const QString script  = workDir + "/realtime_pose_score.py";
    const QString ref     = workDir + "/ref_2.mp4";

    QStringList args;
    args << "-u" << script << ref
         << "--cam" << "0"
         << "--start" << "3"
         << "--disp-scale" << "0.7"
         << "--disp-width" << "640";

    pyProc->setWorkingDirectory(workDir);
    pyProc->start(python, args);

    if (!pyProc->waitForStarted(3000)) {
        qWarning() << "Failed to start python:" << pyProc->errorString();
    }
}

void Widget::onPythonStdout()
{
    while (pyProc->canReadLine()) {
        QByteArray line = pyProc->readLine().trimmed();
        if (line.isEmpty()) continue;

        QByteArray imageData = QByteArray::fromBase64(line);
        QImage image;
        if (!image.loadFromData(imageData, "JPG")) {
            qWarning() << "[Qt] Failed to load image from base64.";
            continue;
        }

        QPixmap pixmap = QPixmap::fromImage(image);
        scene->clear();
        scene->addPixmap(pixmap);
        ui->graphicsView->fitInView(scene->itemsBoundingRect(), Qt::KeepAspectRatio);
    }
}

void Widget::onPythonStderr()
{
    const QByteArray err = pyProc->readAllStandardError();
    qWarning().noquote() << "[Python STDERR]" << QString::fromUtf8(err);
}

void Widget::onPythonFinished(int exitCode, QProcess::ExitStatus status)
{
    qDebug() << "Python finished. exitCode=" << exitCode << "status=" << status;
}
