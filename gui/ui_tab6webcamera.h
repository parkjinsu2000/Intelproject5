/********************************************************************************
** Form generated from reading UI file 'tab6webcamera.ui'
**
** Created by: Qt User Interface Compiler version 6.8.3
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_TAB6WEBCAMERA_H
#define UI_TAB6WEBCAMERA_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QGraphicsView>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_Tab6WebCamera
{
public:
    QVBoxLayout *verticalLayout;
    QGraphicsView *pGPView;
    QPushButton *pPBCamStart;

    void setupUi(QWidget *Tab6WebCamera)
    {
        if (Tab6WebCamera->objectName().isEmpty())
            Tab6WebCamera->setObjectName("Tab6WebCamera");
        Tab6WebCamera->resize(438, 375);
        verticalLayout = new QVBoxLayout(Tab6WebCamera);
        verticalLayout->setObjectName("verticalLayout");
        pGPView = new QGraphicsView(Tab6WebCamera);
        pGPView->setObjectName("pGPView");

        verticalLayout->addWidget(pGPView);

        pPBCamStart = new QPushButton(Tab6WebCamera);
        pPBCamStart->setObjectName("pPBCamStart");
        pPBCamStart->setCheckable(true);

        verticalLayout->addWidget(pPBCamStart);


        retranslateUi(Tab6WebCamera);

        QMetaObject::connectSlotsByName(Tab6WebCamera);
    } // setupUi

    void retranslateUi(QWidget *Tab6WebCamera)
    {
        Tab6WebCamera->setWindowTitle(QCoreApplication::translate("Tab6WebCamera", "Form", nullptr));
        pPBCamStart->setText(QCoreApplication::translate("Tab6WebCamera", "CamStart", nullptr));
    } // retranslateUi

};

namespace Ui {
    class Tab6WebCamera: public Ui_Tab6WebCamera {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_TAB6WEBCAMERA_H
