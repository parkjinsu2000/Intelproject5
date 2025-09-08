/********************************************************************************
** Form generated from reading UI file 'tab3controlpannel.ui'
**
** Created by: Qt User Interface Compiler version 6.8.3
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_TAB3CONTROLPANNEL_H
#define UI_TAB3CONTROLPANNEL_H

#include <QtCore/QVariant>
#include <QtGui/QIcon>
#include <QtWidgets/QApplication>
#include <QtWidgets/QFormLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QScrollArea>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_Tab3ControlPannel
{
public:
    QFormLayout *formLayout_2;
    QScrollArea *scrollArea;
    QWidget *scrollAreaWidgetContents;
    QPushButton *pPBlamp;
    QPushButton *pPBplug;
    QFormLayout *formLayout;
    QLabel *label;

    void setupUi(QWidget *Tab3ControlPannel)
    {
        if (Tab3ControlPannel->objectName().isEmpty())
            Tab3ControlPannel->setObjectName("Tab3ControlPannel");
        Tab3ControlPannel->resize(575, 370);
        formLayout_2 = new QFormLayout(Tab3ControlPannel);
        formLayout_2->setObjectName("formLayout_2");
        scrollArea = new QScrollArea(Tab3ControlPannel);
        scrollArea->setObjectName("scrollArea");
        scrollArea->setWidgetResizable(true);
        scrollAreaWidgetContents = new QWidget();
        scrollAreaWidgetContents->setObjectName("scrollAreaWidgetContents");
        scrollAreaWidgetContents->setGeometry(QRect(0, 0, 555, 350));
        pPBlamp = new QPushButton(scrollAreaWidgetContents);
        pPBlamp->setObjectName("pPBlamp");
        pPBlamp->setGeometry(QRect(120, 140, 71, 71));
        QIcon icon;
        icon.addFile(QString::fromUtf8(":/Images/Images/light_off.png"), QSize(), QIcon::Mode::Normal, QIcon::State::Off);
        icon.addFile(QString::fromUtf8(":/Images/Images/light_on.png"), QSize(), QIcon::Mode::Normal, QIcon::State::On);
        pPBlamp->setIcon(icon);
        pPBlamp->setIconSize(QSize(70, 70));
        pPBlamp->setCheckable(true);
        pPBplug = new QPushButton(scrollAreaWidgetContents);
        pPBplug->setObjectName("pPBplug");
        pPBplug->setGeometry(QRect(380, 163, 71, 71));
        QIcon icon1;
        icon1.addFile(QString::fromUtf8(":/Images/Images/plug_off.png"), QSize(), QIcon::Mode::Normal, QIcon::State::Off);
        icon1.addFile(QString::fromUtf8(":/Images/Images/plug_on.png"), QSize(), QIcon::Mode::Normal, QIcon::State::On);
        pPBplug->setIcon(icon1);
        pPBplug->setIconSize(QSize(70, 70));
        pPBplug->setCheckable(true);
        formLayout = new QFormLayout(scrollAreaWidgetContents);
        formLayout->setObjectName("formLayout");
        label = new QLabel(scrollAreaWidgetContents);
        label->setObjectName("label");
        label->setPixmap(QPixmap(QString::fromUtf8(":/Images/Images/room1.png")));

        formLayout->setWidget(0, QFormLayout::LabelRole, label);

        scrollArea->setWidget(scrollAreaWidgetContents);
        label->raise();
        pPBplug->raise();
        pPBlamp->raise();

        formLayout_2->setWidget(0, QFormLayout::SpanningRole, scrollArea);


        retranslateUi(Tab3ControlPannel);

        QMetaObject::connectSlotsByName(Tab3ControlPannel);
    } // setupUi

    void retranslateUi(QWidget *Tab3ControlPannel)
    {
        Tab3ControlPannel->setWindowTitle(QCoreApplication::translate("Tab3ControlPannel", "Form", nullptr));
        pPBlamp->setText(QString());
        pPBplug->setText(QString());
        label->setText(QString());
    } // retranslateUi

};

namespace Ui {
    class Tab3ControlPannel: public Ui_Tab3ControlPannel {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_TAB3CONTROLPANNEL_H
