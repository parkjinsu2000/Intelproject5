/********************************************************************************
** Form generated from reading UI file 'test_2.ui'
**
** Created by: Qt User Interface Compiler version 6.9.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_TEST_2_H
#define UI_TEST_2_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_test_2
{
public:
    QVBoxLayout *verticalLayout_4;
    QVBoxLayout *verticalLayout_3;
    QPushButton *pushButton;

    void setupUi(QWidget *test_2)
    {
        if (test_2->objectName().isEmpty())
            test_2->setObjectName("test_2");
        test_2->resize(800, 600);
        verticalLayout_4 = new QVBoxLayout(test_2);
        verticalLayout_4->setObjectName("verticalLayout_4");
        verticalLayout_3 = new QVBoxLayout();
        verticalLayout_3->setObjectName("verticalLayout_3");
        pushButton = new QPushButton(test_2);
        pushButton->setObjectName("pushButton");

        verticalLayout_3->addWidget(pushButton);


        verticalLayout_4->addLayout(verticalLayout_3);


        retranslateUi(test_2);

        QMetaObject::connectSlotsByName(test_2);
    } // setupUi

    void retranslateUi(QWidget *test_2)
    {
        test_2->setWindowTitle(QCoreApplication::translate("test_2", "test_2", nullptr));
        pushButton->setText(QCoreApplication::translate("test_2", "PushButton", nullptr));
    } // retranslateUi

};

namespace Ui {
    class test_2: public Ui_test_2 {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_TEST_2_H
