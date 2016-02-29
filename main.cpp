#include "mainwindow.h"
#include <QApplication>
#include <qtextstream.h>
#include <qfile.h>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    // load stylesheet
    QFile f(":qdarkstyle/style.qss");
    if (!f.exists())
    {
        printf("Unable to set stylesheet, file not found\n");
    }
    else
    {
        f.open(QFile::ReadOnly | QFile::Text);
        QTextStream ts(&f);
        a.setStyleSheet(ts.readAll());
    }

    MainWindow w;
    w.show();

    return a.exec();
}
