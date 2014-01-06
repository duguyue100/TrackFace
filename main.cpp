#include "TrackFace.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    TrackFace w;
    w.show();
    
    return a.exec();
}
