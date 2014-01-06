#-------------------------------------------------
#
# Project created by QtCreator 2013-10-29T18:12:56
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = TrackFace
TEMPLATE = app


SOURCES += main.cpp\
        TrackFace.cpp \
    iofunctions.cpp \
    grabfaceform.cpp

HEADERS  += TrackFace.h \
    iofunctions.h \
    grabfaceform.h \
    SharedLibraries.h

FORMS    += trackface.ui \
    grabfaceform.ui

INCLUDEPATH += /usr/local/include/
LIBS += -L/usr/local/lib
LIBS += -lopencv_core
LIBS += -lopencv_imgproc
LIBS += -lopencv_highgui
LIBS += -lopencv_ml
LIBS += -lopencv_video
LIBS += -lopencv_features2d
LIBS += -lopencv_calib3d
LIBS += -lopencv_objdetect
LIBS += -lopencv_contrib
LIBS += -lopencv_legacy
LIBS += -lopencv_flann
LIBS += -lopencv_nonfree

RESOURCES +=
