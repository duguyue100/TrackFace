#ifndef TRACKFACE_H
#define TRACKFACE_H

#include<QMainWindow>
#include<QFile>
#include<QCoreApplication>
#include<QTextStream>
#include<QDir>

#include "SharedLibraries.h"

// Local libraries

#include "iofunctions.h"
#include "grabfaceform.h"

namespace Ui {
class TrackFace;
class GrabFaceForm;
}

class TrackFace : public QMainWindow
{
    Q_OBJECT
    
public:
    explicit TrackFace(QWidget *parent = 0);
    ~TrackFace();

    string fn_path;  // path to resources
    string fn_images; // images path and labels
    string fn_haar; // for front face
    string fn_features; // for features
    string fn_namedb; // for record all names;

    vector<cv::Mat> images; // loaded images
    vector<int> labels; // image labels
    vector<string> names; // corresponding names

    int im_width; // training images' width
    int im_height; // training images' height

    int window_x; // pop up window x
    int window_y; // pop up window y

    cv::VideoCapture capture; // video capture

    IOFunctions iofunctions; // provide IO Functions

    GrabFaceForm grabForm;

    // training

    Ptr<FaceRecognizer> model;
    CascadeClassifier haar_cascade;

    // utilities functions

    // running a training through whole dataset
    void training();

    // detected haar faces
    vector<Rect_<int> > haar_faces(cv::Mat frame);

    // Draw Face based on detected face; (Front only)
    void drawFace(cv::Mat & frame, cv::Rect & face_n, string box_text);

    // Find the face which has maximum size among all detected faces
    size_t findMaxFace(vector<cv::Rect_<int> > faces);

    // Resize the face to training image sacle and preprocessing for recognition.
    cv::Mat resizeRecognitionFace(cv::Mat frame, Rect face_n);

    // Resize the face to certain scale
    cv::Mat resizeFace(cv::Mat face, int width, int height);

    // change color image to black-to-white sketch
    cv::Mat cartoonifyImageSketch(cv::Mat srcColor);

    // change color image to cartoonified image
    cv::Mat cartoonifyImageColor(cv::Mat srcColor);

private slots:
    void on_grabPhoto_clicked();

    void on_training_clicked();

    void on_recognition_clicked();

    void on_cartoonifySketch_clicked();

    void on_cartoonifyColor_clicked();

    void on_drawKeypoints_clicked();

    void on_oneClassClassifier_clicked();

private:
    Ui::TrackFace *ui;
};

#endif // TRACKFACE_H
