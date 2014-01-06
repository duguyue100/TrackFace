#include "TrackFace.h"
#include "ui_trackface.h"

TrackFace::TrackFace(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::TrackFace)
{
    /* Set File Source Variables */

    fn_path="/Users/dgyHome/Documents/QT/TrackFace/";
    fn_images=fn_path+"resources/train_image.in";
    fn_haar=fn_path+"resources/haarcascade_frontalface_default.xml";
    fn_features=fn_path+"resources/features.in";
    fn_namedb=fn_path+"resources/namedb.in";

    im_width=150;
    im_height=150;

    window_x=300;
    window_y=300;

    /* Open DataBase */

    try
    {
        iofunctions.read_images(fn_images,fn_path,images,labels);
        iofunctions.read_name(fn_namedb,names);
    }
    catch(cv::Exception & e)
    {
        cout << "Error" << endl;
        exit(1);
    }

    /* Train/Load Features */
    if (!labels.empty() && labels[labels.size()-1]!=0)
    {
        if (QDir(fn_features.c_str()).exists())
            model->load(fn_features);
        else training();
    }

    haar_cascade.load(fn_haar);

    /* Load GUI */

    ui->setupUi(this);
}

TrackFace::~TrackFace()
{
    delete ui;
}

// variables

// constant
const unsigned int GRABBING_ON=0;
const unsigned int GRABBING_OFF=1;
const unsigned int GRABBING_CLOSE=2;
const unsigned int SIFT_MODE=1;
const unsigned int SURF_MODE=2;

unsigned int grab_state; // refer to constant value
unsigned int track_state; // refer to constant value
unsigned int featureExtractor_state;

void grabFaceCallBack(int event, int x, int y, int flags, void* userdata)
{
     if  (event == cv::EVENT_LBUTTONDOWN)
     {
         grab_state=GRABBING_ON;
     }
     else if (event == cv::EVENT_RBUTTONDOWN)
     {
         grab_state=GRABBING_CLOSE;
     }
}

void TrackFace::on_grabPhoto_clicked()
{
    // new window to collect information
    /*

    transmit data between form is tricky.
    grabForm.show();
    string message=grabForm.getMsg();

    cout << message << endl;
    */

    string name=ui->grabName->text().toStdString();
    cout << name << endl;

    string namepath=iofunctions.addName(name, fn_namedb, fn_path);
    int counter=1;
    int frames=100;

    int label=0;
    if (!labels.empty()) label=labels[labels.size()-1]+1;

    // Face tracking
    TrackFace::capture.open(0);

    string windowName="Grab Face";
    cv::namedWindow(windowName.c_str(), cv::WINDOW_AUTOSIZE);
    moveWindow(windowName.c_str(), window_x, window_y);
    grab_state=GRABBING_OFF;

    while(true)
    {
        cv::Mat frame, buffer;
        if (!capture.isOpened()) break;

        capture >> buffer;
        cv::resize(buffer, frame,Size(buffer.cols/2,buffer.rows/2),0,0,INTER_LINEAR);
        setMouseCallback(windowName.c_str(), grabFaceCallBack, NULL);

        switch(grab_state)
        {
        case GRABBING_OFF:
        {
            putText(frame, "Click anywhere to grab your face", Point(frame.cols/2-250, 100), FONT_HERSHEY_PLAIN, 1.2, CV_RGB(255,0,0),2.0);
            cv::imshow(windowName.c_str(), frame);
            break;
        }
        case GRABBING_ON:
        {
            vector<cv::Rect_<int> > faces=haar_faces(frame);

            if (faces.size()>0)
            {
                size_t n=findMaxFace(faces);

                Mat resizedFace=resizeFace(frame(faces[n]), im_width, im_height);

                while (counter/frames<=20)
                {
                    if (counter%frames==0)
                    {
                        string imgPath=namepath+name+"_"+(char)(counter/frames+'A'-1)+".jpg";
                        cv::imwrite(imgPath,resizedFace);
                        iofunctions.addToTrain(fn_images,"resources/"+name+"/"+name+"_"+(char)(counter/frames+'A'-1)+".jpg", label);
                    }
                    counter++;
                    putText(frame, "Grabbing your face", Point(frame.cols/2-250, 100), FONT_HERSHEY_PLAIN, 1.2, CV_RGB(255,0,0),2.0);
                }

                putText(frame, "Grabbing finished, click right button to close", Point(frame.cols/2-250, 100), FONT_HERSHEY_PLAIN, 1.2, CV_RGB(255,0,0),2.0);
                drawFace(frame, faces[n], name);
            }

            cv::imshow(windowName.c_str(), frame);
            break;
        }
        case GRABBING_CLOSE :
        {
            capture.release();
            cv::destroyWindow(windowName.c_str());
            break;
        }
        default: break;
        }

        while (cv::waitKey(5)==27)
        {
            capture.release();
            cv::destroyWindow(windowName.c_str());
        }
    }
}

void TrackFace::on_training_clicked()
{
    training();
}

void trackFaceCallBack(int event, int x, int y, int flags, void* userdata)
{
     if  ( event == EVENT_LBUTTONDOWN )
     {
          cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
     }
}

void TrackFace::on_recognition_clicked()
{
    TrackFace::capture.open(0);
    string windowName="Track Face";
    cv::namedWindow(windowName.c_str(), cv::WINDOW_AUTOSIZE);
    moveWindow(windowName.c_str(), window_x, window_y);
    setMouseCallback(windowName.c_str(), trackFaceCallBack, NULL);

    while (true)
    {
        cv::Mat frame, buffer;
        if (!capture.isOpened()) break;

        capture >> buffer;
        cv::resize(buffer, frame,Size(buffer.cols/2,buffer.rows/2),0,0,INTER_LINEAR);

        vector<Rect_<int> > faces=haar_faces(frame);

        for (size_t i=0;i<faces.size();i++)
        {
            cv::Mat face_resized=resizeRecognitionFace(frame, faces[i]);

            int prediction=model->predict(face_resized);

            int label=0;
            double confidence=0.0;
            model->predict(face_resized, label, confidence);
            cout << confidence << endl;

            string box_text=format("Prediction is %s", names[prediction].c_str());

            drawFace(frame, faces[i], box_text);
        }

        putText(frame, "Recognizing Face", Point(frame.cols/2-100, 30), FONT_HERSHEY_PLAIN, 1.5, CV_RGB(0,0,255),2.0);

        cv::imshow(windowName.c_str(), frame);
        while (cv::waitKey(5)==27)
        {
            capture.release();
            cv::destroyWindow(windowName.c_str());
        }
    }
}


// Utilities

void TrackFace::training()
{
    model=createFisherFaceRecognizer();

    model->train(images, labels);

    model->save(fn_features);
}

vector<Rect_<int> > TrackFace::haar_faces(cv::Mat frame)
{
    cv::Mat gray;
    cv::cvtColor(frame, gray, CV_BGR2GRAY);

    vector<Rect_<int> > faces;

    haar_cascade.detectMultiScale(gray, faces);

    return faces;
}

void TrackFace::drawFace(cv::Mat & frame, cv::Rect & face_n, string box_text)
{
    rectangle(frame, face_n, CV_RGB(0,255,0),1);
    int pos_x=std::max(face_n.tl().x-10, 0);
    int pos_y=std::max(face_n.tl().y-10, 0);
    putText(frame, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0),2.0);
}

size_t TrackFace::findMaxFace(vector<cv::Rect_<int> > faces)
{
    size_t n=0;
    int max=-1;
    for (size_t i=0;i<faces.size();i++)
    {
        if (max<faces[i].height*faces[i].width)
        {
            max=faces[i].height*faces[i].width;
            n=i;
        }
    }

    return n;
}

cv::Mat TrackFace::resizeRecognitionFace(cv::Mat frame, cv::Rect face_n)
{
    cv::Mat face=frame(face_n);
    cv::Mat grayFace;
    cv::cvtColor(face, grayFace, CV_BGR2GRAY);

    cv::Mat resizedFace;
    cv::resize(grayFace, resizedFace, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);

    return resizedFace;
}

cv::Mat TrackFace::resizeFace(cv::Mat face, int width, int height)
{
    cv::Mat resizedFace;
    cv::resize(face, resizedFace, cv::Size(width, height), 1.0, 1.0, INTER_LINEAR);

    return resizedFace;
}

cv::Mat TrackFace::cartoonifyImageSketch(cv::Mat srcColor)
{
    cv::Mat gray;

    cv::cvtColor(srcColor, gray, CV_BGR2GRAY);

    cv::Mat grayCopy;
    const int MEDIAN_BLUR_FILTER_SIZE=7;
    cv::medianBlur(gray, grayCopy, MEDIAN_BLUR_FILTER_SIZE);

    cv::Mat edges;
    const int LAPLACIAN_FILTER_SIZE=5;
    cv::Laplacian(grayCopy,edges,CV_8U,LAPLACIAN_FILTER_SIZE);

    cv::Mat mask;

    const int EDGES_THRESHOLD=80;
    cv::threshold(edges, mask, EDGES_THRESHOLD, 255, THRESH_BINARY_INV);

    return mask;
}

cv::Mat TrackFace::cartoonifyImageColor(cv::Mat srcColor)
{
    cv::Mat smallImg;

    cv::resize(srcColor, smallImg, Size(srcColor.cols/2, srcColor.rows/2), 0, 0, INTER_LINEAR);

    cv::Mat tmp=Mat(Size(srcColor.cols/2, srcColor.rows/2), CV_8UC3);

    int repetition=7;

    for (int i=0;i<repetition;i++)
    {
        int ksize=9;
        double sigmaColor=9;
        double sigmaSpace=7;

        bilateralFilter(smallImg, tmp, ksize, sigmaColor, sigmaSpace);
        bilateralFilter(tmp, smallImg, ksize, sigmaColor, sigmaSpace);
    }

    cv::Mat bigImg;
    cv::resize(smallImg, bigImg, Size(srcColor.cols,srcColor.rows), 0, 0, INTER_LINEAR);

    cv::Mat dst;

    bigImg.copyTo(dst, cartoonifyImageSketch(srcColor));

    return dst;
}

// functions for fun

void TrackFace::on_cartoonifySketch_clicked()
{
    TrackFace::capture.open(0);

    cv::namedWindow("Black-White Sketch", cv::WINDOW_AUTOSIZE);
    cv::moveWindow("Black-White Sketch", window_x, window_y);

    while (true)
    {
        cv::Mat buffer, frame;
        if (!capture.isOpened()) break;

        capture >> buffer;

        if (!buffer.empty())
        {
            cv::resize(buffer, frame,Size(buffer.cols/2,buffer.rows/2),0,0,INTER_LINEAR);

            frame=cartoonifyImageSketch(frame);

            putText(frame, "Grabbing Face", Point(100, 100), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,0,0),2.0);

            imshow("Black-White Sketch", frame);
        }

        while (cv::waitKey(5)==27)
        {
            capture.release();
            cv::destroyWindow("Black-White Sketch");
        }
    }
}

void TrackFace::on_cartoonifyColor_clicked()
{
    TrackFace::capture.open(0);

    cv::namedWindow("Cartoon Color", cv::WINDOW_AUTOSIZE);
    cv::moveWindow("Cartoon Color", window_x, window_y);

    while (true)
    {
        cv::Mat frame, buffer;

        if (!capture.isOpened()) break;

        capture >> buffer;

        cv::resize(buffer, frame,Size(buffer.cols/2,buffer.rows/2),0,0,INTER_LINEAR);
        frame=cartoonifyImageColor(frame);
        imshow("Cartoon Color", frame);

        while (cv::waitKey(5)==27)
        {
            capture.release();
            cv::destroyWindow("Cartoon Color");
        }
    }
}

void drawKeypointsCallBack(int event, int x, int y, int flags, void* userdata)
{
    if (event==cv::EVENT_LBUTTONDOWN)
    {
        featureExtractor_state=SIFT_MODE;
    }
    else if (event==cv::EVENT_RBUTTONDOWN)
    {
        featureExtractor_state=SURF_MODE;
    }
}

void TrackFace::on_drawKeypoints_clicked()
{
    int nFeatures=128;
    TrackFace::capture.open(0);

    string windowName="Draw Keypoints";
    cv::namedWindow(windowName.c_str(), cv::WINDOW_AUTOSIZE);
    cv::moveWindow(windowName.c_str(), window_x, window_y);

    featureExtractor_state=SIFT_MODE;

    while (true)
    {
        cv::Mat frame, buffer;
        if (!capture.isOpened()) break;

        capture >> buffer;
        cv::resize(buffer, frame,Size(buffer.cols/2,buffer.rows/2),0,0,INTER_LINEAR);
        setMouseCallback(windowName.c_str(), drawKeypointsCallBack, NULL);

        switch(featureExtractor_state)
        {
        case SIFT_MODE:
        {
            SiftFeatureDetector detector( nFeatures );
            std::vector<KeyPoint> keypoints;

            detector.detect(frame, keypoints);
            cv::Mat img_keypoints;
            drawKeypoints(frame, keypoints, img_keypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
            putText(img_keypoints, "SIFT MODE, right click to SURF MODE", Point(10, 20), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,0,0),2.0);

            imshow(windowName.c_str(), img_keypoints);

            break;
        }
        case SURF_MODE:
        {
            SurfFeatureDetector detector( nFeatures );
            std::vector<KeyPoint> keypoints;

            detector.detect(frame, keypoints);
            cv::Mat img_keypoints;
            drawKeypoints(frame, keypoints, img_keypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

            putText(img_keypoints, "SURF MODE, left click to SIFT MODE", Point(10, 20), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,0,0),2.0);

            imshow(windowName.c_str(), img_keypoints);

            break;
        }
        default: break;
        }

        while (cv::waitKey(100)==27)
        {
            capture.release();
            cv::destroyWindow(windowName.c_str());
        }
    }
}

void TrackFace::on_oneClassClassifier_clicked()
{
    TrackFace::capture.open(0);

    string windowName="Family-Stranger Classifier";
    cv::namedWindow(windowName.c_str(), cv::WINDOW_AUTOSIZE);
    cv::moveWindow(windowName.c_str(), window_x, window_y);

    cv::Mat trainingImages;

    for (int i=0;i<images.size();i++)
        //for (int j=1;j<=10;j++)
        trainingImages.push_back(images[i].reshape(1,1));
    //for (int i=images.size()/3*2;i<images.size();i++)
        //for (int j=1;j<=2;j++)
      //      trainingImages.push_back(images[i].reshape(1,1));

    const unsigned int noFeatures=trainingImages.rows;;
    //const unsigned int featureWidth=trainingImages.cols;

    cout << noFeatures << endl;

    float labelImages[noFeatures];
    for (int i=0;i<noFeatures/3;i++) labelImages[i]=1.0;
    for (int i=noFeatures/3;i<noFeatures/3*2;i++) labelImages[i]=1.0;
    for (int i=noFeatures/3*2;i<noFeatures;i++) labelImages[i]=-1.0;

    trainingImages.convertTo(trainingImages, CV_32FC1);
    trainingImages/=128.0;
    trainingImages-=1;

    cv::Mat labelsMat(noFeatures, 1, CV_32FC1, labelImages);

    cv::SVMParams params;
    params.svm_type=cv::SVM::NU_SVC;
    params.kernel_type=cv::SVM::RBF;
    params.nu=0.4;
    params.gamma=100;
    params.term_crit=cv::TermCriteria(CV_TERMCRIT_ITER, (int)1e7, 1e-6);

    cv::SVM svm;

    svm.train(trainingImages, labelsMat, cv::Mat(), cv::Mat(), params);

    while (true)
    {
        cv::Mat frame, buffer;
        if (!capture.isOpened()) break;

        capture >> buffer;
        cv::resize(buffer, frame,Size(buffer.cols/2,buffer.rows/2),0,0,INTER_LINEAR);

        vector<Rect_<int> > faces=haar_faces(frame);

        for (size_t i=0;i<faces.size();i++)
        {
            cv::Mat face_resized=resizeRecognitionFace(frame, faces[i]);
            face_resized=face_resized.reshape(1,1);
            face_resized.convertTo(face_resized, CV_32FC1);
            float response=svm.predict(face_resized);
            string box_text;

            cout << response << endl;

            if (response==1.0)
            {
                box_text=format("Prediction is family");
                drawFace(frame, faces[i], box_text);
            }
            else
            {
                box_text=format("Prediction is stranger");
                drawFace(frame, faces[i], box_text);
            }
        }

        cv::imshow(windowName.c_str(), frame);
        while (cv::waitKey(5)==27)
        {
            capture.release();
            cv::destroyWindow(windowName.c_str());
        }
    }
}
