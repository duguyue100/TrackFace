#ifndef IOFUNCTIONS_H
#define IOFUNCTIONS_H

#include "TrackFace.h"

class IOFunctions
{
public:
    IOFunctions();

    void read_images(const string & filename, const string &fn_path, vector<cv::Mat> & images, vector<int> & labels);
    string addName(string name, string fn_namedb, string fn_path);
    int addToTrain(string fn_images,string namePath, int label);
    void read_name(const string & filename, vector<string> & names);
};

#endif // IOFUNCTIONS_H
