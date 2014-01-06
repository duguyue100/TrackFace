#include "TrackFace.h"
#include "iofunctions.h"

IOFunctions::IOFunctions()
{
}


void IOFunctions::read_images(const string & filename, const string &fn_path, vector<cv::Mat> & images, vector<int> & labels)
{
    char separator= ';';
    ifstream file(filename.c_str(), ifstream::in);

    if (!file)
    {
        string error_message = "error";
        CV_Error(CV_StsBadArg, error_message);
    }

    string line, path, classlabel;

    while (getline(file, line))
    {
        line=fn_path+line;
        stringstream liness(line);

        getline(liness, path, separator);
        getline(liness, classlabel);

        if (!path.empty() && !classlabel.empty())
        {
            images.push_back(imread(path,0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}

string IOFunctions::addName(string name, string fn_namedb, string fn_path)
{
    string namePath=fn_path+"resources/"+name;
    if (!QDir(namePath.c_str()).exists())
    {
        QDir().mkdir(namePath.c_str());
        ofstream writeName;
        writeName.open(fn_namedb.c_str(), ios::out|ios::app);
        writeName << name << endl;
        writeName.close();
    }

    return namePath+"/";
}

int IOFunctions::addToTrain(string fn_images, string namePath, int label)
{
    ofstream writeImages;

    writeImages.open(fn_images.c_str(), ios::out|ios::app);
    writeImages << namePath << ";" << label << endl;
    writeImages.close();

    return 0;
}

void IOFunctions::read_name(const string & filename, vector<string> & names)
{
    ifstream readNames;
    readNames.open(filename.c_str(), ifstream::in);

    string name;
    while (getline(readNames, name))
    {
        names.push_back(name);
    }

    readNames.close();
}
