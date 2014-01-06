#ifndef GRABFACEFORM_H
#define GRABFACEFORM_H

#include <QDialog>
#include "SharedLibraries.h"

namespace Ui {
class GrabFaceForm;
}

class GrabFaceForm : public QDialog
{
    Q_OBJECT
    
public:
    explicit GrabFaceForm(QWidget *parent = 0);
    ~GrabFaceForm();

    string msg;

    string getMsg();

private slots:

    void on_grabClose_clicked();

    void on_grabClear_clicked();

    void on_grabSubmit_clicked();

private:
    Ui::GrabFaceForm *ui;
};

#endif // GRABFACEFORM_H
