#include "grabfaceform.h"
#include "ui_grabfaceform.h"

GrabFaceForm::GrabFaceForm(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::GrabFaceForm)
{
    ui->setupUi(this);
}

GrabFaceForm::~GrabFaceForm()
{
    delete ui;
}

void GrabFaceForm::on_grabClose_clicked()
{
    this->close();
}

void GrabFaceForm::on_grabClear_clicked()
{
    ui->grabName->clear();
}

void GrabFaceForm::on_grabSubmit_clicked()
{
    msg=ui->grabName->toPlainText().toStdString();
}

string GrabFaceForm::getMsg()
{
    return msg;
}
