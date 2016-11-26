#ifndef COMMON_H
#define COMMON_H

#include <vector>
#include <utility>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>

cv::Mat convertimg(Eigen::MatrixXd img);

// void printimage(cv::Mat input){
//   cv::imwrite("build/output.png",input);
// }

Eigen::MatrixXd DownSampleMat(Eigen::MatrixXd mt, int factor);

Eigen::MatrixXd UpSampleMat(Eigen::MatrixXd mt, int factor);

Eigen::MatrixXd DownSampleFromCvMat(cv::Mat img, int cls, int rws);

Eigen::MatrixXd normalize(cv::Mat inp);

void Fix(Eigen::MatrixXd &mat);

#endif
