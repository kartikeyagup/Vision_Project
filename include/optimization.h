#ifndef OPTIMIZATION_H
#define OPTIMIZATION_H

#include <vector>
#include <vector>
#include "init-helpers.h"
#include "ceres/ceres.h"
#include "ceres/dynamic_autodiff_cost_function.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cmath>
#include "common.h"

void delta(Eigen::MatrixXd &mat, Eigen::MatrixXd &delta);
int ceressolver(int, bool);
double L(Eigen::MatrixXd &input1, Eigen::MatrixXd &input2);
void warp(Eigen::MatrixXd &m, Eigen::MatrixXd &mx, Eigen::MatrixXd &my, Eigen::MatrixXd &warped);
double L1Norm(Eigen::MatrixXd &);
double L1NormWithDelta(Eigen::MatrixXd &mat);
double AddL1NormWithDelta(Eigen::MatrixXd &mat1, Eigen::MatrixXd &mat2);

#endif
