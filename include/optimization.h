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

Eigen::MatrixXd delta(Eigen::MatrixXd mat);

double L(Eigen::MatrixXd &input1, Eigen::MatrixXd &input2);
Eigen::MatrixXd warp(Eigen::MatrixXd &m, Eigen::MatrixXd &mx, Eigen::MatrixXd &my);

#endif
