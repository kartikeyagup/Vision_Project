#ifndef OPTIMIZATION_H
#define OPTIMIZATION_H

#include <vector>
#include <vector>
#include "init-helpers.h"
#include "ceres/ceres.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cmath>
#include "common.h"

Eigen::MatrixXf delta(Eigen::MatrixXf mat);

double L(Eigen::MatrixXf &input1, Eigen::MatrixXf &input2);
Eigen::MatrixXf warp(Eigen::MatrixXf &m, motion_field & motion);

#endif
