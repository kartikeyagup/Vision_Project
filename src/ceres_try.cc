// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2015 Google Inc. All rights reserved.
// http://ceres-solver.org/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: sameeragarwal@google.com (Sameer Agarwal)
//
// An example program that minimizes Powell's singular function.
//
//   F = 1/2 (f1^2 + f2^2 + f3^2 + f4^2)
//
//   f1 = x1 + 10*x2;
//   f2 = sqrt(5) * (x3 - x4)
//   f3 = (x2 - 2*x3)^2
//   f4 = sqrt(10) * (x1 - x4)^2
//
// The starting values are x1 = 3, x2 = -1, x3 = 0, x4 = 1.
// The minimum is 0 at (x1, x2, x3, x4) = 0.
//
// From: Testing Unconstrained Optimization Software by Jorge J. More, Burton S.
// Garbow and Kenneth E. Hillstrom in ACM Transactions on Mathematical Software,
// Vol 7(1), March 1981.
#include <vector>
#include "ceres/ceres.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cmath>

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

cv::Mat Io, Ib, W, A;
std::vector <cv::Mat> Vo,Vb;

double img_rows;
double img_cols;

int num_images;

int get_ij(cv::Mat &inp, int i, int j) {
  return (int) inp.at<uchar>(i,j);
}

cv::Mat elementmultiply(cv::Mat &, cv::Mat&);
cv::Mat convolve(cv::Mat& ,cv::Mat&);
cv::Mat warp(cv::Mat &, cv::Mat&);


// T* ka kya karna hai? Har jagah, residual has to be a double. 

struct data_term {
  template <typename T> bool operator()(const T* const Io,
                                        const T* const Ib,
                                        const T* const A,
                                        T* residual) const {
    // f1 = x1 + 10 * x2;
    residual[0] = T(0.0);
    for(int t=0;t<total_data.num_images;t++){
      cv::Mat iovo = warp(Io,Vo[t]);
      cv::Mat ibvb = warp(Ib,Vb[t]);
      for(int i=0;i<img_rows;i++){
        for(int j=0;j<img_cols;j++){
          if(get_ij(iovo,i,j) != 0 && get_ij(ibvb,i,j) != 0){
            residual[0]+=l1norm(get_ij(total_data.frames[t],i,j) - get_ij(iovo,i,j) - get_ij(iovo,i,j)*get_ij(ibvb,i,j);
          }
        }
      }
    	
    }
    return true;
  }
};


struct F2 {
  template <typename T> bool operator()(const T* const A,
                                        T* residual) const {
  	int param=200;
    // f2 = sqrt(5) (x3 - x4)
    residual[0] = param*T(l2norm(delta(A)));
    return true;
  }
};


struct F3 {
  template <typename T> bool operator()(const T* const Io,
                                        const T* const Ib,
                                        T* residual) const {
    // f3 = (x2 - 2 x3)^2
    int param=250;
    residual[0] = param*T(l1norm(delta(Io)) + l1norm(delta(Ib)));
    return true;
  }
};


struct F4 {
  template <typename T> bool operator()(const T* const Io,
                                        const T* const Ib,
                                        T* residual) const {
    // f4 = sqrt(10) (x1 - x4)^2
    int param=100;
    residual[0] = param*T(L(Io,Ib));
    return true;
  }
};

struct F5 {
  template <typename T> bool operator()(const T* const Vo,
                                        const T* const Vb,
                                        T* residual) const {
    // f3 = (x2 - 2 x3)^2
    int param=250;
    residual[0] = param*T(l1norm(delta(Vo)) + l1norm(delta(Vb)));
    return true;
  }
};

struct data_term2 {
  template <typename T> bool operator()(const T* const Vo,
                                        const T* const Vb,
                                        T* residual) const {
    // f1 = x1 + 10 * x2;
    residual[0] = T(0.0);
    for(int t=0;t<total_data.num_images;t++){
    	// residual[0]+=l1norm(total_data.frames[i] - elementmultiply(convolve(W,Vo),Io) - elementmultiply(elementmultiply(convolve(W,Vo),A),elementmultiply(convolve(W,Vb),Ib)));
      cv::Mat iovo = warp(Io,Vo[t]);
      cv::Mat ibvb = warp(Ib,Vb[t]);
      for(int i=0;i<img_rows;i++){
        for(int j=0;j<img_cols;j++){
          if(get_ij(iovo,i,j) != 0 && get_ij(ibvb,i,j) != 0){
            residual[0]+=l1norm(get_ij(total_data.frames[t],i,j) - get_ij(iovo,i,j) - get_ij(iovo,i,j)*get_ij(ibvb,i,j);
          }
        }
      }
    }
    return true;
  }
};

DEFINE_string(minimizer, "trust_region",
              "Minimizer type to use, choices are: line_search & trust_region");


int ceressolver(cv::Mat Io, cv::Mat Ib, cv::Mat A, std::vector<cv::Mat> Vo, std::vector<cv::Mat> Vb) {
  // google::ParseCommandLineFlags(&argc, &argv, true);
  // google::InitGoogleLogging(argv[0]);

  cv::Mat guess_Io = Io;
  cv::Mat guess_Ib = Ib;
  cv::Mat guess_A = A;
  cv::Mat guess_W = W;
  std::vector <cv::Mat> guess_Vo = Vo;
  std::vector <cv::Mat> guess_Vb = Vb;
  //Assuming the input images to be total_data.frames[i]
  Problem problem;
  // Add residual terms to the problem using the using the autodiff
  // wrapper to get the derivatives automatically. The parameters, x1 through
  // x4, are modified in place.
  problem.AddResidualBlock(new AutoDiffCostFunction<data_term, 1, 1, 1>(new data_term),
                           NULL,
                           &guess_Io, &guess_Ib, &guess_A);
  problem.AddResidualBlock(new AutoDiffCostFunction<F2, 1, 1, 1>(new F2),
                           NULL,
                           &guess_A);
  problem.AddResidualBlock(new AutoDiffCostFunction<F3, 1, 1, 1>(new F3),
                           NULL,
                           &guess_Io, &guess_Ib);
  problem.AddResidualBlock(new AutoDiffCostFunction<F4, 1, 1, 1>(new F4),
                           NULL,
                           &guess_Io, &guess_Ib);
  Solver::Options options;
  LOG_IF(FATAL, !ceres::StringToMinimizerType(FLAGS_minimizer,
                                              &options.minimizer_type))
      << "Invalid minimizer: " << FLAGS_minimizer
      << ", valid options are: trust_region and line_search.";
  options.max_num_iterations = 100;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;
  // std::cout << "Initial x1 = " << x1
  //           << ", x2 = " << x2
  //           << ", x3 = " << x3
  //           << ", x4 = " << x4
  //           << "\n";
  // // Run the solver!
  Solver::Summary summary;
  Solve(options, &problem, &summary);

  Io = guess_Io;
  Ib = guess_Ib;
  A = guess_A;

  Problem problem2;

  for(int t=0;t<num_images;t++){
    problem2.AddResidualBlock(new AutoDiffCostFunction<data_term, 1, 1, 1>(new data_term2),
                             NULL,
                             &guess_Vo[t], &guess_Vb[t]);

    problem2.AddResidualBlock(new AutoDiffCostFunction<data_term, 1, 1, 1>(new F5),
                             NULL,
                             &guess_Vo[t], &guess_Vb[t]);
  }
  Solver::Options options2;
  LOG_IF(FATAL, !ceres::StringToMinimizerType(FLAGS_minimizer,
                                              &options.minimizer_type))
      << "Invalid minimizer: " << FLAGS_minimizer
      << ", valid options are: trust_region and line_search.";
  options2.max_num_iterations = 100;
  options2.linear_solver_type = ceres::DENSE_QR;
  options2.minimizer_progress_to_stdout = true;

  Solver::Summary summary2;
  Solve(options2, &problem2, &summary2);

  
  // std::cout << summary.FullReport() << "\n";
  // std::cout << "Final x1 = " << x1
  //           << ", x2 = " << x2
  //           << ", x3 = " << x3
  //           << ", x4 = " << x4
  //           << "\n";


  // PRINT THE MATRICES IF YOU WANT TO.

  return 0;
}

cv::Mat elementmultiply(cv::Mat input1, cv::Mat input2){
	cv::Mat result;
	input1.copyTo(result);
	for(int i = 0 ; i < result.rows ; i++){
		for(int j = 0 ; j < result.cols ; j++){
			result[i][j] = input1[i][j] * input2[i][j];
		}
	}
	return result;
}

cv::Mat convolve(cv::Mat src, cv::Mat kernel){
	cv::Mat result;
	cv::filter2D(src, result, -1, kernel, cv::Point(-1,1), 0, cv::BORDER_DEFAULT);
	return result;
}

cv::Mat delta(cv::Mat mat) {
  cv::Mat grad_x,grad_y,grad;
  // mat.copyTo(delta);
  cv::Mat kernel;
  cv::Sobel(mat, grad_x, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
  cv::Sobel(mat, grad_y, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);
  convertScaleAbs( grad_x, grad_x );
  convertScaleAbs( grad_y, grad_y );

  addWeighted(grad_x, 0.5, grad_y, 0.5, 0, grad );

  return grad;
}

double l1norm(cv::Mat input){
	double answer = 0;
	for(int j = 0 ; j < input.cols ; j++){
		double mysum = 0;
		for(int i = 0 ; i < input.rows ; i++){
			mysum += abs(get_ij(input,i,j));
		}
		if(mysum > answer){
			answer = mysum;
		}
	}
	return answer;
}

double l2norm(cv::Mat input){
	// find eigen values of input*input'
	// Largest eigen value ka under-root is the answer
  Eigen::Map<Eigen::Matrix3f> eigen(input.data());
  // Eigen::MatrixXf eigenTranspose = eigen.transpose();
  // Eigen::MatrixXf res = eigen*eigenTranspose;
  // Eigen::VectorXcd eivals = res.eigenvalues();
  // double max = 0;
  // for(int i = 0 ; i < eivals.size() ; i++){
  //   double mymax = eivals[i];
  //   if(mymax > max){
  //     max = mymax;
  //   }
  // }
  double x = eigen.squaredNorm();

	return sqrt(x);
}

double L(cv::Mat input1, cv::Mat input2){
	double product = 0;
	input1 = delta(input1);
	input2 = delta(input2);
	for(int i = 0 ; i < input1.rows ; i++){
		for(int j = 0 ; j < input1.cols ; j++){
			product + = input1[i][j] * input1[i][j] * input2[i][j] * input2[i][j];
		}
	}
	return product;
}

cv::Mat warp(cv::Mat mat, cv::Mat kernel){
  cv::Mat result;
  // mat.copyTo(result);
  result = cv::Mat::zeros(mat.rows, mat.cols, CV_8UC1);
  double limy = mat.rows;
  double limx = mat.cols;
  for(int i=0;i<limy;i++){
    for(int j=0; j<limx;j++){
      double posx = j + get_ij(kernel,i,j);
      double posy = i + get_ij(kernel,i,j);
      if(posx<=limx && posy<=limy)
        result[i][j] = get_ij(mat,i,j); 
    }
  }
}