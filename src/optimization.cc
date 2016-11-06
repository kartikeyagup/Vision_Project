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

#include "optimization.h"

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

Eigen::MatrixXf Io, Ib, A;
std::vector<motion_field> Vo, Vb;

int img_rows;
int img_cols;
int num_images;

total_data input;

DEFINE_string(minimizer, "trust_region",
              "Minimizer type to use, choices are: line_search & trust_region");


// T* ka kya karna hai? Har jagah, residual has to be a double. 

struct data_term {
  template <typename T> bool operator()(const T* const Io,
                                        const T* const Ib,
                                        const T* const A,
                                        T* residual) const {
    residual[0] = T(0.0);
    for(int t=0; t<num_images; t++) {
      Eigen::MatrixXf iovo = warp(Io, Vo[t]);
      Eigen::MatrixXf ibvb = warp(Ib, Vb[t]);
      Eigen::MatrixXf iavo = warp(A, Vo[t]);
      Eigen::MatrixXf iavoibvb = iavo * ibvb;
      residual[0] += (input.normalised_frames[t] - iovo - iavoibvb).lpNorm<1>();
    }
    return true;
  }
};

struct F2 {
  template <typename T> bool operator()(const T* const A,
                                        T* residual) const {
  	int param=200;
    residual[0] = param*T(delta(A).norm());
    return true;
  }
};

struct F3 {
  template <typename T> bool operator()(const T* const Io,
                                        const T* const Ib,
                                        T* residual) const {
    int param=250;
    Eigen::MatrixXf io = Io[0];
    Eigen::MatrixXf ib = Ib[0];
    residual[0] = param*T((delta(io)).lpNorm<1>() + delta(ib).lpNorm<1>()) ;
    return true;
  }
};

struct F4 {
  template <typename T> bool operator()(const T* const Io,
                                        const T* const Ib,
                                        T* residual) const {
    // f4 = sqrt(10) (x1 - x4)^2
    int param=100;
    residual[0] = param*T(L(Io, Ib));
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
    for(int t=0; t<num_images; t++) {
      Eigen::MatrixXf iovo = warp(Io, Vo[t]);
      Eigen::MatrixXf ibvb = warp(Ib, Vb[t]);
      Eigen::MatrixXf iavo = warp(A, Vo[t]);
      Eigen::MatrixXf iavoibvb = iavo * ibvb;
      residual[0] += (input.normalised_frames[t] - iovo - iavoibvb).lpNorm<1>();
    }
    return true;
  }
};


// int ceressolver(cv::Mat Io, cv::Mat Ib, cv::Mat A, std::vector<cv::Mat> Vo, std::vector<cv::Mat> Vb) {
//   // google::ParseCommandLineFlags(&argc, &argv, true);
//   // google::InitGoogleLogging(argv[0]);

//   cv::Mat guess_Io = Io;
//   cv::Mat guess_Ib = Ib;
//   cv::Mat guess_A = A;
//   cv::Mat guess_W = W;
//   std::vector <cv::Mat> guess_Vo = Vo;
//   std::vector <cv::Mat> guess_Vb = Vb;
//   //Assuming the input images to be total_data.frames[i]
//   Problem problem;
//   // Add residual terms to the problem using the using the autodiff
//   // wrapper to get the derivatives automatically. The parameters, x1 through
//   // x4, are modified in place.
//   problem.AddResidualBlock(new AutoDiffCostFunction<data_term, 1, 1, 1>(new data_term),
//                            NULL,
//                            &guess_Io, &guess_Ib, &guess_A);
//   problem.AddResidualBlock(new AutoDiffCostFunction<F2, 1, 1, 1>(new F2),
//                            NULL,
//                            &guess_A);
//   problem.AddResidualBlock(new AutoDiffCostFunction<F3, 1, 1, 1>(new F3),
//                            NULL,
//                            &guess_Io, &guess_Ib);
//   problem.AddResidualBlock(new AutoDiffCostFunction<F4, 1, 1, 1>(new F4),
//                            NULL,
//                            &guess_Io, &guess_Ib);
//   Solver::Options options;
//   LOG_IF(FATAL, !ceres::StringToMinimizerType(FLAGS_minimizer,
//                                               &options.minimizer_type))
//       << "Invalid minimizer: " << FLAGS_minimizer
//       << ", valid options are: trust_region and line_search.";
//   options.max_num_iterations = 100;
//   options.linear_solver_type = ceres::DENSE_QR;
//   options.minimizer_progress_to_stdout = true;
//   // std::cout << "Initial x1 = " << x1
//   //           << ", x2 = " << x2
//   //           << ", x3 = " << x3
//   //           << ", x4 = " << x4
//   //           << "\n";
//   // // Run the solver!
//   Solver::Summary summary;
//   Solve(options, &problem, &summary);

//   Io = guess_Io;
//   Ib = guess_Ib;
//   A = guess_A;

//   Problem problem2;

//   for(int t=0;t<num_images;t++){
//     problem2.AddResidualBlock(new AutoDiffCostFunction<data_term, 1, 1, 1>(new data_term2),
//                              NULL,
//                              &guess_Vo[t], &guess_Vb[t]);

//     problem2.AddResidualBlock(new AutoDiffCostFunction<data_term, 1, 1, 1>(new F5),
//                              NULL,
//                              &guess_Vo[t], &guess_Vb[t]);
//   }
//   Solver::Options options2;
//   LOG_IF(FATAL, !ceres::StringToMinimizerType(FLAGS_minimizer,
//                                               &options.minimizer_type))
//       << "Invalid minimizer: " << FLAGS_minimizer
//       << ", valid options are: trust_region and line_search.";
//   options2.max_num_iterations = 100;
//   options2.linear_solver_type = ceres::DENSE_QR;
//   options2.minimizer_progress_to_stdout = true;

//   Solver::Summary summary2;
//   Solve(options2, &problem2, &summary2);

  
//   // std::cout << summary.FullReport() << "\n";
//   // std::cout << "Final x1 = " << x1
//   //           << ", x2 = " << x2
//   //           << ", x3 = " << x3
//   //           << ", x4 = " << x4
//   //           << "\n";


//   // PRINT THE MATRICES IF YOU WANT TO.

//   return 0;
// }

// TODO Shrey
Eigen::MatrixXf delta(Eigen::MatrixXf mat) {
  // cv::Mat grad_x, grad_y, grad;
  // // mat.copyTo(delta);
  // cv::Mat kernel;
  // cv::Sobel(mat, grad_x, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
  // cv::Sobel(mat, grad_y, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);
  // convertScaleAbs( grad_x, grad_x );
  // convertScaleAbs( grad_y, grad_y );

  // addWeighted(grad_x, 0.5, grad_y, 0.5, 0, grad );
  Eigen::MatrixXf answer;
  return answer;
}

double L(Eigen::MatrixXf &input1, Eigen::MatrixXf &input2) {
	double product = 0;
	Eigen::MatrixXf del1 = delta(input1);
	Eigen::MatrixXf del2 = delta(input2);
	for(int i = 0 ; i < del1.rows() ; i++){
		for(int j = 0 ; j < del1.cols() ; j++){
			product += del1(i,j) * del1(i,j) * del2(i,j) * del2(i,j);
		}
	}
	return product;
}

Eigen::MatrixXf warp(Eigen::MatrixXf &mat, motion_field &motion) {
  Eigen::MatrixXf result(mat.rows(), mat.cols());
  result.setZero();
  double limy = mat.rows();
  double limx = mat.cols();
  assert(limx = motion.data[0].size());
  assert(limy = motion.data.size());
  for(int i=0; i<limy; i++){
    for(int j=0; j<limx; j++){
      int posx = j + motion.getx(i,j);
      int posy = i + motion.gety(i,j);
      if(posx<=limx && posy<=limy)
        result(posy, posx) = mat(i,j); 
    }
  }
  return result;
}
