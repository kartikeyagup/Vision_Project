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
#include "ceres/dynamic_autodiff_cost_function.h"

using ceres::AutoDiffCostFunction;
using ceres::DynamicAutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

Eigen::MatrixXd Orig_Io, Orig_Ib, Orig_A; 
std::vector<Eigen::MatrixXd> Orig_VoX, Orig_VoY, Orig_VbX, Orig_VbY;

int img_rows;
int img_cols;
int num_images;

total_data input;

DEFINE_string(minimizer, "trust_region",
              "Minimizer type to use, choices are: line_search & trust_region");
 
struct dynamic_data_term {
  template <typename T> bool operator()(T const* const* paramters,
                                        T* residual) const {
    // residual[0] = T(0.0); // Might have to remove this
    Eigen::MatrixXd io = Eigen::Map<Eigen::MatrixXd>((double*) paramters[0], img_rows, img_cols);
    Eigen::MatrixXd ib = Eigen::Map<Eigen::MatrixXd>((double*) paramters[1], img_rows, img_cols);
    Eigen::MatrixXd a  = Eigen::Map<Eigen::MatrixXd>((double*) paramters[2], img_rows, img_cols);

    for(int t=0; t<num_images; t++) {
      Eigen::MatrixXd iovo = warp(io, Orig_VoX[t], Orig_VoY[t]);
      Eigen::MatrixXd ibvb = warp(ib, Orig_VbX[t], Orig_VbY[t]);
      Eigen::MatrixXd iavo = warp(a, Orig_VoX[t], Orig_VoY[t]);
      // TODO Element wise multiplication
      Eigen::MatrixXd iavoibvb = iavo * ibvb;
      residual[0] += T((input.normalised_frames[t] - iovo - iavoibvb).lpNorm<1>());
    }
    return true;
  }
};

// struct data_term {
//   template <typename T> bool operator()(const T* const Io,
//                                         const T* const Ib,
//                                         const T* const A,
//                                         T* residual) const {
//     residual[0] = T(0.0);
//     Eigen::MatrixXd io = Eigen::Map<Eigen::MatrixXd>((double*) Io, img_rows, img_cols);
//     Eigen::MatrixXd ib = Eigen::Map<Eigen::MatrixXd>((double*) Ib, img_rows, img_cols);
//     Eigen::MatrixXd a  = Eigen::Map<Eigen::MatrixXd>((double*) A, img_rows, img_cols);
      
//     for(int t=0; t<num_images; t++) {
//       Eigen::MatrixXd iovo = warp(io, Orig_VoX[t], Orig_VoY[t]);
//       Eigen::MatrixXd ibvb = warp(ib, Orig_VbX[t], Orig_VbY[t]);
//       Eigen::MatrixXd iavo = warp(a, Orig_VoX[t], Orig_VoY[t]);
//       // TODO Element wise multiplication
//       Eigen::MatrixXd iavoibvb = iavo * ibvb;      
//       residual[0] += T((input.normalised_frames[t] - iovo - iavoibvb).lpNorm<1>());
//     }
//     return true;
//   }
// };

// struct F2 {
//   template <typename T> bool operator()(const T* const A,
//                                         T* residual) const {
//   	int param=200;
//     Eigen::MatrixXd a = Eigen::Map<Eigen::MatrixXd>((double*)A, img_rows, img_cols);
//     residual[0] = T(param * delta(a).norm());
//     return true;
//   }
// };

// struct F3 {
//   template <typename T> bool operator()(const T* const Io,
//                                         const T* const Ib,
//                                         T* residual) const {
//     int param=250;
//     Eigen::MatrixXd io = Eigen::Map<Eigen::MatrixXd>((double*)Io, img_rows, img_cols);
//     Eigen::MatrixXd ib = Eigen::Map<Eigen::MatrixXd>((double*)Ib, img_rows, img_cols);
//     residual[0] = T(param*((delta(io)).lpNorm<1>() + delta(ib).lpNorm<1>())) ;
//     return true;
//   }
// };

// struct F4 {
//   template <typename T> bool operator()(const T* const Io,
//                                         const T* const Ib,
//                                         T* residual) const {
//     int param=100;
//     Eigen::MatrixXd io = Eigen::Map<Eigen::MatrixXd>((double*)Io, img_rows, img_cols);
//     Eigen::MatrixXd ib = Eigen::Map<Eigen::MatrixXd>((double*)Ib, img_rows, img_cols);
//     residual[0] = T(param*L(io, ib));
//     return true;
//   }
// };

// struct F5 {
//   template <typename T> bool operator()(const T* const VoX,
//                                         const T* const VbX,
//                                         const T* const VoY,
//                                         const T* const VbY,
//                                         T* residual) const {
//     // f3 = (x2 - 2 x3)^2
//     int param=250;
//     Eigen::MatrixXd vox = Eigen::Map<Eigen::MatrixXd>((double*) VoX, img_rows, img_cols);
//     Eigen::MatrixXd vbx = Eigen::Map<Eigen::MatrixXd>((double*) VbX, img_rows, img_cols);
//     Eigen::MatrixXd voy = Eigen::Map<Eigen::MatrixXd>((double*) VoY, img_rows, img_cols);
//     Eigen::MatrixXd vby = Eigen::Map<Eigen::MatrixXd>((double*) VbY, img_rows, img_cols);
//     residual[0] = T(param*(delta(vox).lpNorm<1>() + delta(vbx).lpNorm<1>() + 
//                               delta(voy).lpNorm<1>() + delta(vby).lpNorm<1>()));
//     return true;
//   }
// };

// TODO: Feed in image id somewhere
// struct data_term2 {
//   template <typename T> bool operator()(const T* const VoX,
//                                         const T* const VbX,
//                                         const T* const VoY,
//                                         const T* const VbY,
//                                         T* residual) const {
//     residual[0] = T(0.0);
//     Eigen::MatrixXd vox = Eigen::Map<Eigen::MatrixXd>((double*) VoX, img_rows, img_cols);
//     Eigen::MatrixXd vbx = Eigen::Map<Eigen::MatrixXd>((double*) VbX, img_rows, img_cols);
//     Eigen::MatrixXd voy = Eigen::Map<Eigen::MatrixXd>((double*) VoY, img_rows, img_cols);
//     Eigen::MatrixXd vby = Eigen::Map<Eigen::MatrixXd>((double*) VbY, img_rows, img_cols);
//     int img_id = 0;

//     Eigen::MatrixXd iovo = warp(Orig_Io, vox, voy);
//     Eigen::MatrixXd ibvb = warp(Orig_Ib, vbx, vby);
//     Eigen::MatrixXd iavo = warp(Orig_A, vox, voy);
//     // TODO: Element wise multiplication
//     Eigen::MatrixXd iavoibvb = iavo * ibvb;
//     residual[0] += T((input.normalised_frames[img_id] - iovo - iavoibvb).lpNorm<1>());
//     return true;
//   }
// };


int ceressolver() {
  // google::ParseCommandLineFlags(&argc, &argv, true);
  // google::InitGoogleLogging(argv[0]);

  // Eigen::MatrixXd guess_Io = Io;
  // Eigen::MatrixXd guess_Ib = Ib;
  // Eigen::MatrixXd guess_A = A;
  // std::vector<motion_field> guess_Vo = Vo;
  // std::vector<motion_field> guess_Vb = Vb;
  //Assuming the input images to be total_data.frames[i]
  Problem problem;
  // Add residual terms to the problem using the using the autodiff
  // wrapper to get the derivatives automatically. The parameters, x1 through
  // x4, are modified in place.
  // const int dim = img_rows*img_cols;
  const int dim = 1280*720/16;
  DynamicAutoDiffCostFunction<dynamic_data_term> data_term(new dynamic_data_term());
  data_term.SetNumResiduals(1);
  data_term.AddParameterBlock(dim);
  data_term.AddParameterBlock(dim);
  data_term.AddParameterBlock(dim);
  std::vector<double *> v;
  v.push_back(Orig_Io.data());
  v.push_back(Orig_Ib.data());
  v.push_back(Orig_A.data());
  problem.AddResidualBlock(&data_term, NULL, v);
  // problem.AddResidualBlock(new AutoDiffCostFunction<data_term, 1, dim, dim, dim>(new data_term),
  //                          NULL,
  //                          Orig_Io.data(), Orig_Ib.data(), Orig_A.data());
  // problem.AddResidualBlock(new AutoDiffCostFunction<F2, 1, dim>(new F2),
  //                          NULL,
  //                          Orig_A.data());
  // problem.AddResidualBlock(new AutoDiffCostFunction<F3, 1, dim, dim>(new F3),
  //                          NULL,
  //                          Orig_Io.data(), Orig_Ib.data());
  // problem.AddResidualBlock(new AutoDiffCostFunction<F4, 1, dim, dim>(new F4),
  //                          NULL,
  //                          Orig_A.data(), Orig_Ib.data());
  Solver::Options options;
  LOG_IF(FATAL, !ceres::StringToMinimizerType(FLAGS_minimizer,
                                              &options.minimizer_type))
      << "Invalid minimizer: " << FLAGS_minimizer
      << ", valid options are: trust_region and line_search.";
  options.max_num_iterations = 100;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = true;
  // std::cout << "Initial x1 = " << x1
  //           << ", x2 = " << x2
  //           << ", x3 = " << x3
  //           << ", x4 = " << x4
  //           << "\n";
  // // Run the solver!
  Solver::Summary summary;
  Solve(options, &problem, &summary);

  // for(int t=0;t<num_images;t++){
  //   // TODO: Set t
  //   Problem problem2;
  //   problem2.AddResidualBlock(new AutoDiffCostFunction<data_term2, 1, dim, dim, dim, dim>(new data_term2),
  //                            NULL,
  //                            Orig_VoX[t].data() , Orig_VbX[t].data(), Orig_VoY[t].data(), Orig_VbY[t].data());

  //   problem2.AddResidualBlock(new AutoDiffCostFunction<F5, 1, dim, dim, dim, dim>(new F5),
  //                            NULL,
  //                            Orig_VoX[t].data() , Orig_VbX[t].data(), Orig_VoY[t].data(), Orig_VbY[t].data());
  // }
  // Solver::Options options2;
  // LOG_IF(FATAL, !ceres::StringToMinimizerType(FLAGS_minimizer,
  //                                             &options.minimizer_type))
  //     << "Invalid minimizer: " << FLAGS_minimizer
  //     << ", valid options are: trust_region and line_search.";
  // options2.max_num_iterations = 100;
  // options2.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  // options2.minimizer_progress_to_stdout = true;

  // Solver::Summary summary2;
  // Solve(options2, &problem2, &summary2);
  
  // std::cout << summary.FullReport() << "\n";
  // std::cout << "Final x1 = " << x1
  //           << ", x2 = " << x2
  //           << ", x3 = " << x3
  //           << ", x4 = " << x4
  //           << "\n";


  // PRINT THE MATRICES IF YOU WANT TO.

  return 0;
}

// TODO Shrey
Eigen::MatrixXd delta(Eigen::MatrixXd mat) {
  // cv::Mat grad_x, grad_y, grad;
  // // mat.copyTo(delta);
  // cv::Mat kernel;
  // cv::Sobel(mat, grad_x, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
  // cv::Sobel(mat, grad_y, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);
  // convertScaleAbs( grad_x, grad_x );
  // convertScaleAbs( grad_y, grad_y );

  // addWeighted(grad_x, 0.5, grad_y, 0.5, 0, grad );
  Eigen::MatrixXd answer;
  return answer;
}

double L(Eigen::MatrixXd &input1, Eigen::MatrixXd &input2) {
	double product = 0;
	Eigen::MatrixXd del1 = delta(input1);
	Eigen::MatrixXd del2 = delta(input2);
	for(int i = 0 ; i < del1.rows() ; i++){
		for(int j = 0 ; j < del1.cols() ; j++){
			product += del1(i,j) * del1(i,j) * del2(i,j) * del2(i,j);
		}
	}
	return product;
}

Eigen::MatrixXd warp(Eigen::MatrixXd &mat, Eigen::MatrixXd &mx, Eigen::MatrixXd &my) {
  Eigen::MatrixXd result(mat.rows(), mat.cols());
  result.setZero();
  double limy = mat.rows();
  double limx = mat.cols();
  // assert(limx = motion.data[0].size());
  // assert(limy = motion.data.size());
  for(int i=0; i<limy; i++){
    for(int j=0; j<limx; j++){
      int posx = j + mx(i,j);
      int posy = i + my(i,j);
      if(posx<=limx && posy<=limy)
        result(posy, posx) = mat(i,j); 
    }
  }
  return result;
}
