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
#include "ceres/dynamic_autodiff_cost_function.h"
// #include "ceres/dynamic_numericdiff_cost_function.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
using ceres::AutoDiffCostFunction;
using ceres::DynamicAutoDiffCostFunction;
using ceres::DynamicNumericDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;


Eigen::MatrixXd delta (Eigen::MatrixXd A);

struct num_dynamic_eigen_4 {
  bool operator()(double const* const* parameters, 
                  double* residual) const {
 
    Eigen::MatrixXd io = Eigen::Map<Eigen::MatrixXd>((double*) parameters[0], 100, 100);
    
    residual[0] = ((sqrt(10.0) * io(0,0) - io(1,1))*(sqrt(10.0) * io(0,0) - io(1,1)));
    residual[0] += (io(0,0) + 10*io(1,0))*(io(0,0) + 10*io(1,0));
    residual[0] += (io(1,0) - 2*io(0,1))*(io(1,0) - 2*io(0,1));
    residual[0] += sqrt(5)*(io(0,1) - io(1,1))*sqrt(5)*(io(0,1) - io(1,1));
    // // residual[0] += delta(io).norm();
    std::cout << "Residual " << residual[0] << "\n";
    // residual[0] += io.lpNorm<1>();
    return true;
  }
};

DEFINE_string(minimizer, "line_search",
              "Minimizer type to use, choices are: line_search & trust_region");
int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  // double x1[2] =  {3,0};
  // Eigen::Vector2d t;
  Eigen::MatrixXd matrix(100,100);
  double x1 = 3;
  double x2 = -1;
  double x3 = 0;
  double x4 = 2;
  matrix(0,0) = x1;
  matrix(0,1) = x2;
  matrix(1,0) = x3;
  matrix(1,1) = x4;

  // matrix << x1 ,x2,
  //           x3, x4;
  Problem problem;
  // double * X = matrix.data();
  DynamicNumericDiffCostFunction<num_dynamic_eigen_4>* c1 = new 
                        DynamicNumericDiffCostFunction<num_dynamic_eigen_4> (new num_dynamic_eigen_4());
  c1->SetNumResiduals(1);
  std::vector<double*> p1;// = new std::vector<double*>[1];;
  double *dat = matrix.data();
  std::cout << dat[0] << "\n";
  std::cout << dat[1] << "\n";
  std::cout << dat[2] << "\n";
  std::cout << dat[3] << "\n";

  c1->AddParameterBlock(100*100);
  p1.push_back(dat);
  problem.AddResidualBlock(c1, NULL, p1);

  Solver::Options options;
  // options.
  LOG_IF(FATAL, !ceres::StringToMinimizerType(FLAGS_minimizer,
                                              &options.minimizer_type))
      << "Invalid minimizer: " << FLAGS_minimizer
      << ", valid options are: trust_region and line_search.";
  options.max_num_iterations = 1000;
  options.num_threads = 8;
  options.num_linear_solver_threads = 8;
  // options.preconditioner_type = ceres::IDENTITY;
  options.line_search_direction_type = ceres::BFGS;
  options.use_inner_iterations = true;
  // options.use_explicit_schur_complement = false; 
  options.linear_solver_type = ceres::ITERATIVE_SCHUR;
  options.minimizer_progress_to_stdout = true;
  // options.trust_region_strategy_type = ceres::DOGLEG;
  options.use_nonmonotonic_steps = true;
  std::cout << "Initial x1 = " << matrix << std::endl;
            // << ", x2 = " << x2
            // << ", x3 = " << x3
            // << ", x4 = " << x4
            // << "\n";
  // Run the solver!
  Solver::Summary summary;
  Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << "\n";
  std::cout << "Final x1 = " << matrix << std::endl;
            // << ", x2 = " << x2
            // << ", x3 = " << x3
            // << ", x4 = " << x4
            // << "\n";
  return 0;
}

Eigen::MatrixXd delta(Eigen::MatrixXd A) {
  Eigen::MatrixXd answer, delx, dely;
  delx = Eigen::MatrixXd(A.rows()-1,A.cols()-1);
  dely = Eigen::MatrixXd(A.rows()-1,A.cols()-1);
  for(int i = 0 ; i < delx.rows() ; i++){
    for(int j = 0 ; j < delx.cols() ; j++){
      delx(i,j) = A(i,j) - A(i+1,j);
      dely(i,j) = A(i,j) - A(i,j+1);
    }
  }
  answer = (delx + dely)/2;
  return answer;
}