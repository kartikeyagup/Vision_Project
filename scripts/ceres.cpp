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
#include "gflags/gflags.h"
#include "glog/logging.h"
using ceres::AutoDiffCostFunction;
using ceres::DynamicAutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;
struct F1 {
  template <typename T> bool operator()(const T* const x1,
                                        const T* const x2,
                                        T* residual) const {
    // f1 = x1 + 10 * x2;
    residual[0] = x1[0] + T(10.0) * x2[0];
    return true;
  }
};

struct F1_D {
  template <typename T> bool operator()(T const* const* parameters,
                                        T* residual) const {
    // f1 = x1 + 10 * x2;
    residual[0] = parameters[0][0] + T(10.0) * parameters[1][0];
    return true;
  }
};

struct F2 {
  template <typename T> bool operator()(const T* const x3,
                                        const T* const x4,
                                        T* residual) const {
    // f2 = sqrt(5) (x3 - x4)
    residual[0] = T(sqrt(5.0)) * (x3[0] - x4[0]);
    return true;
  }
};

struct F2_D {
  template <typename T> bool operator()(T const* const* parameters,
                                        T* residual) const {
    residual[0] = T(sqrt(5.0)) * (parameters[0][0] - parameters[1][0]);
    return true;
  }
};


struct F3 {
  template <typename T> bool operator()(const T* const x2,
                                        const T* const x4,
                                        T* residual) const {
    // f3 = (x2 - 2 x3)^2
    residual[0] = (x2[0] - T(2.0) * x4[0]) * (x2[0] - T(2.0) * x4[0]);
    return true;
  }
};

struct F3_D {
  template <typename T> bool operator()(T const* const* parameters,
                                        T* residual) const {
    // f3 = (x2 - 2 x3)^2
    // double *x2 = (double*) parameters[0];
    // double *x4 = (double*) parameters[1]; 
    residual[0] = (parameters[0][0] - T(2.0) * parameters[1][0]) * (parameters[0][0] - T(2.0) * parameters[1][0]);
    return true;
  }
};

struct F4 {
  template <typename T, typename T2> bool operator()(const T2* const x1,
                                        const T* const x4,
                                        T* residual) const {
    // f4 = sqrt(10) (x1 - x4)^2
    residual[0] = T(sqrt(10.0)) * (x1[0] - x4[0]) * (x1[0] - x4[0]);
    return true;
  }
};

struct F4_D {
  template <typename T> bool operator()(T const* const* parameters, 
                                        T* residual) const {
    // f4 = sqrt(10) (x1 - x4)^2
    // double *x1 = (double*) parameters[0];
    // double *x4 = (double*) parameters[1];
    residual[0] = T(sqrt(10.0)) * (parameters[0][0] - parameters[1][0]) * (parameters[0][0] - parameters[1][0]);
    return true;
  }
};

DEFINE_string(minimizer, "trust_region",
              "Minimizer type to use, choices are: line_search & trust_region");
int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  // double x1[2] =  {3,0};
  Eigen::Vector2d t;
  t(0,0) = 3;
  t(1,0) = 1;
  double x1 = 3.0;
  double x2 = -1.0;
  double x3 =  0.0;
  double x4 =  1.0;
  Problem problem;
  // Add residual terms to the problem using the using the autodiff
  // wrapper to get the derivatives automatically. The parameters, x1 through
  // x4, are modified in place.
  // DynamicAutoDiffCostFunction<F1_D> c1(new F1_D());
  // c1.SetNumResiduals(1);
  // c1.AddParameterBlock(1);
  // c1.AddParameterBlock(1);
  // std::vector<double*> p1;
  // p1.push_back(&x1);
  // p1.push_back(&x2);
  // problem.AddResidualBlock(&c1, NULL, p1);

  // DynamicAutoDiffCostFunction<F2_D> c2(new F2_D());
  // c2.SetNumResiduals(1);
  // c2.AddParameterBlock(1);
  // c2.AddParameterBlock(1);
  // std::vector<double*> p2;
  // p2.push_back(&x3);
  // p2.push_back(&x4);
  // problem.AddResidualBlock(&c2, NULL, p2);

  // DynamicAutoDiffCostFunction<F3_D> c3(new F3_D());
  // c3.SetNumResiduals(1);
  // c3.AddParameterBlock(1);
  // c3.AddParameterBlock(1);
  // std::vector<double*> p3;
  // p3.push_back(&x2);
  // p3.push_back(&x3);
  // problem.AddResidualBlock(&c3, NULL, p3);

  // DynamicAutoDiffCostFunction<F4_D> c4(new F4_D());
  // c4.SetNumResiduals(1);
  // c4.AddParameterBlock(1);
  // c4.AddParameterBlock(1);
  // std::vector<double*> p4;
  // p4.push_back(&x1);
  // p4.push_back(&x4);
  // problem.AddResidualBlock(&c4, NULL, p4);

  problem.AddResidualBlock(new AutoDiffCostFunction<F1, 1, 2, 1>(new F1),
                           NULL,
                           t.data(), &x2);
  problem.AddResidualBlock(new AutoDiffCostFunction<F2, 1, 1, 1>(new F2),
                           NULL,
                           &x3, &x4);
  problem.AddResidualBlock(new AutoDiffCostFunction<F3, 1, 1, 1>(new F3),
                           NULL,
                           &x2, &x3);
  problem.AddResidualBlock(new AutoDiffCostFunction<F4, 1, 2, 1>(new F4),
                           NULL,
                           t.data(), &x4);
  Solver::Options options;
  LOG_IF(FATAL, !ceres::StringToMinimizerType(FLAGS_minimizer,
                                              &options.minimizer_type))
      << "Invalid minimizer: " << FLAGS_minimizer
      << ", valid options are: trust_region and line_search.";
  options.max_num_iterations = 100;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;
  std::cout << "Initial x1 = " << t(0,0)
            << ", x2 = " << x2
            << ", x3 = " << x3
            << ", x4 = " << x4
            << "\n";
  // Run the solver!
  Solver::Summary summary;
  Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << "\n";
  std::cout << "Final x1 = " << t(0,0)
            << ", x2 = " << x2
            << ", x3 = " << x3
            << ", x4 = " << x4
            << "\n";
  return 0;
}
