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
#include "init-helpers.h"
using ceres::AutoDiffCostFunction;
using ceres::DynamicAutoDiffCostFunction;
using ceres::DynamicNumericDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

Eigen::MatrixXd Orig_Io, Orig_Ib, Orig_A, Orig_Io_DS, Orig_Ib_DS; 
std::vector<Eigen::MatrixXd> Orig_VoX, Orig_VoY, Orig_VbX, Orig_VbY;
std::vector<Eigen::MatrixXd> Orig_VoX_US, Orig_VoY_US, Orig_VbX_US, Orig_VbY_US;
Eigen::MatrixXd iovo, ibvb;
Eigen::MatrixXd bn, s;
Eigen::MatrixXd ib0, ib1, ib2;

int img_rows;
int numiter;
int img_cols;
int num_images;

total_data input;

DEFINE_string(minimizer, "trust_region",
              "Minimizer type to use, choices are: line_search & trust_region");
 
struct dynamic_data_term {
  private:
    int img_id;
  public:
  bool operator()(double const* const* parameters,
                                        double* residual) const {
    // double param1 = 1;
    // double param2 = 100;
    // double param3 = 1000;

    Eigen::MatrixXd io = Eigen::Map<Eigen::MatrixXd>((double*) parameters[0], img_rows, img_cols);
    // Eigen::MatrixXd ib = Eigen::Map<Eigen::MatrixXd>((double*) parameters[1], img_rows, img_cols);
    ib0 = input.base_img_normalised - io;
    // double abc=0;
    // Eigen::MatrixXd iovo(img_rows, img_cols);
    warp(io, Orig_VoX_US[img_id], Orig_VoY_US[img_id], iovo);
    // Eigen::MatrixXd ibvb(img_rows, img_cols);
    warp(ib0, Orig_VbX_US[img_id], Orig_VbY_US[img_id], ibvb);
    s = input.normalised_frames[img_id] - iovo - ibvb;
    // abc += L1Norm(s);
    residual[0] = sqrt(L1Norm(s));
    return true;
  }

  dynamic_data_term(int im) {
    img_id = im;
  }
};

struct dynamic_norm_l1_ioib {
  bool operator()(double const* const* parameters,
                                        double* residual) const {
    double param = 0.1;
    Eigen::MatrixXd io = Eigen::Map<Eigen::MatrixXd>((double*) parameters[0], img_rows, img_cols);
    // Eigen::MatrixXd ib = Eigen::Map<Eigen::MatrixXd>((double*) parameters[1], img_rows, img_cols);
    ib1 = input.base_img_normalised - io;
    residual[0] = sqrt((param * ((double) (L1NormWithDelta(io) + L1NormWithDelta(ib1)))));
    return true;
  }
};

struct dynamic_l {
  bool operator()(double const* const* parameters,
                                        double* residual) const {
    double param = 3000;
    Eigen::MatrixXd io = Eigen::Map<Eigen::MatrixXd>((double*) parameters[0], img_rows, img_cols);
    // Eigen::MatrixXd ib = Eigen::Map<Eigen::MatrixXd>((double*) parameters[1], img_rows, img_cols);
    ib2 = input.base_img_normalised - io;
    residual[0] = sqrt((param * (double) (L(io,ib2))));
    return true;
  }
};

struct forgotten {
  bool operator()(double const* const* parameters,
                                        double* residual) const {
    double param = 100000;
    Eigen::MatrixXd io = Eigen::Map<Eigen::MatrixXd>((double*) parameters[0], img_rows, img_cols);
    ib2 = input.base_img_normalised - io;
    // ib2 = ib2 * -1;
    residual[0] = sqrt(param * std::max(0.0, -(ib2.minCoeff())));
    return true;
  }
};

struct dynamic_data_term_motion {

private:
  int img_id;
public:
  bool operator()(double const* const* parameters,
                                        double* residual) const {
    Eigen::MatrixXd vox = Eigen::Map<Eigen::MatrixXd>((double*) parameters[0], Orig_VoX[0].rows(), Orig_VoX[0].cols());
    Eigen::MatrixXd voy = Eigen::Map<Eigen::MatrixXd>((double*) parameters[1], Orig_VoX[0].rows(), Orig_VoX[0].cols());
    Eigen::MatrixXd vbx = Eigen::Map<Eigen::MatrixXd>((double*) parameters[2], Orig_VoX[0].rows(), Orig_VoX[0].cols());
    Eigen::MatrixXd vby = Eigen::Map<Eigen::MatrixXd>((double*) parameters[3], Orig_VoX[0].rows(), Orig_VoX[0].cols());
    // Eigen::MatrixXd iovo(img_rows, img_cols);
    warp(Orig_Io_DS, vox, voy, iovo);
    // Eigen::MatrixXd ibvb(img_rows, img_cols);
    warp(Orig_Ib_DS, vbx, vby, ibvb);
    bn = input.normalised_frames[img_id] - iovo - ibvb;
    residual[0] = sqrt(L1Norm(bn));
    return true;
  }

  dynamic_data_term_motion(int im){
    img_id = im;
  }
};

struct dynamic_motion {
  bool operator()(double const* const* parameters,
                                        double* residual) const {
    double param1 = 0.25;

    Eigen::MatrixXd vox = Eigen::Map<Eigen::MatrixXd>((double*) parameters[0], Orig_VoX[0].rows(), Orig_VoX[0].cols());
    Eigen::MatrixXd voy = Eigen::Map<Eigen::MatrixXd>((double*) parameters[1], Orig_VoX[0].rows(), Orig_VoX[0].cols());
    Eigen::MatrixXd vbx = Eigen::Map<Eigen::MatrixXd>((double*) parameters[2], Orig_VoX[0].rows(), Orig_VoX[0].cols());
    Eigen::MatrixXd vby = Eigen::Map<Eigen::MatrixXd>((double*) parameters[3], Orig_VoX[0].rows(), Orig_VoX[0].cols());
    residual[0] = param1 * ((AddL1NormWithDelta(vox, voy)) + (AddL1NormWithDelta(vbx, vby)));
    residual[0] = sqrt(residual[0]);
    return true;
  }
};

int ceressolver(int factor, bool pyramid) {
  numiter = 0;
  iovo = Eigen::MatrixXd(img_rows, img_cols);
  ibvb = Eigen::MatrixXd(img_rows, img_cols);
  bn = Eigen::MatrixXd(img_rows, img_cols);
  s = Eigen::MatrixXd(img_rows, img_cols);
  ib0 = Eigen::MatrixXd(img_rows, img_cols);
  ib1 = Eigen::MatrixXd(img_rows, img_cols);
  ib2 = Eigen::MatrixXd(img_rows, img_cols);
  
  std::cout<<"Starting on the first problem\n";
  Problem problem;
  
  int dim = img_cols * img_rows;
  Eigen::Matrix2d matrix;
  double* m = Orig_Io.data();
  for(int i=0; i<num_images; i++){
    DynamicNumericDiffCostFunction<dynamic_data_term>* c1 = new 
                                          DynamicNumericDiffCostFunction<dynamic_data_term> (new dynamic_data_term(i));
                        
    c1->SetNumResiduals(1);
    std::vector<double*> v;
    c1->AddParameterBlock(dim);
    // c1->AddParameterBlock(dim);
    v.push_back(Orig_Io.data());
    // v.push_back(Orig_Ib.data());
    problem.AddResidualBlock(c1, NULL, v);
  }

  DynamicNumericDiffCostFunction<dynamic_norm_l1_ioib>* c3 = new 
                                        DynamicNumericDiffCostFunction<dynamic_norm_l1_ioib> (new dynamic_norm_l1_ioib());
                      
  c3->SetNumResiduals(1);
  std::vector<double*> v3;
  c3->AddParameterBlock(dim);
  // c3->AddParameterBlock(dim);
  v3.push_back(Orig_Io.data());
  // v3.push_back(Orig_Ib.data());
  problem.AddResidualBlock(c3, NULL, v3);

  DynamicNumericDiffCostFunction<dynamic_l>* c4 = new 
                                        DynamicNumericDiffCostFunction<dynamic_l> (new dynamic_l());
  c4->SetNumResiduals(1);
  std::vector<double*> v4;
  c4->AddParameterBlock(dim);
  // c4->AddParameterBlock(dim);
  v4.push_back(Orig_Io.data());
  // v4.push_back(Orig_Ib.data());
  problem.AddResidualBlock(c4, NULL, v4);

 DynamicNumericDiffCostFunction<forgotten>* c5 = new 
                                        DynamicNumericDiffCostFunction<forgotten> (new forgotten());
                      
  c5->SetNumResiduals(1);
  std::vector<double*> v0;
  c5->AddParameterBlock(dim);
  // c3->AddParameterBlock(dim);
  v0.push_back(Orig_Io.data());
  // v3.push_back(Orig_Ib.data());
  problem.AddResidualBlock(c5, NULL, v0);


  Solver::Options options;

  LOG_IF(FATAL, !ceres::StringToMinimizerType(FLAGS_minimizer,
                                              &options.minimizer_type))
      << "Invalid minimizer: " << FLAGS_minimizer
      << ", valid options are: trust_region and line_search.";
  if(pyramid)
    options.max_num_iterations = 100;
  else
    options.max_num_iterations = 100;
  options.linear_solver_type = ceres::CGNR;
  options.preconditioner_type = ceres::IDENTITY;
  options.minimizer_progress_to_stdout = true;
  options.update_state_every_iteration = true;
  options.num_threads = 40;
  options.num_linear_solver_threads = 40;
  // options.initial_trust_region_radius = 1e-2;
  // options.max_trust_region_radius = 1;
  if(pyramid){
    options.use_nonmonotonic_steps = true;
    options.parameter_tolerance = 1e-3;
  }
  // options.min_trust_region_radius = 1e-5;
  Solver::Summary summary;
  Solve(options, &problem, &summary);
  std::cout << "HAHAHA" <<std::endl;
  std::cout << summary.FullReport() << "\n";

  Orig_Ib = input.base_img_normalised - Orig_Io;
  Fix(Orig_Io);
  Fix(Orig_Ib);
  Orig_Io_DS = DownSampleMat(Orig_Io, factor);
  Orig_Ib_DS = DownSampleMat(Orig_Ib, factor);

  numiter = 0;
  dim /= (factor*factor);
  assert(Orig_VoX[0].rows()*Orig_VoX[0].cols() == dim);
  assert(Orig_VoX[0].rows() == Orig_Io_DS.rows());
  assert(Orig_VoX[0].cols() == Orig_Io_DS.cols());
  std::cout<<"Starting on the second problem\n";
  for(int t=0;t<num_images;t++){
    Problem problem2;
    DynamicNumericDiffCostFunction<dynamic_data_term_motion>* data_term_motion = new 
                                        DynamicNumericDiffCostFunction<dynamic_data_term_motion> (new dynamic_data_term_motion(t));
    data_term_motion->SetNumResiduals(1);
    data_term_motion->AddParameterBlock(dim);
    data_term_motion->AddParameterBlock(dim);
    data_term_motion->AddParameterBlock(dim);
    data_term_motion->AddParameterBlock(dim);
    std::vector<double *> v5;
    v5.push_back(Orig_VoX[t].data());
    v5.push_back(Orig_VoY[t].data());
    v5.push_back(Orig_VbX[t].data());
    v5.push_back(Orig_VbY[t].data());
    problem2.AddResidualBlock(data_term_motion, NULL, v5);

    DynamicNumericDiffCostFunction<dynamic_motion>* data_term_motion1 = new 
                                        DynamicNumericDiffCostFunction<dynamic_motion> (new dynamic_motion());
    data_term_motion1->SetNumResiduals(1);
    data_term_motion1->AddParameterBlock(dim);
    data_term_motion1->AddParameterBlock(dim);
    data_term_motion1->AddParameterBlock(dim);
    data_term_motion1->AddParameterBlock(dim);
    std::vector<double *> v6;
    v6.push_back(Orig_VoX[t].data());
    v6.push_back(Orig_VoY[t].data());
    v6.push_back(Orig_VbX[t].data());
    v6.push_back(Orig_VbY[t].data());
    problem2.AddResidualBlock(data_term_motion1, NULL, v6);
    Solver::Summary summary2;
    Solve(options, &problem2, &summary2);
    std::cout << summary2.FullReport() << std::endl;  
  }
  return 0;
}

void delta(Eigen::MatrixXd &A, Eigen::MatrixXd &answer) {
  // Eigen::MatrixXd answer, delx, dely;
  // delx = Eigen::MatrixXd(A.rows()-1,A.cols()-1);
  // dely = Eigen::MatrixXd(A.rows()-1,A.cols()-1);
  for(int i = 0 ; i < A.rows() - 1 ; i++){
    for(int j = 0 ; j < A.cols() - 1 ; j++){
      answer(i,j) = A(i,j) - A(i+1,j);
      answer(i,j) += A(i,j) - A(i,j+1);
      answer(i,j) /= 2;
    }
  }
  // answer = (delx + dely)/2;
  // return answer;
}

double L(Eigen::MatrixXd &input1, Eigen::MatrixXd &input2) {
	double product = 0;
  assert(input1.rows() == input2.rows() && input2.cols() == input1.cols());
	double temp1, temp2;
  for(int i = 0 ; i < input1.rows() - 1 ; i++){
    for(int j = 0 ; j < input1.cols() - 1 ; j++){
      temp1  = input1(i,j) - input1(i+1,j);
      temp1 += input1(i,j) - input1(i,j+1);
      temp2  = input2(i,j) - input2(i+1,j);
      temp2 += input2(i,j) - input2(i,j+1);
      product += temp1*temp1*temp2*temp2*0.25;
    }
  }
	return product;
}

double L1Norm(Eigen::MatrixXd &mat) {
  double answer = 0.0;
  for (int i=0; i<mat.cols(); i++) {
    double tempanswer = 0.0;
    for (int j=0; j<mat.rows(); j++) {
      tempanswer += fabs(mat(j, i));
    }
    answer = std::max(answer, tempanswer);
  }
  assert(answer>=0);
  return answer;
}

double L1NormWithDelta(Eigen::MatrixXd &mat) {
  double answer = 0.0;
  for (int i=0; i<mat.cols() - 1; i++) {
    double tempanswer = 0.0;
    for (int j=0; j<mat.rows() -1 ; j++) {
      tempanswer += fabs(2*mat(j, i) - mat(j+1, i) - mat(j, i+1))/2;
    }
    answer = std::max(answer, tempanswer);
  }
  assert(answer>=0);
  return answer;
}

double AddL1NormWithDelta(Eigen::MatrixXd &mat1, Eigen::MatrixXd &mat2) {
  double answer = 0.0;
  for (int i=0; i<mat1.cols() - 1; i++) {
    double tempanswer = 0.0;
    for (int j=0; j<mat1.rows() -1 ; j++) {
      tempanswer += fabs(2*mat1(j, i) - mat1(j+1, i) - mat1(j, i+1) + 2*mat2(j, i) - mat2(j+1, i) - mat2(j, i+1))/2;
    }
    answer = std::max(answer, tempanswer);
  }
  assert(answer>=0);
  return answer;
}


void warp(Eigen::MatrixXd &mat, Eigen::MatrixXd &mx, Eigen::MatrixXd &my, Eigen::MatrixXd &result) {
  result.setZero();
  int limy = mat.rows();
  int limx = mat.cols();
  assert(limy == mx.rows() and limy== my.rows());
  assert(limx == mx.cols() and limx== my.cols());
  for(int i=0; i<limy; i++){
    for(int j=0; j<limx; j++){
      int posx = j + mx(i,j);
      int posy = i + my(i,j);
      if(posx<limx && posy<limy && posx >=0 && posy >=0)
        result(posy, posx) = mat(i,j);
    }
  }
  // return result;
}
