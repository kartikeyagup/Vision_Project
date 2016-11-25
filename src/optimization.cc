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

Eigen::MatrixXd Orig_Io, Orig_Ib, Orig_A; 
std::vector<Eigen::MatrixXd> Orig_VoX, Orig_VoY, Orig_VbX, Orig_VbY;

int img_rows;
int numiter;
int img_cols;
int num_images;

total_data input;

DEFINE_string(minimizer, "line_search",
              "Minimizer type to use, choices are: line_search & trust_region");
 
struct dynamic_data_term {
  bool operator()(double const* const* parameters,
                                        double* residual) const {
    double param1 = 1;
    double param2 = 100;
    double param3 = 1000;

    Eigen::MatrixXd io = Eigen::Map<Eigen::MatrixXd>((double*) parameters[0], img_rows, img_cols);
    Eigen::MatrixXd ib = Eigen::Map<Eigen::MatrixXd>((double*) parameters[1], img_rows, img_cols);
    Eigen::MatrixXd a  = Eigen::Map<Eigen::MatrixXd>((double*) parameters[2], img_rows, img_cols);
    double abc=0;
    // std::cout << "Adding shit\n";
    // // residual[0] = T(0.0);
    for(int t = 0; t < num_images; t++) {
      Eigen::MatrixXd iovo = warp(io, Orig_VoX[t], Orig_VoY[t]);
      Eigen::MatrixXd ibvb = warp(ib, Orig_VbX[t], Orig_VbY[t]);
      Eigen::MatrixXd iavo = warp(a , Orig_VoX[t], Orig_VoY[t]);
      assert(iovo.rows() == ibvb.rows());
      assert(iovo.cols() == ibvb.cols());
      assert(iavo.rows() == ibvb.rows());
      assert(iavo.cols() == ibvb.cols());
      // assert(io.rows() == )
      Eigen::MatrixXd iavoibvb = iavo.cwiseProduct(ibvb);

      // std::cout << "Abc changed from " << abc;
      Eigen::MatrixXd s = input.normalised_frames[t] - iovo - iavoibvb;
      // abc += ((input.normalised_frames[t] - iovo - iavoibvb).lpNorm<1>());
      abc += L1Norm(s);

    }
    residual[0] = abc;
    // residual[0] +=  (param1 * ((double) delta(a).norm()));
    // residual[0] += (param2 * ((double) (L1Norm(delta(io)) + L1Norm(delta(ib)))));
    // residual[0] += (param3 * (double) (L(io,ib)));
    std::cout << "\rNum iter "<< numiter <<" Residual is " << residual[0] << " ,,,,,,,," << std::flush;
    numiter++;
    // std::cout << io.norm() <<std::endl;
    return true;
  }
};

// struct dynamic_grad_a {
//   bool operator()(double const* const* parameters,
//                                         double* residual) const {
//     Eigen::MatrixXd a  = Eigen::Map<Eigen::MatrixXd>((double*) parameters[0], img_rows, img_cols);
//     double param = 1;
//     residual[0] =  (param * ((double) delta(a).norm()));
//     return true;
//   }
// };

struct dynamic_norm_l1_ioib {
  bool operator()(double const* const* parameters,
                                        double* residual) const {
    double param = 100;
    Eigen::MatrixXd io = Eigen::Map<Eigen::MatrixXd>((double*) parameters[0], img_rows, img_cols);
    Eigen::MatrixXd ib = Eigen::Map<Eigen::MatrixXd>((double*) parameters[1], img_rows, img_cols);
    residual[0] = (param * ((double) (L1Norm(delta(io)) + L1Norm(delta(ib)))));
    return true;
  }
};

struct dynamic_l {
  bool operator()(double const* const* parameters,
                                        double* residual) const {
    double param = 100;
    Eigen::MatrixXd io = Eigen::Map<Eigen::MatrixXd>((double*) parameters[0], img_rows, img_cols);
    Eigen::MatrixXd ib = Eigen::Map<Eigen::MatrixXd>((double*) parameters[1], img_rows, img_cols);
    residual[0] = (param * (double) (L(io,ib)));
    return true;
  }
};

struct dynamic_data_term_motion {

private:
  int img_id;
public:
  bool operator()(double const* const* parameters,
                                        double* residual) const {
    // residual[0] = T(0.0);
    double param1 = 10;

    Eigen::MatrixXd vox = Eigen::Map<Eigen::MatrixXd>((double*) parameters[0], img_rows, img_cols);
    Eigen::MatrixXd voy = Eigen::Map<Eigen::MatrixXd>((double*) parameters[1], img_rows, img_cols);
    Eigen::MatrixXd vbx = Eigen::Map<Eigen::MatrixXd>((double*) parameters[2], img_rows, img_cols);
    Eigen::MatrixXd vby = Eigen::Map<Eigen::MatrixXd>((double*) parameters[3], img_rows, img_cols);
    // int img_id = 0;
    Eigen::MatrixXd iovo = warp(Orig_Io, vox, voy);
    Eigen::MatrixXd ibvb = warp(Orig_Ib, vbx, vby);
    Eigen::MatrixXd iavo = warp(Orig_A, vox, voy);
    Eigen::MatrixXd iavoibvb = iavo.cwiseProduct(ibvb);
    residual[0] = (L1Norm(input.normalised_frames[img_id] - iovo - iavoibvb));
    residual[0] = residual[0] + param1 * ((L1Norm(delta(vox) + delta(voy))/2) + (L1Norm(delta(vbx) + delta(vby))/2));
    std::cout << "\rResidual is " << residual[0] << " ,,,,,,,,,,,,,,,,,, " << std::flush;
    return true;
  }

  dynamic_data_term_motion(int im){
    img_id = im;
  }
};

struct dynamic_motion {
  bool operator()(double const* const* parameters,
                                        double* residual) const {
    // residual[0] = T(0.0);
    double param1 = 10;

    Eigen::MatrixXd vox = Eigen::Map<Eigen::MatrixXd>((double*) parameters[0], img_rows, img_cols);
    Eigen::MatrixXd voy = Eigen::Map<Eigen::MatrixXd>((double*) parameters[1], img_rows, img_cols);
    Eigen::MatrixXd vbx = Eigen::Map<Eigen::MatrixXd>((double*) parameters[2], img_rows, img_cols);
    Eigen::MatrixXd vby = Eigen::Map<Eigen::MatrixXd>((double*) parameters[3], img_rows, img_cols);
    residual[0] = param1 * ((L1Norm(delta(vox) + delta(voy))/2) + (L1Norm(delta(vbx) + delta(vby))/2));
    return true;
  }
};

int ceressolver() {
  numiter = 0;
  std::cout<<"Starting on the first problem\n";
  Problem problem;
  
  const int dim = img_cols * img_rows;
  Eigen::Matrix2d matrix;
  double* m = Orig_Io.data();

  DynamicNumericDiffCostFunction<dynamic_data_term>* c1 = new 
                                        DynamicNumericDiffCostFunction<dynamic_data_term> (new dynamic_data_term());
                      
  c1->SetNumResiduals(1);
  std::vector<double*> v;
  c1->AddParameterBlock(dim);
  c1->AddParameterBlock(dim);
  c1->AddParameterBlock(dim);
  v.push_back(Orig_Io.data());
  v.push_back(Orig_Ib.data());
  v.push_back(Orig_A.data());
  problem.AddResidualBlock(c1, NULL, v);
  problem.SetParameterBlockConstant(Orig_A.data());
  // Solver::Options options;

  // DynamicNumericDiffCostFunction<dynamic_grad_a>* c2 = new 
  //                                       DynamicNumericDiffCostFunction<dynamic_grad_a> (new dynamic_grad_a());
                      
  // c2->SetNumResiduals(1);
  // std::vector<double*> v2;
  // c2->AddParameterBlock(dim);
  // v2.push_back(Orig_A.data());
  // problem.AddResidualBlock(c2, NULL, v2);

  DynamicNumericDiffCostFunction<dynamic_norm_l1_ioib>* c3 = new 
                                        DynamicNumericDiffCostFunction<dynamic_norm_l1_ioib> (new dynamic_norm_l1_ioib());
                      
  c3->SetNumResiduals(1);
  std::vector<double*> v3;
  c3->AddParameterBlock(dim);
  c3->AddParameterBlock(dim);
  // c3->AddParameterBlock(dim);
  v3.push_back(Orig_Io.data());
  v3.push_back(Orig_Ib.data());
  // v.push_back(Orig_A.data());
  problem.AddResidualBlock(c3, NULL, v3);

  DynamicNumericDiffCostFunction<dynamic_l>* c4 = new 
                                        DynamicNumericDiffCostFunction<dynamic_l> (new dynamic_l());
                      
  c4->SetNumResiduals(1);
  std::vector<double*> v4;
  c4->AddParameterBlock(dim);
  c4->AddParameterBlock(dim);
  // c3->AddParameterBlock(dim);
  v4.push_back(Orig_Io.data());
  v4.push_back(Orig_Ib.data());
  // v.push_back(Orig_A.data());
  problem.AddResidualBlock(c4, NULL, v4);

  Solver::Options options;

  LOG_IF(FATAL, !ceres::StringToMinimizerType(FLAGS_minimizer,
                                              &options.minimizer_type))
      << "Invalid minimizer: " << FLAGS_minimizer
      << ", valid options are: trust_region and line_search.";
  options.max_num_iterations = 10;
  options.linear_solver_type = ceres::CGNR;
  options.preconditioner_type = ceres::IDENTITY;
  options.minimizer_progress_to_stdout = true;
  options.num_threads = 40;
  // options.use_inner_iterations=true;
  options.num_linear_solver_threads = 40;

  Solver::Summary summary;
  Solve(options, &problem, &summary);
  std::cout << "HAHAHA" <<std::endl;
  std::cout << summary.FullReport() << "\n";

  Problem problem2;
  numiter = 0;
  std::cout<<"Starting on the second problem\n";
  for(int t=0;t<num_images;t++){
    
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
  }
  Solver::Options options2;
  LOG_IF(FATAL, !ceres::StringToMinimizerType(FLAGS_minimizer,
                                              &options.minimizer_type))
      << "Invalid minimizer: " << FLAGS_minimizer
      << ", valid options are: trust_region and line_search.";

  options2.max_num_iterations = 10;
  options2.linear_solver_type = ceres::CGNR;
  // options2.line_search_direction_type = ceres::BFGS;
  options2.preconditioner_type=ceres::IDENTITY;
  // options2.use_inner_iterations = true;
  // options2.use_nonmonotonic_steps = true;
  options2.minimizer_progress_to_stdout = true;
  options2.num_threads = 40;
  options2.num_linear_solver_threads = 40;

  Solver::Summary summary2;
  Solve(options2, &problem2, &summary2);
  
  std::cout << summary2.FullReport() << std::endl;
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

double L(Eigen::MatrixXd &input1, Eigen::MatrixXd &input2) {
	double product = 0;
  assert(input1.rows() == input2.rows() && input2.cols() == input1.cols());
	Eigen::MatrixXd del1 = delta(input1);
	Eigen::MatrixXd del2 = delta(input2);
	for(int i = 0 ; i < del1.rows() ; i++){
		for(int j = 0 ; j < del1.cols() ; j++){
			product += del1(i,j) * del1(i,j) * del2(i,j) * del2(i,j);
		}
	}
	return product;
}

double L1Norm(Eigen::MatrixXd mat) {
  double answer = 0.0;
  for (int i=0; i<mat.cols(); i++) {
    double tempanswer = 0.0;
    for (int j=0; j<mat.rows(); j++) {
      tempanswer += fabs(mat(j, i));
    }
    answer = std::max(answer, tempanswer);
  }
  return answer;
}

Eigen::MatrixXd warp(Eigen::MatrixXd &mat, Eigen::MatrixXd &mx, Eigen::MatrixXd &my) {
  Eigen::MatrixXd result(mat.rows(), mat.cols());
  result.setZero();
  int limy = mat.rows();
  int limx = mat.cols();
  assert(limy == mx.rows() and limy== my.rows());
  assert(limx == mx.cols() and limx== my.cols());
  // assert(limx = motion.data[0].size());
  // assert(limy = motion.data.size());
  for(int i=0; i<limy; i++){
    for(int j=0; j<limx; j++){
      int posx = j + mx(i,j);
      int posy = i + my(i,j);
      // std::cout << i<< ", "<< j <<" went to " << posy <<", " << posx<<"\n";
      if(posx<limx && posy<limy && posx >=0 && posy >=0)
        result(posy, posx) = mat(i,j);
    }
  }
  // std::cout << limy << "\t" << limx <<"\n";
  return result;
}