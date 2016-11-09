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

DEFINE_string(minimizer, "line_search",
              "Minimizer type to use, choices are: line_search & trust_region");
 
struct dynamic_data_term {
  template <typename T> bool operator()(T const* const* parameters,
                                        T* residual) const {
    // residual[0] = T(0.0); // Might have to remove this

    double param1 = 1;
    double param2 = 100;
    double param3 = 1000;

    //  The first term is added here
    Eigen::MatrixXd io = Eigen::Map<Eigen::MatrixXd>((double*) parameters[0], img_rows, img_cols);
    Eigen::MatrixXd ib = Eigen::Map<Eigen::MatrixXd>((double*) parameters[1], img_rows, img_cols);
    Eigen::MatrixXd a  = Eigen::Map<Eigen::MatrixXd>((double*) parameters[2], img_rows, img_cols);
    // residual[0] = T(0.0);
    for(int t = 0; t < num_images+1; t++) {
      if(t == num_images){
        continue;
      }
      Eigen::MatrixXd iovo = warp(io, Orig_VoX[t], Orig_VoY[t]);
      Eigen::MatrixXd ibvb = warp(ib, Orig_VbX[t], Orig_VbY[t]);
      Eigen::MatrixXd iavo = warp(a , Orig_VoX[t], Orig_VoY[t]);

      std::cout<<"Image number: "<<t+1<<std::endl;
      std::cout<<iavo.rows()<<"\t"<<iavo.cols()<<"\n";
      std::cout<<ibvb.rows()<<"\t"<<ibvb.cols()<<"\n";
      
      assert(iovo.rows() == ibvb.rows());
      assert(iovo.cols() == ibvb.cols());
      assert(iavo.rows() == ibvb.rows());
      assert(iavo.cols() == ibvb.cols());
      // assert(io.rows() == )
      Eigen::MatrixXd iavoibvb = iavo.cwiseProduct(ibvb);

      residual[0] += T((input.normalised_frames[t] - iovo - iavoibvb).lpNorm<1>());
    }

    residual[1] =  T(param1 * ((double) delta(a).norm()));
    residual[2] = T(param2 * ((double) (delta(io).lpNorm<1>() + delta(ib).lpNorm<1>())));
    residual[3] = T(param3 * (double) (L(io,ib)));
    std::cout << "Calculating 0\t" << residual[0] << "\n"<<"Calculating 1\t"<<residual[1]<<"\n";
    std::cout << "Calculating 2\t" << residual[2] << "\n"<<"Calculating 3\t"<<residual[3]<<"\n";
    return true;
  }
};

struct dynamic_grad_a{
  template<typename T> bool operator()(T const* const* parameters,
                                       T* residual) const {
    
    double param1 = 100;
    Eigen::MatrixXd a  = Eigen::Map<Eigen::MatrixXd>((double*) parameters[0], img_rows, img_cols);
    residual[0] = T(param1 * ((double) delta(a).norm()));
    return true;
  }
};

struct dynamic_grad_io_ib {
  template<typename T> bool operator()(T const* const* parameters,
                                       T* residual) const {
    
    double param2 = 100;
    Eigen::MatrixXd io = Eigen::Map<Eigen::MatrixXd>((double*) parameters[0], img_rows, img_cols);
    Eigen::MatrixXd ib = Eigen::Map<Eigen::MatrixXd>((double*) parameters[1], img_rows, img_cols);
    residual[0] = T(param2 * ((double) (delta(io).lpNorm<1>() + delta(ib).lpNorm<1>())));
    return true;
  }
};

struct dynamic_L {
  template<typename T> bool operator()(T const* const* parameters,
                                       T* residual) const { 
    double param3 = 100;
    Eigen::MatrixXd io = Eigen::Map<Eigen::MatrixXd>((double*) parameters[0], img_rows, img_cols);
    Eigen::MatrixXd ib = Eigen::Map<Eigen::MatrixXd>((double*) parameters[1], img_rows, img_cols);
    residual[0] = param3 * T(L(io,ib));
    return true;
  }
};

struct dynamic_data_term_motion {

private:
  int img_id;
public:
  dynamic_data_term_motion(int t){
    img_id = t;
  }

  template <typename T> bool operator()(T const* const* parameters,
                                        T* residual) const {
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
    residual[0] = T((input.normalised_frames[img_id] - iovo - iavoibvb).lpNorm<1>());

    residual[0] = residual[0] + param1 * (((delta(vox) + delta(voy))/2).lpNorm<1>() + ((delta(vbx) + delta(vby))/2).lpNorm<1>());

    return true;
  }
};


int ceressolver() {
  std::cout<<"Starting on the first problem\n";
  Problem problem;
  
  const int dim = img_cols * img_rows;

  double *x1 = Orig_Io.data();
  double *x2 = Orig_Ib.data();
  double *x3 = Orig_A.data();
  // x1 = Orig_Io.data();

  DynamicAutoDiffCostFunction<dynamic_data_term> data_term(new dynamic_data_term());
  data_term.SetNumResiduals(5);
  data_term.AddParameterBlock(dim);
  data_term.AddParameterBlock(dim);
  data_term.AddParameterBlock(dim);
  std::vector<double *> v;
  v.push_back(Orig_Io.data());
  v.push_back(Orig_Ib.data());
  v.push_back(Orig_A.data());
  problem.AddResidualBlock(&data_term, NULL, v);


  // DynamicAutoDiffCostFunction<dynamic_data_term> data_term(new dynamic_data_term());
  // data_term.SetNumResiduals(1);
  // data_term.AddParameterBlock(dim);
  // data_term.AddParameterBlock(dim);
  // data_term.AddParameterBlock(dim);
  // std::vector<double *> v;
  // v.push_back(x1);
  // v.push_back(x2);
  // v.push_back(x3);
  // problem.AddResidualBlock(&data_term, NULL, v);


  // DynamicAutoDiffCostFunction<dynamic_grad_a> data_term_grad_a(new dynamic_grad_a());
  // data_term_grad_a.SetNumResiduals(1);
  // data_term_grad_a.AddParameterBlock(dim);
  // std::vector<double *> v1;
  // v1.push_back(x3);
  // problem.AddResidualBlock(&data_term_grad_a, NULL, v1);


  // DynamicAutoDiffCostFunction<dynamic_grad_io_ib> data_term_grad_io_ib(new dynamic_grad_io_ib());
  // data_term_grad_io_ib.SetNumResiduals(1);
  // data_term_grad_io_ib.AddParameterBlock(dim);
  // data_term_grad_io_ib.AddParameterBlock(dim);
  // std::vector<double *> v2;
  // v2.push_back(x1);
  // v2.push_back(x2);
  // problem.AddResidualBlock(&data_term_grad_io_ib, NULL, v2);


  // DynamicAutoDiffCostFunction<dynamic_L> data_term_L(new dynamic_L());
  // data_term_L.SetNumResiduals(1);
  // data_term_L.AddParameterBlock(dim);
  // data_term_L.AddParameterBlock(dim);
  // std::vector<double *> v3;
  // v3.push_back(x1);
  // v3.push_back(x2);
  // problem.AddResidualBlock(&data_term_L, NULL, v3);
  // std::cout<<"Here\n";

  Solver::Options options;
  LOG_IF(FATAL, !ceres::StringToMinimizerType(FLAGS_minimizer,
                                              &options.minimizer_type))
      << "Invalid minimizer: " << FLAGS_minimizer
      << ", valid options are: trust_region and line_search.";
  options.max_num_iterations = 10;
  options.linear_solver_type = ceres::ITERATIVE_SCHUR;
  options.minimizer_progress_to_stdout = true;
  options.num_threads = 8;
  // std::cout << "Initial x1 = " << x1
  //           << ", x2 = " << x2
  //           << ", x3 = " << x3
  //           << ", x4 = " << x4
  //           << "\n";
  // // Run the solver!
  Solver::Summary summary;
  Solve(options, &problem, &summary);

  Problem problem2;
  std::cout<<"Starting on the second problem\n";
  for(int t=0;t<num_images;t++){
    // TODO: Set t
    DynamicAutoDiffCostFunction<dynamic_data_term_motion> data_term_motion(new dynamic_data_term_motion(t));
    data_term_motion.SetNumResiduals(1);
    data_term_motion.AddParameterBlock(dim);
    data_term_motion.AddParameterBlock(dim);
    data_term_motion.AddParameterBlock(dim);
    data_term_motion.AddParameterBlock(dim);
    std::vector<double *> v4;
    v4.push_back(Orig_VoX[t].data());
    v4.push_back(Orig_VoY[t].data());
    v4.push_back(Orig_VbX[t].data());
    v4.push_back(Orig_VbY[t].data());
    problem.AddResidualBlock(&data_term_motion, NULL, v4);
  }
  Solver::Options options2;
  LOG_IF(FATAL, !ceres::StringToMinimizerType(FLAGS_minimizer,
                                              &options.minimizer_type))
      << "Invalid minimizer: " << FLAGS_minimizer
      << ", valid options are: trust_region and line_search.";
  options2.max_num_iterations = 100;
  // options2.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
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

// Eigen::MatrixXd element