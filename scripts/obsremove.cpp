#include "init-helpers.h"
#include <iostream>
#include "optimization.h"
#include <gflags/gflags.h>
#include <opencv2/core/core.hpp>
#include <Eigen/Dense>

DEFINE_string(dirname, "init_data/", "Directory from which images need to picked up.");
DEFINE_int32(num_images, 5, "Number of images which need to be picked up");
DEFINE_string(out_dir, "out_data/", "Directory to dump the results in.");
DEFINE_bool(reflection, true, "Removing a reflection or occlusion");

extern Eigen::MatrixXd Orig_Io, Orig_Ib, Orig_A;
extern std::vector<Eigen::MatrixXd> Orig_VoX, Orig_VoY, Orig_VbX, Orig_VbY;
extern std::vector<Eigen::MatrixXd> Orig_VoX_US, Orig_VoY_US, Orig_VbX_US, Orig_VbY_US;
extern int img_rows;
extern int img_cols;
extern int num_images;
extern total_data input;

int main(int argc, char **argv)
{
  google::SetUsageMessage("obsremove --help");
  google::SetVersionString("1.0.0");
  google::ParseCommandLineFlags(&argc, &argv, true);

  input = total_data(FLAGS_dirname, FLAGS_num_images);
  input.dump_data(FLAGS_out_dir);
  
  initialise(input, FLAGS_out_dir, Orig_Io, Orig_A, Orig_Ib, Orig_VoX, Orig_VoY, Orig_VbX, Orig_VbY);
  img_rows = Orig_Io.rows();
  img_cols = Orig_Io.cols();
  num_images = input.frames.size();
  Orig_VoX_US.resize(num_images);
  Orig_VoY_US.resize(num_images);
  Orig_VbX_US.resize(num_images);
  Orig_VbY_US.resize(num_images);


  save_normalised(Orig_Io, FLAGS_out_dir+"origio.png");
  save_normalised(Orig_Ib, FLAGS_out_dir+"origib.png");
  save_normalised(input.base_img_normalised, FLAGS_out_dir+"origbase.png");
  for (int i=0; i<input.frames.size(); i++) {
    save_normalised(input.normalised_frames[i], FLAGS_out_dir+"origbase_"+std::to_string(i)+".png");
  }

  std::vector<Eigen::MatrixXd> orig_norm_mats;

  for (int i=0; i<num_images; i++) {
    orig_norm_mats.push_back(input.normalised_frames[i]);
  }
  // return 0;
  const int initfact = 16;
  img_rows /= initfact;
  img_cols /= initfact;
  Orig_Io = DownSampleMat(Orig_Io, initfact);
  Orig_Ib = DownSampleMat(Orig_Ib, initfact);
  Orig_A = DownSampleMat(Orig_A, initfact);
  input.base_img_normalised = DownSampleFromCvMat(input.base_img, Orig_Io.cols(), Orig_Io.rows());
  for (int i=0; i<num_images; i++) {
    Orig_VoX[i] = DownSampleMat(Orig_VoX[i], initfact)/initfact;
    Orig_VoY[i] = DownSampleMat(Orig_VoY[i], initfact)/initfact;
    Orig_VbX[i] = DownSampleMat(Orig_VbX[i], initfact)/initfact;
    Orig_VbY[i] = DownSampleMat(Orig_VbY[i], initfact)/initfact;
    Orig_VoX_US[i] = Orig_VoX[i];
    Orig_VoY_US[i] = Orig_VoY[i];
    Orig_VbX_US[i] = Orig_VbX[i];
    Orig_VbY_US[i] = Orig_VbY[i];
    input.normalised_frames[i] = DownSampleFromCvMat(input.frames[i], Orig_VoX[i].cols(), Orig_VoX[i].rows());
    std::cout << "Vox " << Orig_VoX[i].rows() << "\t" << Orig_VoX[i].cols() << "\n";
    std::cout << "Norm frame " << input.normalised_frames[i].rows() << "\t" << input.normalised_frames[i].cols() << "\n";
    assert(input.normalised_frames[i].cols() == Orig_VoX[i].cols());
    assert(input.normalised_frames[i].rows() == Orig_VoX[i].rows());
  }
  assert(input.normalised_frames[0].cols() == Orig_Io.cols());
  assert(input.normalised_frames[0].rows() == Orig_Io.rows());

  // for (int i=0; i<Orig_VoX[0].rows(); i++) {
  //   for (int j=0; j<Orig_VoX[0].cols(); j++) {
  //     std::cerr<< Orig_VbX[0](i, j) << "\t" << Orig_VbY[0](i,j) <<"\n";
  //   }
  // }
  // return 0;

  for (int i=0; i<20; i++)
    ceressolver(1, false);


  for (int it=2; it<=initfact; it*=2) {
    save_normalised(Orig_Io, "Final_O_"+std::to_string(initfact)+"_" + std::to_string(it) + "_included.png");
    save_normalised(Orig_Ib, "Final_B_"+std::to_string(initfact)+"_" + std::to_string(it) + "_included.png");

    img_rows *=2;
    img_cols *=2;
    Orig_Io = UpSampleMat(Orig_Io, 2);
    Orig_Ib = UpSampleMat(Orig_Ib, 2);
    Orig_A = UpSampleMat(Orig_A, 2);
    Fix(Orig_Io);
    Fix(Orig_Ib);
    Fix(Orig_A);
    input.base_img_normalised = DownSampleFromCvMat(input.base_img, Orig_Io.cols(), Orig_Io.rows());
    for (int i=0; i<num_images; i++) {
      Orig_VoX_US[i] = UpSampleMat(Orig_VoX[i], it)*it;
      Orig_VoY_US[i] = UpSampleMat(Orig_VoY[i], it)*it;
      Orig_VbX_US[i] = UpSampleMat(Orig_VbX[i], it)*it;
      Orig_VbY_US[i] = UpSampleMat(Orig_VbY[i], it)*it;
      input.normalised_frames[i] = DownSampleFromCvMat(input.frames[i], Orig_Io.cols(), Orig_Io.rows());
      std::cout << "Vox " << Orig_VoX[i].rows() << "\t" << Orig_VoX[i].cols() << "\n";
      std::cout << "Norm frame " << input.normalised_frames[i].rows() << "\t" << input.normalised_frames[i].cols() << "\n";
      assert(input.normalised_frames[i].cols() == Orig_Io.cols());
      assert(input.normalised_frames[i].rows() == Orig_Io.rows());
    }
    ceressolver(it, true);
  }

  save_normalised(Orig_Io, "Final_O_8.png");
  save_normalised(Orig_Ib, "Final_B_8.png");

  return 0;
}
