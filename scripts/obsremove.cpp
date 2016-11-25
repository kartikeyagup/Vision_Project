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

  const int initfact = 8;
  img_rows /= initfact;
  img_cols /= initfact;
  Orig_Io = DownSampleMat(Orig_Io, initfact);
  Orig_Ib = DownSampleMat(Orig_Ib, initfact);
  Orig_A = DownSampleMat(Orig_A, initfact);
  for (int i=0; i<num_images; i++) {
    Orig_VoX[i] = DownSampleMat(Orig_VoX[i], initfact)/initfact;
    Orig_VoY[i] = DownSampleMat(Orig_VoY[i], initfact)/initfact;
    Orig_VbX[i] = DownSampleMat(Orig_VbX[i], initfact)/initfact;
    Orig_VbY[i] = DownSampleMat(Orig_VbY[i], initfact)/initfact;
    input.normalised_frames[i] = DownSampleFromCvMat(input.frames[i], Orig_VoX[i].cols(), Orig_VoX[i].rows());
    std::cout << "Vox " << Orig_VoX[i].rows() << "\t" << Orig_VoX[i].cols() << "\n";
    std::cout << "Norm frame " << input.normalised_frames[i].rows() << "\t" << input.normalised_frames[i].cols() << "\n";
    assert(input.normalised_frames[i].cols() == Orig_VoX[i].cols());
    assert(input.normalised_frames[i].rows() == Orig_VoX[i].rows());
  }
  assert(input.normalised_frames[0].cols() == Orig_Io.cols());
  assert(input.normalised_frames[0].rows() == Orig_Io.rows());
  for (int i=0; i<4; i++)
    ceressolver();

  for (int it=2; it<=initfact; it*=2) {
    save_normalised(Orig_Io, "Final_O_" + std::to_string(it) + ".png");
    save_normalised(Orig_Ib, "Final_B_" + std::to_string(it) + ".png");

    img_rows *=2;
    img_cols *=2;
    Orig_Io = UpSampleMat(Orig_Io, 2);
    Orig_Ib = UpSampleMat(Orig_Ib, 2);
    Orig_A = UpSampleMat(Orig_A, 2);
    for (int i=0; i<num_images; i++) {
      Orig_VoX[i] = UpSampleMat(Orig_VoX[i], 2)*2;
      Orig_VoY[i] = UpSampleMat(Orig_VoY[i], 2)*2;
      Orig_VbX[i] = UpSampleMat(Orig_VbX[i], 2)*2;
      Orig_VbY[i] = UpSampleMat(Orig_VbY[i], 2)*2;
      input.normalised_frames[i] = DownSampleFromCvMat(input.frames[i], Orig_VoX[i].cols(), Orig_VoX[i].rows());
      std::cout << "Vox " << Orig_VoX[i].rows() << "\t" << Orig_VoX[i].cols() << "\n";
      std::cout << "Norm frame " << input.normalised_frames[i].rows() << "\t" << input.normalised_frames[i].cols() << "\n";
      assert(input.normalised_frames[i].cols() == Orig_VoX[i].cols());
      assert(input.normalised_frames[i].rows() == Orig_VoX[i].rows());
    }
    ceressolver();
  }

  save_normalised(Orig_Io, "Final_O.png");
  save_normalised(Orig_Ib, "Final_B.png");

  return 0;
}
