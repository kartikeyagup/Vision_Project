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
  num_images = input.frames.size() - 1;

  save_normalised(Orig_Io, FLAGS_out_dir+"origio.png");
  save_normalised(Orig_Ib, FLAGS_out_dir+"origib.png");
  save_normalised(input.base_img_normalised, FLAGS_out_dir+"origbase.png");
  for (int i=0; i<input.frames.size(); i++) {
    save_normalised(input.normalised_frames[i], FLAGS_out_dir+"origbase_"+std::to_string(i)+".png");
  }
  //ceressolver();

  return 0;
}
