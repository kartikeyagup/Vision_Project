#include "init-helpers.h"
#include <iostream>
#include <gflags/gflags.h>
#include <opencv2/core/core.hpp>

DEFINE_string(dirname, "init_data/", "Directory from which images need to picked up.");
DEFINE_int32(num_images, 5, "Number of images which need to be picked up");
DEFINE_string(out_dir, "out_data/", "Directory to dump the results in.");
DEFINE_bool(reflection, true, "Removing a reflection or occlusion");

int main(int argc, char **argv)
{
  google::SetUsageMessage("obsremove --help");
  google::SetVersionString("1.0.0");
  google::ParseCommandLineFlags(&argc, &argv, true);

  total_data input(FLAGS_dirname, FLAGS_num_images);
  input.dump_data(FLAGS_out_dir);
  
  initialise(input, FLAGS_out_dir);

  return 0;
}
