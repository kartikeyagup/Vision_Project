#ifndef INIT_HELPERS_H
#define INIT_HELPERS_H

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>

struct total_data {
  cv::Mat base_img;
  std::vector<cv::Mat> frames;

  total_data() {};
  total_data(std::string inp_dir, int num_images) {
    assert(num_images>=2);
    base_img = cv::imread(inp_dir+"/img_0.png");

    for (int i=1; i<num_images; i++) {
      cv::Mat temp = cv::imread(inp_dir+"/img_"+std::to_string(i)+".png");
      frames.push_back(temp);
    }
    
    assert(frames.size() + 1 == num_images);
  }

  /**
   * @brief Stores the images on the disk
   * 
   * @param out_dir path of the directory in which the images have to be stored
   */
  void dump_data(std::string out_dir) {
    cv::imwrite(out_dir+"out_img0.png", base_img);
    for (int i=0; i<frames.size(); i++) {
      cv::imwrite(out_dir+"out_img"+std::to_string(i+1)+".png", frames[i]);
    }
  }
};

#endif
