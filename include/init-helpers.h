#ifndef INIT_HELPERS_H
#define INIT_HELPERS_H

#include <vector>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <unordered_map>
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

namespace std {
  template <>
  struct hash<cv::Point2i> {
    std::size_t operator()(const cv::Point2i& k) const {
      return ((hash<int>()(k.x)) ^ (hash<int>()(k.y)));
    }
  };
}

/**
 * @brief Checkes if a point lies in the bounds of an image
 */
bool inBounds(cv::Point2f &p, cv::Mat &img);

/**
 * @brief Tracks an edge from 1 image to another
 * 
 * @param edge vector of points correspodnign to edge
 * @param img1 base image
 * @param img2 2nd image
 * @param dx displacement along x which is obtained
 * @param dy displacement along y which is obtained
 * @return true or false if the edge was matched in 2nd image
 *          if true, then dx and dy are the displacements
 */
bool Track(std::vector<cv::Point2f> &edge, 
  cv::Mat &img1, cv::Mat &img2,
  int &dx, int &dy);

/**
 * @brief Function meant to initialise the data
 * @param input all images
 * @param out_dir debug directory to dump images in
 */
void initialise(total_data &input, std::string out_dir);

#endif
