#ifndef INIT_HELPERS_H
#define INIT_HELPERS_H

#include <vector>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <unordered_map>
#include <algorithm>
#include <string>
#include "common.h"
#include <Eigen/Dense>

struct total_data {
  cv::Mat base_img, base_img_gr;
  Eigen::MatrixXd base_img_normalised;
  std::vector<cv::Mat> frames;
  std::vector<cv::Mat> frames_gr;
  std::vector<Eigen::MatrixXd> normalised_frames;

  total_data() {};
  total_data(std::string inp_dir, int num_images) {
    assert(num_images>=2);
    base_img = cv::imread(inp_dir+"/img_0.png");
    cv::cvtColor(base_img, base_img_gr, CV_BGR2GRAY);
  
    for (int i=1; i<num_images; i++) {
      cv::Mat temp = cv::imread(inp_dir+"/img_"+std::to_string(i)+".png");
      cv::Mat temp_gr;
      cv::cvtColor(temp, temp_gr, CV_BGR2GRAY);
      frames.push_back(temp);
      frames_gr.push_back(temp_gr);
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
 * @param Io Eigen mat for obstruction part
 * @param A obstruction alpha mat
 * @param Ib Eigen mat for background
 * @param VoX Eigen mat represtions motion along x for obs
 * @param VoY Eigen mat represtions motion along y for obs
 * @param VbX Eigen mat represtions motion along x for bg
 * @param VbY Eigen mat represtions motion along y for bg
 */
void initialise(total_data &input, std::string out_dir, 
  Eigen::MatrixXd &Io, Eigen::MatrixXd &A, Eigen::MatrixXd &Ib,
  std::vector<Eigen::MatrixXd> &VoX, std::vector<Eigen::MatrixXd> &VoY,
  std::vector<Eigen::MatrixXd> &VbX, std::vector<Eigen::MatrixXd> &VbY);

void form_motion_field(int rows, int cols, cv::Mat homo, Eigen::MatrixXd &mx, Eigen::MatrixXd &my);

void save_normalised(Eigen::MatrixXd &img, std::string path);

#endif
