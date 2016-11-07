#include "common.h"

cv::Mat convertimg(Eigen::MatrixXd img) {
  // Eigen::Map<MatrixXd> eigenT( cvT.data() ); 
  img = img * 255;
  cv::Mat answer(img.rows(), img.cols(), CV_8UC1);
  for(int i = 0 ; i < img.rows() ; i++){
    for(int j = 0 ; j < img.cols() ; j++){
      answer.at<uchar>(i,j) = (int) img(i,j);
    }
  }
  return answer;
}
