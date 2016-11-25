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

Eigen::MatrixXd DownSampleMat(Eigen::MatrixXd mt, int factor) {
	Eigen::MatrixXd temp = Eigen::MatrixXd(mt.rows()/factor, mt.cols()/factor);
	for (int i=0; i<temp.rows(); i++) {
		for (int j=0; j<temp.cols(); j++) {
			temp(i,j) = mt(i*factor, j*factor);
		}
	}
	assert(temp.rows() == mt.rows()/factor);
	assert(temp.cols() == mt.cols()/factor);
	return temp;
}

Eigen::MatrixXd UpSampleMat(Eigen::MatrixXd mt, int factor) {	
	Eigen::MatrixXd temp = Eigen::MatrixXd(mt.rows()*factor, mt.cols()*factor);
	for (int i=0; i<temp.rows(); i++) {
		for (int j=0; j<temp.cols(); j++) {
			temp(i,j) = mt(i/factor, j/factor);
		}
	}
	assert(temp.rows() == mt.rows()*factor);
	assert(temp.cols() == mt.cols()*factor);	
	return temp;
}

Eigen::MatrixXd DownSampleFromCvMat(cv::Mat img, int cls, int rws) {
	cv::Mat resized;
	cv::resize(img, resized, cv::Size(cls, rws));
	return normalize(resized);
}

Eigen::MatrixXd normalize(cv::Mat inp) {
  cv::Mat m;
  cv::cvtColor(inp, m, CV_BGR2GRAY);
  Eigen::MatrixXd answer(m.rows,m.cols);
  for(int i=0;i<m.rows;i++){
    for(int j=0;j<m.cols;j++){
      answer(i,j) = ((int) (inp.at<cv::Vec3b>(i, j)[0])) + 
                    ((int) (inp.at<cv::Vec3b>(i, j)[1])) +
                    ((int) (inp.at<cv::Vec3b>(i, j)[2]));
      answer(i,j) /= (3*255.0);
      // answer(i,j) = ((int) m.at<uchar>(i,j))/255.0;
    }
  }
  return answer;
}
