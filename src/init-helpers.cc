#include "init-helpers.h"

bool inBounds(cv::Point2f &p, cv::Mat &img) {
  return ((p.x < img.cols-1) && (p.y < img.rows-1) && (p.x > 0) && (p.y > 0));
}

bool Track(std::vector<cv::Point2f> &edge, 
  cv::Mat &img1, cv::Mat &img2,
  int &dx, int &dy) {
  int gridsize = 9;
  bool answer = false;
  float best = 10000;
  for (int i=-gridsize; i<gridsize; i++) {
    for (int j=-gridsize; j<gridsize; j++) {
      cv::Point2f delta(i,j);
      bool sofar = true;
      float tot(0);
      for (int k=0; k<edge.size(); k++) {
        cv::Point2f pt = edge[k] + delta;
        if (inBounds(pt, img2)) {
          cv::Vec3b c1 = img1.at<cv::Vec3b>(edge[k]);
          cv::Vec3b c2 = img2.at<cv::Vec3b>(pt);
          tot += ((int) c2[0])-((int) c1[0]);
          tot += ((int) c2[1])-((int) c1[1]);
          tot += ((int) c2[2])-((int) c1[2]);
        } else {
          sofar = false;
          break;
        }
      }
      if (sofar) {
        tot /= 3.0;
        tot /= edge.size();
        tot = fabs(tot);

        if (tot < best) {
          best= tot;
          dx = i;
          dy = j;
        }
      }
    }
  }
  if (best < 1) {
    answer = true;
  }
  return answer;
}

/**
 * @brief Sorting comparator function
 */
bool comfun(const std::pair<cv::Point2i, int> &p1,const std::pair<cv::Point2i, int> &p2) {
  return (p1.second > p2.second);
}

void initialise(total_data &input, std::string out_dir, 
  Eigen::MatrixXd &Io, Eigen::MatrixXd &A, Eigen::MatrixXd &Ib,
  std::vector<Eigen::MatrixXd> &VoX, std::vector<Eigen::MatrixXd> &VoY,
  std::vector<Eigen::MatrixXd> &VbX, std::vector<Eigen::MatrixXd> &VbY) {
  // Eigen::MatrixXd zeros = ;
  VoX.clear();
  VoY.clear();
  VbX.clear();
  VbY.clear();
  VoX.resize(input.frames.size());
  VoY.resize(input.frames.size());
  VbX.resize(input.frames.size());
  VbY.resize(input.frames.size());
  for(int i=0 ; i<input.frames.size();i++){
    VoX[i] = Eigen::MatrixXd(input.base_img.rows, input.base_img.cols);
    VoY[i] = Eigen::MatrixXd(input.base_img.rows, input.base_img.cols);
    VbX[i] = Eigen::MatrixXd(input.base_img.rows, input.base_img.cols);
    VbY[i] = Eigen::MatrixXd(input.base_img.rows, input.base_img.cols);
  }
  A = Eigen::MatrixXd(input.base_img.rows, input.base_img.cols);
  Io = Eigen::MatrixXd(input.base_img.rows, input.base_img.cols);
  Ib = Eigen::MatrixXd(input.base_img.rows, input.base_img.cols);
  for(int i = 0; i<input.base_img.rows;i++){
    for(int j=0;j<input.base_img.cols;j++){
      A(i,j) = 0.1;
      // double x = 
      Io(i,j) = ((double) rand() / (RAND_MAX));
      Ib(i,j) = ((double) rand() / (RAND_MAX));
    }
  }

  cv::Mat baseEdges;
  cv::Canny(input.base_img, baseEdges, 25, 50);
  cv::imwrite(out_dir+"edges_base.png", baseEdges);

  int posx, posy;
  bool found = false;
  for (int i=0; i<baseEdges.rows && !found; i++) {
    for (int j=0; j<baseEdges.cols && !found; j++) {
      if (!baseEdges.at<uchar>(i, j)) {
        posx = i;
        posy = j;
        found = true;
      }
    }
  }
  cv::Mat labels(baseEdges.size(), CV_32S);
  int nlabels = cv::connectedComponents(baseEdges, labels, 8);
  std::vector<cv::Vec3b> colors(nlabels);
  for (int i=0; i<nlabels; i++) {
    colors[i] = cv::Vec3b( (rand()&255), (rand()&255), (rand()&255));
  }
  int label_bg = labels.at<int>(posx, posy);
  std::cout << "Background label was " << label_bg << "\n";
  colors[label_bg] = cv::Vec3b(0,0,0);
  cv::Mat segments(baseEdges.size(), CV_8UC3);
  for (int i=0; i<segments.rows; i++) {
    for (int j=0; j<segments.cols; j++) {
      segments.at<cv::Vec3b>(i, j) = colors[labels.at<int>(i,j)];
    }
  }
  cv::imwrite(out_dir+"labels.png", segments);
  std::cout << "Labels " << nlabels << "\n";

  std::unordered_map<int, std::vector<cv::Point2f> > all_edges;
  for (int i=0; i<labels.cols; i++) {
    for (int j=0; j<labels.rows; j++) {
      if (labels.at<int>(j,i) != label_bg)
        all_edges[labels.at<int>(j,i)].push_back(cv::Point2f(i,j));
    }
  }

  for (int fr=0; fr<input.frames.size(); fr++) {
    int tot(0), mt(0);
    std::vector<cv::Point2f> initial_pts, final_pts, initial_pts_fg, final_pts_fg;
    std::vector<std::pair<cv::Point2f, std::vector<cv::Point2f> > > Edges, Edges_2, Edges_bg, Edges_fg;
    for (auto it: all_edges) {
      tot++;
      int dx, dy;
      if (it.second.size() < 10) {
        continue;
      } 
      if (Track(it.second, input.base_img, input.frames[fr],  dx, dy)) {
        mt++;
        Edges.push_back(std::make_pair(cv::Point2f(dx, dy), it.second));
        cv::Rect bb = cv::boundingRect(it.second);
        for (auto it1: it.second) {
          initial_pts.push_back(it1);
          final_pts.push_back(it1 + cv::Point2f(dx, dy));
        }
      }
    }
    std::cout << "Matched " << mt << " out of " << tot << "\n";

    cv::Mat tr1, tr2;
    std::vector<uchar> mask1, mask2;
    std::vector<cv::Point2f> points_bg, points_fg;
    assert(initial_pts.size() == final_pts.size());
    tr1 = cv::findHomography(final_pts, initial_pts, CV_RANSAC, 3, mask1);
    std::cout << tr1.rows << "\t" << tr1.cols << "\t" << mask1.size() << "\n" ;
    int countbg(0), countfg(0);
    for (int i=0; i<mask1.size(); i++) {
      if (!(mask1[i])) {
        initial_pts_fg.push_back(initial_pts[i]);
        final_pts_fg.push_back(final_pts[i]);
      } else {
        points_bg.push_back(initial_pts[i]);
        countbg++;
      }
    }
    std::cout << countbg << " inliers in bg\n";
    assert(initial_pts_fg.size() == final_pts_fg.size());
    tr2 = cv::findHomography(final_pts_fg, initial_pts_fg, CV_RANSAC, 3, mask2);
    std::cout << tr2.rows << "\t" << tr2.cols << "\t" << mask2.size() << "\n" ;
    assert(mask2.size() == initial_pts_fg.size());
    for (int i=0; i<mask2.size(); i++) {
      if (mask2[i]) {
        points_fg.push_back(initial_pts_fg[i]);
        countfg++;
      }
    }
    cv::Mat bg(baseEdges.size(), CV_8UC3);
    cv::Mat fg(baseEdges.size(), CV_8UC3);
    for (auto it: points_bg) {
        cv::circle(bg, it, 2, cv::Scalar(0,0,255), -1);
    }

    for (auto it: points_fg) {
        cv::circle(fg, it, 2, cv::Scalar(0,0,255), -1);
    }

    std::cout << "Foming 1\n";
    form_motion_field(input.base_img.rows, input.base_img.cols, tr1, VbX[fr], VbY[fr]);
    std::cout << "Foming 2\n";
    form_motion_field(input.base_img.rows, input.base_img.cols, tr2, VoX[fr], VoY[fr]);
    
    cv::imwrite(out_dir + "edges_bg.png", bg);
    cv::imwrite(out_dir + "edges_fg.png", fg);
    std::cout << countfg << " inliers in fg\n";
  }
  // TODO: Fill Io, A, Ib
  input.base_img_normalised = normalize(input.base_img);
  for (auto it: input.frames) {
    input.normalised_frames.push_back(normalize(it));
  }
}

void form_motion_field(int rows, int cols, cv::Mat homo, Eigen::MatrixXd &mx, Eigen::MatrixXd &my){
  // Eigen::Map<Matrix3d> homography(homo.data());
  Eigen::Matrix3d homography(3,3);
  for(int i=0; i<3; i++){
    for(int j=0;j<3;j++){
      homography(i,j) = (double)homo.at<uchar>(i,j);
    }
  }
  
  for(int i = 0; i<rows;i++){
    for(int j=0; j<cols;j++){
      Eigen::MatrixXd point(3,1);
      point << j,i,1;
      Eigen::MatrixXd ans = homography * point;
      double val = ans(2,0);
      ans(0,0) = ans(0,0)/val;
      ans(1,0) = ans(1,0)/val;
      // if(ans(0,0) < cols && ans(1,0) < rows){
      mx(i,j) = ans(0,0) - j;
      my(i,j) = ans(1,0) - i;
      // }
    }
  }
}
  
Eigen::MatrixXd normalize(cv::Mat m) {
  Eigen::MatrixXd answer(m.rows,m.cols);
  for(int i=0;i<m.rows;i++){
    for(int j=0;j<m.cols;j++){
      answer(i,j) = m.at<uchar>(i,j)/255;
    }
  }
  return answer;
}