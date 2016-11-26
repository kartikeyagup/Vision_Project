#include "init-helpers.h"

bool inBounds(cv::Point2f &p, cv::Mat &img) {
  return ((p.x < img.cols-1) && (p.y < img.rows-1) && (p.x > 0) && (p.y > 0));
}

bool Track(std::vector<cv::Point2f> &edge, 
  cv::Mat &img1, cv::Mat &img2,
  int &dx, int &dy) {
  int gridsize = 10;
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

void FillVXY(cv::Mat &tr1, Eigen::MatrixXd &vx, Eigen::MatrixXd &vy) {
  // Fills vx and vy based on homography m.
  // std::cout << tr1.at<double>(0,0) << ", " << tr1.at<double>(0,1) << ", " << tr1.at<double>(0,2) << "\n";
  // std::cout << tr1.at<double>(1,0) << ", " << tr1.at<double>(1,1) << ", " << tr1.at<double>(1,2) << "\n";
  // std::cout << tr1.at<double>(2,0) << ", " << tr1.at<double>(2,1) << ", " << tr1.at<double>(2,2) << "\n";
  for (int i=0; i<vx.rows(); i++) {
    for (int j=0; j<vx.cols(); j++) {
      double nx = tr1.at<double>(0, 0)*j + tr1.at<double>(0, 1)*i + tr1.at<double>(0, 2);
      double ny = tr1.at<double>(1, 0)*j + tr1.at<double>(1, 1)*i + tr1.at<double>(1, 2);
      double nz = tr1.at<double>(2, 0)*j + tr1.at<double>(2, 1)*i + tr1.at<double>(2, 2);
      double cx = nx/nz;
      double cy = ny/nz;
      vx(i, j) = cx - j;
      vy(i, j) = cy - i;
    }
  }
}

void GenerateWarped(cv::Mat &input_img, cv::Mat &target, Eigen::MatrixXd &VoX, Eigen::MatrixXd &VoY) {
  // Fill the target image based on the Vox and Voy
  for (int i=0; i<input_img.rows;  i++) {
    for (int j=0; j<input_img.cols; j++) {
      cv::Point2f newp(j - VoX(i, j), i - VoY(i, j));
      if (inBounds(newp, input_img)) {
        target.at<cv::Vec3b>(newp) = input_img.at<cv::Vec3b>(i, j);
      }
    }
  }
}

void GetGoodPoints(std::vector<cv::Point2f> &prevtracking,
  std::vector<cv::Point2f> &inversetracking, 
  std::vector<uchar> &status,
  std::vector<uchar> &statusinv) {
  int tot, removed;
  tot = prevtracking.size();
  removed=0;
  for (int i=0; i<prevtracking.size(); i++)  {
    if (status[i]==0) {
      continue;
    }
    if (statusinv[i] == 0) {
      status[i] = 0;
      continue;
    }
    status[i]=0;
    cv::Point2f temmpPoint = inversetracking[i]-prevtracking[i];
    float magnitude = (temmpPoint.x)*(temmpPoint.x) + (temmpPoint.y)*(temmpPoint.y);
    if (magnitude<=4) {
      status[i]=1;
    } else {
      removed++;
    }
  }
  // std::cout << "Removed " << removed << " points from " << tot << " points\n";
  assert (removed < tot);
}

void initialise(total_data &input, std::string out_dir, 
  Eigen::MatrixXd &Io, Eigen::MatrixXd &A, Eigen::MatrixXd &Ib,
  std::vector<Eigen::MatrixXd> &VoX, std::vector<Eigen::MatrixXd> &VoY,
  std::vector<Eigen::MatrixXd> &VbX, std::vector<Eigen::MatrixXd> &VbY) {
  VoX.clear();
  VoY.clear();
  VbX.clear();
  VbY.clear();
  VoX.resize(input.frames.size());
  VoY.resize(input.frames.size());
  VbX.resize(input.frames.size());
  VbY.resize(input.frames.size());
  std::vector<cv::Mat> warped;
  warped.resize(input.frames.size());
  for(int i=0 ; i<input.frames.size();i++){
    VoX[i] = Eigen::MatrixXd(input.base_img.rows, input.base_img.cols);
    VoY[i] = Eigen::MatrixXd(input.base_img.rows, input.base_img.cols);
    VbX[i] = Eigen::MatrixXd(input.base_img.rows, input.base_img.cols);
    VbY[i] = Eigen::MatrixXd(input.base_img.rows, input.base_img.cols);
    cv::Mat temp = cv::Mat::zeros(input.base_img.rows, input.base_img.cols, CV_8UC3);
    temp.copyTo(warped[i]);
  }
  A = Eigen::MatrixXd(input.base_img.rows, input.base_img.cols);
  Io = Eigen::MatrixXd(input.base_img.rows, input.base_img.cols);
  Ib = Eigen::MatrixXd(input.base_img.rows, input.base_img.cols);

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
  // Alloted labels to edges in base img

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
    // for (auto it: all_edges) {
    //   tot++;
    //   int dx, dy;
    //   if (it.second.size() < 10) {
    //     continue;
    //   } 
    //   if (Track(it.second, input.base_img, input.frames[fr],  dx, dy)) {
    //     mt++;
    //     Edges.push_back(std::make_pair(cv::Point2f(dx, dy), it.second));
    //     cv::Rect bb = cv::boundingRect(it.second);
    //     for (auto it1: it.second) {
    //       initial_pts.push_back(it1);
    //       final_pts.push_back(it1 + cv::Point2f(dx, dy));
    //     }
    //   }
    // }
    std::vector<cv::Point2f> corners_prev, corners_inverse, corners;
    int maxCorners = 100000;
    double qualityLevel = 0.001;
    double minDistance = 2;
    int blockSize = 7;
    bool useHarrisDetector = false;
    double k = 0.04;
    cv::Mat mask;
    cv::goodFeaturesToTrack(input.base_img_gr,
        corners_prev, 
        maxCorners, 
        qualityLevel, 
        minDistance,
        mask,
        blockSize,
        useHarrisDetector,
        k);

    std::vector<uchar> status, status_inverse;
    std::vector<float> err;
    cv::Size winSize(15, 15);
    cv::TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10, 0.03);
    calcOpticalFlowPyrLK(input.base_img_gr, input.frames_gr[fr],
        corners_prev, corners, status,
        err, winSize, 2, termcrit, 0, 0.001);

    calcOpticalFlowPyrLK(input.frames_gr[fr], input.base_img_gr,
        corners, corners_inverse, status_inverse,
        err, winSize, 2, termcrit, 0, 0.001);
      
    GetGoodPoints(corners_prev,corners_inverse,status,status_inverse);

    for (int i=0; i<corners.size(); i++) {
      tot++;
      if (status[i]) {
        mt++;
        initial_pts.push_back(corners_prev[i]);
        final_pts.push_back(corners[i]);
      }
    }

    std::cout << "Matched " << mt << " out of " << tot << "\n";

    cv::Mat tr1, tr2;
    std::vector<uchar> mask1, mask2;
    std::vector<cv::Point2f> points_bg, points_fg;
    assert(initial_pts.size() == final_pts.size());
    tr1 = cv::findHomography(initial_pts, final_pts, CV_RANSAC, 3, mask1);
    FillVXY(tr1, VbX[fr], VbY[fr]);
    GenerateWarped(input.frames[fr], warped[fr], VbX[fr], VbY[fr]);

    cv::imwrite(out_dir + std::to_string(fr) + "_warped.png", warped[fr]);
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
    tr2 = cv::findHomography(initial_pts_fg, final_pts_fg, CV_RANSAC, 3, mask2);
    FillVXY(tr2, VoX[fr], VoY[fr]);
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

    cv::imwrite(out_dir + std::to_string(fr) + "_edges_bg.png", bg);
    cv::imwrite(out_dir + std::to_string(fr) + "_edges_fg.png", fg);
    cv::waitKey(30);
    std::cout << countfg << " inliers in fg\n";
  }

  input.base_img_normalised = normalize(input.base_img);
  for (auto it: input.frames) {
    input.normalised_frames.push_back(normalize(it));
  }

  for(int i = 0; i<input.base_img.rows;i++){
    for(int j=0;j<input.base_img.cols;j++){
      A(i, j) = 1.0;
      // Traverse warped images
      float minsofar = ((int) (input.base_img.at<cv::Vec3b>(i, j)[0])) + 
                       ((int) (input.base_img.at<cv::Vec3b>(i, j)[1])) +
                       ((int) (input.base_img.at<cv::Vec3b>(i, j)[2]));
      float bg = minsofar/(3*255.0);
      for (int k=0; k<input.frames.size(); k++) {
        int presentcol = ((int) (warped[k].at<cv::Vec3b>(i, j)[0])) + 
                         ((int) (warped[k].at<cv::Vec3b>(i, j)[1])) +
                         ((int) (warped[k].at<cv::Vec3b>(i, j)[2]));
        if (presentcol>0 && (presentcol<minsofar)) {
          minsofar = presentcol;
        }
      }
      Ib(i, j) = minsofar/(3*255.0);
      // Set Ib as min
      Io(i, j) = bg - Ib(i, j);
      assert(Io(i,j) >= 0);
      // Set Io as subtraction
    }
  }

}

void save_normalised(Eigen::MatrixXd &img, std::string path) {
  cv::Mat conv_img(img.rows(), img.cols(), CV_8UC1);
  for (int i=0; i<img.rows(); i++) {
    for (int j=0; j<img.cols(); j++) {
      assert(img(i,j) <= 1.0000001);
      assert(img(i,j) >= -2.0000001);
      conv_img.at<uchar>(i, j) = floor(img(i,j)*255);
    }
  }
  cv::imwrite(path, conv_img);
}
