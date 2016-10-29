#include "init-helpers.h"

bool inBounds(cv::Point2f &p, cv::Mat &img) {
  return ((p.x < img.cols-1) && (p.y < img.rows-1));
}

bool Track(std::vector<cv::Point2f> &edge, 
    cv::Mat &img1, cv::Mat &img2,
    int &dx, int &dy) {
  int gridsize = 5;
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
        // std::cout << delta << " " << tot <<" ";

        if (tot < best) {
          best= tot;
          dx = i;
          dy = j;
        }
      }
    }
  }
  if (best < 10) {
    // std::cout << best ;
    answer = true;
  }
  return answer;
}

bool comfun(const std::pair<cv::Point2i, int> &p1,const std::pair<cv::Point2i, int> &p2) {
  return (p1.second > p2.second);
}

void initialise(total_data &input, std::string out_dir) {
  cv::Mat baseEdges;
  cv::Canny(input.base_img, baseEdges, 50, 100);
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
  int tot(0), mt(0);
  // cv::namedWindow("initial0");
  // cv::namedWindow("initial1");
  std::unordered_map<cv::Point2i, int> counts_of_tr;
  for (auto it: all_edges) {
    tot++;
    int dx, dy;
    // if (it.second.size() < 50) {
    //   continue;
    // }
    if (Track(it.second, input.base_img, input.frames[0],  dx, dy)) {
      mt++;
      // std::cout << dx << " , "<< dy << "\n";
      counts_of_tr[cv::Point2i(dx, dy)]++;
      // std::cout << it.first << "\t" << it.second.size() <<  "\n";
      // // std::cout << dx << "\t" << dy << "\n";
      // cv::Rect bb = cv::boundingRect(it.second);
      // cv::Point2f plc(bb.x, bb.y);
      // cv::Point2f prb(bb.x + bb.width, bb.y + bb.height);
      // std::cout << plc << "\n";
      // std::cout << prb << "\n";
      // cv::Mat img0, img1;
      // input.base_img.copyTo(img0);
      // input.frames[0].copyTo(img1);
      // cv::rectangle(img0, plc, prb, cv::Scalar(0,0,0), 2, 4);
      // cv::rectangle(img1, plc + cv::Point2f(dx, dy), prb + cv::Point2f(dx, dy), cv::Scalar(0,0,0), 2, 4);
      // for (auto it1 : it.second) {
      //   cv::circle(img0, it1, 1, cv::Scalar(0,0,255), -1);
      // }
      // cv::imshow("initial0", img0);
      // cv::imshow("initial1", img1);
      // while (cv::waitKey(10) != 27) {

      // }
    }
  }
  std::vector<std::pair<cv::Point2i, int> > counts_pairs;
  for (auto it : counts_of_tr) {
    counts_pairs.push_back(std::make_pair(it.first, it.second));
    std::cout << it.first << "\t" << it.second << "\n";
  }
  std::sort(counts_pairs.begin(), counts_pairs.end(), comfun);
  assert (cv::Point2i(1,1) == cv::Point2i(1,1));
  std::cout << "Matched " << mt << " out of " << tot << "\n";
  for (auto it: counts_pairs) {
    std::cout << it.first << "\t" << it.second << "\n";
  }
}
