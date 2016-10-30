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
  int tot(0), mt(0);
  // cv::namedWindow("initial0");
  // cv::namedWindow("initial1");
  std::unordered_map<cv::Point2i, int> counts_of_tr;
  std::vector<cv::Point2f> initial_pts, final_pts, initial_pts_fg, final_pts_fg;
  std::vector<std::pair<cv::Point2f, std::vector<cv::Point2f> > > Edges, Edges_2, Edges_bg, Edges_fg;
  for (auto it: all_edges) {
    tot++;
    int dx, dy;
    if (it.second.size() < 1) {
      continue;
    } 
    if (Track(it.second, input.base_img, input.frames[0],  dx, dy)) {
      mt++;
      Edges.push_back(std::make_pair(cv::Point2f(dx, dy), it.second));
      // std::cout << dx << " , "<< dy << "\n";
      counts_of_tr[cv::Point2i(dx, dy)]++;
      // std::cout << it.first << "\t" << it.second.size() <<  "\n";
      // // std::cout << dx << "\t" << dy << "\n";
      cv::Rect bb = cv::boundingRect(it.second);
      initial_pts.push_back(cv::Point2f(bb.x + bb.width/2, bb.y + bb.y/2));
      final_pts.push_back(cv::Point2f(dx + bb.x + bb.width/2, dy + bb.y + bb.y/2));
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
  std::cout << "Matched " << mt << " out of " << tot << "\n";

  // std::vector<std::pair<cv::Point2i, int> > counts_pairs;
  // for (auto it : counts_of_tr) {
  //   counts_pairs.push_back(std::make_pair(it.first, it.second));
  //   // std::cout << it.first << "\t" << it.second << "\n";
  // }
  // std::sort(counts_pairs.begin(), counts_pairs.end(), comfun);
  // // for (auto it: counts_pairs) {
  // //   std::cout << it.first << "\t" << it.second << "\n";
  // // }

  cv::Mat tr1, tr2;
  std::vector<uchar> mask1, mask2;
  assert(initial_pts.size() == final_pts.size());
  assert(Edges.size() == initial_pts.size());
  tr1 = cv::findHomography(initial_pts, final_pts, CV_RANSAC, 3, mask1);
  std::cout << tr1.rows << "\t" << tr1.cols << "\t" << mask1.size() << "\n" ;
  int countbg(0), countfg(0);
  for (int i=0; i<mask1.size(); i++) {
    if (!(mask1[i])) {
      Edges_2.push_back(Edges[i]);
      initial_pts_fg.push_back(initial_pts[i]);
      final_pts_fg.push_back(final_pts[i]);
    } else {
      Edges_bg.push_back(Edges[i]);
      countbg++;
    }
  }
  std::cout << countbg << " inliers in bg\n";
  assert(initial_pts_fg.size() == final_pts_fg.size());
  tr2 = cv::findHomography(initial_pts_fg, final_pts_fg, CV_RANSAC, 3, mask2);
  std::cout << tr2.rows << "\t" << tr2.cols << "\t" << mask2.size() << "\n" ;
  assert(mask2.size() == initial_pts_fg.size());
  assert(Edges_2.size() == mask2.size());
  for (int i=0; i<mask2.size(); i++) {
    if (mask2[i]) {
      Edges_fg.push_back(Edges_2[i]);
      countfg++;
    }
  }
  cv::Mat bg(baseEdges.size(), CV_8UC3);
  cv::Mat fg(baseEdges.size(), CV_8UC3);
  for (auto it: Edges_bg) {
    for (auto it1: it.second) {
      cv::circle(bg, it1, 2, cv::Scalar(0,0,255), -1);
    }
  }

  for (auto it: Edges_fg) {
    for (auto it1: it.second) {
      cv::circle(fg, it1, 2, cv::Scalar(0,0,255), -1);
    }
  }

  cv::imwrite(out_dir + "edges_bg.png", bg);
  cv::imwrite(out_dir + "edges_fg.png", fg);

  std::cout << countfg << " inliers in fg\n";
}
