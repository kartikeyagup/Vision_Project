#include "init-helpers.h"

bool inBounds(cv::Point2f &p, cv::Mat &img) {
  return ((p.x < img.cols-1) && (p.y < img.rows-1) && (p.x > 0) && (p.y > 0));
}

double Track(std::vector<cv::Point2f> &edge, 
    cv::Mat &img1, cv::Mat &img2,
    int &dx, int &dy, cv::Mat &grad1, cv::Mat &grad2) {
  int gridsize = 150;
  double answer = 10000000;
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
          cv::Vec3b c1_grad = grad1.at<cv::Vec3b>(edge[k]);
          cv::Vec3b c2_grad = grad2.at<cv::Vec3b>(pt);
          tot += fabs(((int) c2[0])-((int) c1[0]));
          tot += fabs(((int) c2[1])-((int) c1[1]));
          tot += fabs(((int) c2[2])-((int) c1[2]));
          tot += 0.01*fabs(((int) c2_grad[0])-((int) c1_grad[0]));
          tot += 0.01*fabs(((int) c2_grad[1])-((int) c1_grad[1]));
          tot += 0.01*fabs(((int) c2_grad[2])-((int) c1_grad[2]));
        } else {
          sofar = false;
          break;
        }
      }
      if (sofar) {
        tot /= 3.0;
        tot /= edge.size();
        tot /= 255;
        tot = fabs(tot);

        if (tot < best) {
          best= tot;
          dx = i;
          dy = j;
        }
      }
    }
  }
  if (best < 0.04) {
    std::cout << "Edge with " << edge.size() << " tracked with error" << best << "\n";
    answer = best;
  }
  return answer;
}

bool comfun(const std::pair<cv::Point2i, int> &p1,const std::pair<cv::Point2i, int> &p2) {
  return (p1.second > p2.second);
}

void initialise(total_data &input, std::string out_dir) {
  cv::Mat baseEdges;
  cv::Mat oneEdges;
  cv::Canny(input.base_img, baseEdges, 25, 50);
  cv::Canny(input.frames[0], oneEdges, 25, 50);
  //cv::Mat grayinput(input.base_img.rows, input.base_img.cols, CV_8UC1);
  //cv::cvtColor(input.base_img, grayinput, CV_BGR2GRAY);
  //cv::Laplacian(input.base_img, baseEdges, -1, 5, 1, 0, cv::BORDER_DEFAULT);
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
  //cv::imwrite(out_dir+"pitjaoge.png", labels);
  cv::imwrite(out_dir+"edge_1.png", oneEdges);

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

  cv::Mat trace_baseimg_firstimg(baseEdges.size(), CV_32S);
  input.base_img.copyTo(trace_baseimg_firstimg);

  std::vector<cv::Point2f> initial_pts, final_pts, initial_pts_fg, final_pts_fg;
  std::vector<std::pair<cv::Point2f, std::vector<cv::Point2f> > > Edges, Edges_2, Edges_bg, Edges_fg;

  for (auto it: all_edges) {
    tot++;
    int dx, dy;
    if (it.second.size() < 100) {
      continue;
    } 
 //   double trackerror = Track(it.second, input.base_img, input.frames[0],  dx, dy, baseEdges, oneEdges);
    /////////////////////////////////////////////////////
    std::vector<cv::Point2f> cornersA; 
    std::vector<cv::Point2f> cornersB; 
    std::vector<cv::Point2f> cornersGF; 

    goodFeaturesToTrack( baseEdges,cornersGF,50000,0.01,10.0,cv::Mat(),3,0,0.04);

    for(auto it3 = cornersGF.begin(); it3 != cornersGF.end(); it3++)
    {
      if(find(it.second.begin(), it.second.end(), *it3) != it.second.end())
        cornersA.push_back(*it3);
    }

    std::cout << "Pit jaoge" << cornersA.size() << std::endl;

    std::vector<uchar> features_found; 
    std::vector<float> feature_errors; 

    if(cornersA.size() > 0)
      calcOpticalFlowPyrLK( baseEdges, oneEdges, cornersA, cornersB, features_found, feature_errors ,
         cv::Size( 31, 31 ), 5,
          cvTermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.3 ), 0 );



    //////////////////////////////////////////////////////
    //std::cout << trackerror << "\n";

    for(int i = 0; i < features_found.size(); i++)
    {
      if(features_found[i] == true)
      {
        mt++;
        Edges.push_back(std::make_pair(cv::Point2f(dx, dy), it.second));
        int i=0;
        for (auto it1: it.second) {
          initial_pts.push_back(it1);
          final_pts.push_back(it1 + cv::Point2f(dx, dy));
          i++;
          if(i%20 == 0)
          {
              arrowedLine(trace_baseimg_firstimg, it1, it1+cv::Point2f(dx, dy),cv::Vec3b(0,0255-2550*feature_errors[i],2550*feature_errors[i]));
            circle(trace_baseimg_firstimg, it1, 5, cv::Vec3b(0,14,10));  
          }
          
        }
      }
    }

    // if ( trackerror != 10000000 ) {
    //   mt++;
    //   Edges.push_back(std::make_pair(cv::Point2f(dx, dy), it.second));
    //   cv::Rect bb = cv::boundingRect(it.second);
    //   int i=0;
    //   for (auto it1: it.second) {
    //     initial_pts.push_back(it1);
    //     final_pts.push_back(it1 + cv::Point2f(dx, dy));
    //     i++;
    //     if(i%20 == 0)
    //     {
    //       arrowedLine(trace_baseimg_firstimg, it1, it1+cv::Point2f(dx, dy),cv::Vec3b(0,0255-2550*trackerror,2550*trackerror));
    //       circle(trace_baseimg_firstimg, it1, 5, cv::Vec3b(0,14,10));  
    //     }
        
    //   }
    // }
  }
  std::cout << "Matched " << mt << " out of " << tot << "\n";

  cv::imwrite(out_dir+"track.png", trace_baseimg_firstimg);

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

  cv::imwrite(out_dir + "edges_bg.png", bg);
  cv::imwrite(out_dir + "edges_fg.png", fg);

  std::cout << countfg << " inliers in fg\n";
}
