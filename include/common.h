#ifndef COMMON_H
#define COMMON_H

#include <vector>
#include <utility>

struct motion_field {
  std::vector<std::vector<std::pair<int, int> > > data;

  int getx(int i, int j) {
    // i row, j column
    return data[i][j].second;
  }

  int gety(int i, int j) {
    // i row, j column
    return data[i][j].first;
  }

};

#endif
