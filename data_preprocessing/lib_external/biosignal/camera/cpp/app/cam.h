#ifndef CAM_H
#define CAM_H

#include <opencv2/opencv.hpp>

namespace cam {
class Cam {
  cv::VideoCapture cap;
  int width;
  int height;
  cv::Mat frame;

  int count;
  double fps;
  double time;
  double time_prev;

public:
  Cam(int indexmake);
  ~Cam();
  int read();

  cv::Mat get_frame() { return frame; };
  double get_time() { return time; };
  int get_width() { return width; };
  int get_height() { return height; };
  double get_fps() { return fps; };
};

class Rec {
  cv::VideoWriter writer;

public:
  Rec(std::string path, float fps, int width, int height);
  ~Rec();
  int write(cv::Mat image);
};
} // namespace cam

#endif