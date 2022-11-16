#include <chrono>

#include "cam.h"

static double GetTimestamp(void) {
  double timestamp = std::chrono::duration_cast<std::chrono::duration<double>>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count();
  return timestamp;
}

cam::Cam::Cam(int index) {
  cap = cv::VideoCapture(index);
  width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
  height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
  time_prev = GetTimestamp();
  time = GetTimestamp();
};

cam::Cam::~Cam() {
  cap.release();
  std::cout << "Cap. Released!" << std::endl;
}

int cam::Cam::read() {
  int success = cap.read(frame);
  time = GetTimestamp();

  count++;
  int check = 30;
  if (count % check == 0) {
    fps = check / (time - time_prev);
    time_prev = time;
  }

  int key = cv::waitKey(5) & 0xFF;
  if (key == 27)
    success = 0;
  return success;
}

cam::Rec::Rec(std::string path, float fps, int width, int height) {
  int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
  writer = cv::VideoWriter(path, fourcc, fps, cv::Size(width, height));
}

cam::Rec::~Rec() {
  writer.release();
  std::cout << "Rec. Released!" << std::endl;
};

int cam::Rec::write(cv::Mat image) {
  writer.write(image);
  return 1;
}
