#include <algorithm>
#include <cstdio>
#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

#include "cam.h"
#include "signalgo.h"

class FlowModule {
  cv::Mat prev;
  cv::Mat next;
  int count;

public:
  FlowModule();

  cv::Mat run(cv::Mat image);
};

FlowModule::FlowModule() { count = 0; }

cv::Mat FlowModule::run(cv::Mat image) {
  prev = next;
  next = image;

  cv::Mat flow;
  count++;
  if (count >= 2) {
    flow = cv::Mat(image.size(), CV_32FC2);
    cv::calcOpticalFlowFarneback(prev, next, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
  } else {
    flow = cv::Mat::zeros(image.size(), CV_32FC2);
  }
  return flow;
}

cv::Mat GetFlowView(cv::Mat flow) {
  cv::Mat flow_parts[2];
  cv::split(flow, flow_parts);
  cv::Mat magnitude, angle, magn_norm;
  cv::cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
  cv::normalize(magnitude, magn_norm, 0.0f, 1.0f, cv::NORM_MINMAX);
  angle *= ((1.f / 360.f) * (180.f / 255.f));

  cv::Mat _hsv[3], hsv, hsv8, bgr;
  _hsv[0] = angle;
  _hsv[1] = cv::Mat::ones(angle.size(), CV_32F);
  _hsv[2] = magn_norm;
  cv::merge(_hsv, 3, hsv);
  hsv.convertTo(hsv8, CV_8U, 255.0);
  cv::cvtColor(hsv8, bgr, cv::COLOR_HSV2BGR);

  return bgr;
}

// TODO: circular queue?
class Queue {
  unsigned int size_;

public:
  std::vector<double> data;
  Queue();
  Queue(int size);
  void set(int size);
  void push_back(double x);
  int size() { return size_; };
};

Queue::Queue() { size_ = 0; }

void Queue::set(int size) {
  size_ = size;
  for (int i = 0; i < size; i++) {
    data.push_back(0.f);
  }
}

void Queue::push_back(double x) {
  data.push_back(x);
  if (data.size() > size_) {
    data.erase(data.begin());
  }
}

double clip(double value, double min, double max) {
  if (value < min) {
    return min;
  } else if (value > max) {
    return max;
  }
  return value;
}

class SignalModule {
  int size_;
  double integral;

public:
  Queue times;
  Queue pq;
  Queue iq;
  std::vector<int> inspiration;
  double frequency;
  SignalModule(int size);
  void run(double timestamp, double value);
};

SignalModule::SignalModule(int size) {
  size_ = size;
  times.set(size);
  pq.set(size);
  iq.set(size);
  integral = 0.f;
  frequency = 0.f;
}

void SignalModule::run(double timestamp, double value) {
  value = clip(value, -0.5, +0.5);

  times.push_back(timestamp);
  pq.push_back(value);
  integral += value;
  iq.push_back(integral);

  inspiration = signalgo::find_peaks(iq.data, 0.05, -1, 5);

  int count = inspiration.size();
  if (count >= 2) {
    int p0 = inspiration[0];
    int p1 = inspiration[inspiration.size() - 1];
    double t0 = times.data[p0];
    double t1 = times.data[p1];
    double dt = t1 - t0;
    frequency = (count - 1) / dt;
  }
}

cv::Mat get_graph(SignalModule sm) {
  const int scale = 320;
  const int size = sm.pq.size();
  cv::Mat graph = cv::Mat::zeros(cv::Size(scale, size), CV_8UC3);

  for (int i = 0; i < (int)sm.inspiration.size(); i++) {
    int r = sm.inspiration[i];
    for (int x = 0; x < scale; x++) {
      graph.data[(r * scale + x) * 3] = 255; // red channel;
    }
  }

  const double min =
      *std::min_element(std::begin(sm.iq.data), std::end(sm.iq.data));
  const double max =
      *std::max_element(std::begin(sm.iq.data), std::end(sm.iq.data));
  const double max_min = max - min + 1e-6;

  int s_prev = (int)((sm.iq.data[0] - min) / max_min * scale);
  for (int i = 0; i < size; i++) {
    double v = sm.iq.data[i];
    v = (v - min) / max_min;
    int s = (int)(v * scale);

    int p0 = std::min(s, s_prev);
    int p1 = std::max(s, s_prev);
    for (int p = p0; p < p1; p++) {
      graph.data[(i * scale + p) * 3 + 1] = 255; // green channel;
    }
    s_prev = s;
  }

  return graph;
}

int test() {
  cv::namedWindow("view");
  cv::moveWindow("view", 0, 0);
  cv::namedWindow("input");
  cv::moveWindow("input", 0, 400);

  cam::Cam camera(0);
  int width = camera.get_width() / 2;
  int height = camera.get_height() / 2;

  auto f_module = FlowModule();
  auto s_module = SignalModule(360);

  cv::Mat image, image_p, gray;

  while (camera.read()) {
    auto frame = camera.get_frame();
    cv::resize(frame, image, cv::Size(width, height));

    image_p = image(cv::Rect(0, height / 2, width, height / 2));
    cv::cvtColor(image_p, gray, cv::COLOR_BGR2GRAY);
    cv::imshow("input", gray);

    cv::Mat flow = f_module.run(gray.clone());
    cv::Mat flow_view = GetFlowView(flow);

    auto mean = cv::mean(flow);

    s_module.run(camera.get_time(), mean[1]);

    cv::Mat graph = get_graph(s_module);

    cv::Mat view, view2;
    cv::vconcat(image_p, flow_view, view);
    cv::hconcat(view, graph, view2);

    cv::Point pos(640, 40);
    cv::Scalar yellow(0, 255, 255);
    char buf[64] = {0};
    sprintf(buf, " %4.1f fps", camera.get_fps());
    cv::putText(view2, buf, pos, 0, 0.5, yellow, 1);
    pos.y += 20;
    sprintf(buf, " %4.1f brpm", s_module.frequency * 60);
    cv::putText(view2, buf, pos, 0, 0.5, yellow, 1);

    cv::imshow("view", view2);
  }

  return 0;
}

int main() {
  test();
  return 0;
}