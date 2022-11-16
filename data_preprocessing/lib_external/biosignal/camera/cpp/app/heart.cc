#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

#include "cam.h"
#include "signalgo.h"

class Queue {
  unsigned int size_;

public:
  std::vector<double> data;
  Queue();
  Queue(int size);
  void set(int size);
  void push_back(double x);
  int size() { return size_; };
  double mean();
  double var();
  double std();
};

Queue::Queue() { size_ = 0; }

void Queue::set(int size) {
  size_ = size;
  for (int i = 0; i < size; i++) {
    data.push_back(0.0);
  }
}

void Queue::push_back(double x) {
  data.push_back(x);
  if (data.size() > size_) {
    data.erase(data.begin());
  }
}

double Queue::mean() {
  if (data.size() == 0)
    return 0.0;
  double sum = 0;
  for (int i = 0; i < data.size(); i++) {
    sum += data[i];
  }
  return sum / data.size();
}

double Queue::var() {
  if (data.size() == 0)
    return 0.0;
  double m = mean();
  double sum = 0;
  for (int i = 0; i < data.size(); i++) {
    double d = data[i] - m;
    sum += (d * d);
  }
  return sum / data.size();
}

double Queue::std() { return sqrt(var()); }

double clip(double value, double min, double max) {
  if (value < min) {
    return min;
  } else if (value > max) {
    return max;
  }
  return value;
}

double *bandpassfilter(double fl, double fh, double fs, int N) {
  double *filter = NULL;
  double flc = 0;
  double omegal = 0;
  double fhc = 0;
  double omegah = 0;
  int i = 0;
  int middle = 0;

  if (fl <= 0 || fh <= 0 || fl >= fh || fs <= 0 || N <= 0)
    return NULL;

  filter = new double[N];
  if (!filter)
    return NULL;

  memset(filter, 0, N * sizeof(double));

  middle = (int)(N / 2);
  flc = fl / fs;
  omegal = 2 * M_PI * flc;
  fhc = fh / fs;
  omegah = 2 * M_PI * fhc;

  for (i = 0; i < N; i++) {
    if (i == middle) {
      filter[i] = 2 * fhc - 2 * flc;
    } else {
      filter[i] = sin(omegah * (i - middle)) / (M_PI * (i - middle)) -
                  sin(omegal * (i - middle)) / (M_PI * (i - middle));
    }
  }
  return filter;
}

class SignalModule {
  int size_;

public:
  Queue times;
  Queue qx;
  Queue qy;
  double *sig;
  double *sig_filt;
  int filter_size;
  double *filter;
  std::vector<int> peaks;
  double frequency;
  SignalModule(int size);
  ~SignalModule();
  void run(double timestamp, double r, double g, double b);
  cv::Mat get_graph();
};

SignalModule::SignalModule(int size) {
  size_ = size;
  times.set(size);
  qx.set(size);
  qy.set(size);
  sig = new double[size];
  sig_filt = new double[size];
  filter_size = 33;
  filter = bandpassfilter(0.4, 2.0, 15, filter_size);
  frequency = 0.0;
}

SignalModule::~SignalModule() {
  delete[] sig;
  delete[] filter;
}

void SignalModule::run(double timestamp, double r, double g, double b) {
  double x = 3 * r - 2 * g;
  double y = 1.5 * r + g - 1.5 * b;
  qx.push_back(x);
  qy.push_back(y);

  double beta = qx.std() / (qy.std() + 1e-9);

  for (int i = 0; i < size_; i++) {
    sig[i] = beta * qx.data[i] - qy.data[i];
  }

  // filtering
  for (int i = filter_size; i < size_; i++) {
    double value = 0;
    for (int j = 0; j < filter_size; j++) {
      value += (sig[i - j] * filter[j]);
    }
    sig_filt[i] = value;
  }

  std::vector<double> src(sig_filt, sig_filt + size_);

  peaks = signalgo::find_peaks(src, 0.1);
}

cv::Mat SignalModule::get_graph() {
  int scale = 160;
  cv::Mat graph1 = cv::Mat::zeros(cv::Size(scale, size_), CV_8UC3);
  double *s = sig;
  double min = s[0];
  double max = s[0];
  for (int i = 1; i < size_; i++) {
    if (s[i] > max) {
      max = s[i];
    }
    if (s[i] < min) {
      min = s[i];
    }
  }
  double max_min = max - min + 1e-6;

  int s_prev = (int)((s[0] - min) / max_min * scale);
  for (int i = 1; i < size_; i++) {
    double v = s[i];
    v = (v - min) / max_min;
    int s = (int)(v * scale);

    int p0 = std::min(s, s_prev);
    int p1 = std::max(s, s_prev);
    for (int p = p0; p < p1; p++) {
      graph1.data[(i * scale + p) * 3 + 1] = 255; // green channel
    }
    s_prev = s;
  }

  cv::Mat graph2 = cv::Mat::zeros(cv::Size(scale, size_), CV_8UC3);
  s = sig_filt;
  min = s[filter_size];
  max = s[filter_size];
  for (int i = filter_size + 1; i < size_; i++) {
    if (s[i] > max) {
      max = s[i];
    }
    if (s[i] < min) {
      min = s[i];
    }
  }
  max_min = max - min + 1e-6;

  s_prev = (int)((s[filter_size] - min) / max_min * scale);
  for (int i = filter_size + 1; i < size_; i++) {
    double v = s[i];
    v = (v - min) / max_min;
    int s = (int)(v * scale);

    int p0 = std::min(s, s_prev);
    int p1 = std::max(s, s_prev);
    for (int p = p0; p < p1; p++) {
      graph2.data[(i * scale + p) * 3 + 1] = 255; // green channel
    }
    s_prev = s;
  }

  for (int i = 0; i < peaks.size(); i++) {
    int peak = peaks[i];
    for (int j = 0; j < scale; j++) {
      graph2.data[3 * (peak * scale + j) + 2] = 255;
    }
  }

  cv::Mat graph;
  cv::hconcat(graph1, graph2, graph);

  return graph;
}

void filtering(int size, double *samples) {}

int test() {
  cv::namedWindow("view");
  cv::moveWindow("view", 0, 0);
  cv::namedWindow("crop");
  cv::moveWindow("crop", 0, 400);
  cv::namedWindow("mask");
  cv::moveWindow("mask", 0, 560);
  cv::namedWindow("hist");
  cv::moveWindow("hist", 320, 400);

  cam::Cam camera(0);
  int width = camera.get_width() / 2;
  int height = camera.get_height() / 2;

  auto s_module = SignalModule(360);

  cv::Mat image;

  while (camera.read()) {
    auto frame = camera.get_frame();
    cv::resize(frame, image, cv::Size(width, height));

    int bsize = 120;
    cv::Rect bbox((width - bsize) / 2, (height - bsize) / 2, bsize, bsize);
    cv::Mat crop = image(bbox).clone();

    cv::Mat hsv;
    cv::cvtColor(crop, hsv, cv::COLOR_BGR2HSV);

    std::vector<cv::Mat> hsv_planes;
    cv::split(hsv, hsv_planes);
    // cv::imshow("hsv", hsv_planes[1]);

    int hist_size = 256;
    float range[] = {0, 256};
    const float *hist_range[] = {range};
    bool uniform = true, accumulate = false;
    cv::Mat hist, temp;
    cv::calcHist(&hsv_planes[1], 1, 0, temp, hist, 1, &hist_size, hist_range,
                 uniform, accumulate);

    cv::Mat hist_m;
    cv::medianBlur(hist, hist_m, 5);

    double minVal;
    double maxVal;
    cv::Point minLoc;
    cv::Point maxLoc;

    cv::minMaxLoc(hist_m, &minVal, &maxVal, &minLoc, &maxLoc);
    int hist_max = maxLoc.y;

    double alpha = 0.4;
    double TH_range = alpha * hist_max;
    int range0 = (int)(hist_max - TH_range / 2);
    int range1 = (int)(hist_max + TH_range / 2);

    cv::Mat sat = hsv_planes[1];
    int sat_size = sat.rows * sat.cols;

    std::vector<int> mask;
    for (int i = 0; i < sat_size; i++) {
      if (sat.data[i] > range0) {
        if (sat.data[i] < range1) {
          mask.push_back(i);
        }
      }
    }

    double r_sum = 0, g_sum = 0, b_sum = 0;
    for (int i = 0; i < mask.size(); i++) {
      int index = mask[i];
      b_sum += crop.data[3 * index];
      g_sum += crop.data[3 * index + 1];
      r_sum += crop.data[3 * index + 2];
    }
    double r = r_sum / (mask.size() + 1e-9);
    double g = g_sum / (mask.size() + 1e-9);
    double b = b_sum / (mask.size() + 1e-9);

    s_module.run(camera.get_time(), r, g, b);

    int scale = 320;
    cv::Mat hist_view = cv::Mat::zeros(cv::Size(scale, 256), CV_8UC3);
    for (int i = 0; i < 256; i++) {
      int value = (int)hist_m.at<float>(i);
      if (value >= scale)
        value = scale - 1;
      for (int j = 0; j < value; j++) {
        hist_view.data[3 * (i * scale + j) + 1] = 64;
      }
    }
    for (int j = 0; j < scale; j++) {
      hist_view.data[3 * (hist_max * scale + j) + 2] = 255;
      hist_view.data[3 * (range0 * scale + j) + 0] = 255;
      hist_view.data[3 * (range1 * scale + j) + 0] = 255;
    }

    cv::Mat mask_view = cv::Mat::zeros(crop.size(), CV_8UC3);
    for (int i = 0; i < mask.size(); i++) {
      int index = 3 * mask[i];
      mask_view.data[index + 0] = crop.data[index + 0];
      mask_view.data[index + 1] = crop.data[index + 1];
      mask_view.data[index + 2] = crop.data[index + 2];
    }

    cv::Mat graph = s_module.get_graph();

    cv::rectangle(image, bbox, cv::Scalar(0, 255, 0), 1);
    cv::Mat view;
    cv::hconcat(image, graph, view);

    cv::imshow("view", view);
    cv::imshow("crop", crop);
    cv::imshow("hist", hist_view);
    cv::imshow("mask", mask_view);
  }

  return 0;
}

int main() {
  test();
  return 0;
}