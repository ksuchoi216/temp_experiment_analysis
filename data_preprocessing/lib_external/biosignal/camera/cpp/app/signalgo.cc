#include "signalgo.h"

#include <algorithm>

// [References]
// https://github.com/scipy/scipy/blob/v1.7.1/scipy/signal/_peak_finding.py#L723-L1003
// https://github.com/scipy/scipy/blob/v1.7.1/scipy/signal/_peak_finding_utils.pyx

std::vector<double> signalgo::ema(std::vector<double> src, const double alpha) {
  std::vector<double> dst;
  const double _alpha = 1.0 - alpha;
  double y = src[0];
  dst.push_back(y);
  for (int i = 1; i < (int)src.size(); i++) {
    y = y * _alpha + src[i] * alpha;
    dst.push_back(y);
  }
  return dst;
}

std::vector<double> signalgo::subtract_ema(std::vector<double> src,
                                           const double alpha) {
  std::vector<double> dst;
  const double _alpha = 1.0 - alpha;
  double x = src[0];
  double y = x;
  double z = x - y;
  dst.push_back(z);
  for (int i = 1; i < (int)src.size(); i++) {
    x = src[i];
    y = y * _alpha + x * alpha;
    z = x - y;
    dst.push_back(z);
  }
  return dst;
}

typedef struct {
  std::vector<int> peaks;
  std::vector<double> prominences;
  std::vector<int> left_bases;
  std::vector<int> right_bases;
  std::vector<int> widths;
} peak_properties;

static peak_properties _local_maxima_1d(std::vector<double> x) {
  peak_properties props;

  int i = 1; // Pointer to current sample, first one can't be maxima
  int i_max = x.size() - 1; // Last sample can't be maxima
  while (i < i_max) {
    // Test if previous sample is smaller
    if (x[i - 1] < x[i]) {
      int i_ahead = i + 1; // Index to look ahead of current sample

      // Find next sample that is unequal to x[i]
      while (i_ahead < i_max && x[i_ahead] == x[i]) {
        i_ahead += 1;
      }

      // Maxima is found if next unequal sample is smaller than x[i]
      if (x[i_ahead] < x[i]) {
        int peak = (i + i_ahead - 1) / 2;
        props.peaks.push_back(peak);
        // skip samples that can't be maximum
        i = i_ahead;
      }
    }
    i++;
  }
  return props;
}

static peak_properties _peak_prominences(std::vector<double> x,
                                         std::vector<int> peaks,
                                         const double min_prominence,
                                         const double max_prominence,
                                         const int wlen) {
  peak_properties props;

  for (int peak_nr = 0; peak_nr < (int)peaks.size(); peak_nr++) {
    int peak = peaks[peak_nr];
    int i_min = 0;
    int i_max = x.size() - 1;
    int i = 0;

    if (2 <= wlen) {
      // Adjust window around the evaluated peak (within bounds);
      // if wlen is even the resulting window length is implicitly
      // rounded to next odd integer
      i_min = std::max(peak - wlen / 2, i_min);
      i_max = std::min(peak + wlen / 2, i_max);
    }

    // Find the left base in interval [i_min, peak]
    i = peak;
    int left_base = i;
    double left_min = x[peak];
    while (i_min <= i && x[i] <= x[peak]) {
      if (x[i] < left_min) {
        left_min = x[i];
        left_base = i;
      }
      i--;
    }

    // Find the right base in interval [peak, i_max]
    i = peak;
    int right_base = i;
    double right_min = x[peak];
    while (i <= i_max && x[i] <= x[peak]) {
      if (x[i] < right_min) {
        right_min = x[i];
        right_base = i;
      }
      i++;
    }

    double prominence = x[peak] - std::max(left_min, right_min);
    if (min_prominence <= prominence || min_prominence < 0) {
      if (prominence <= max_prominence || max_prominence < 0) {
        props.peaks.push_back(peak);
        props.prominences.push_back(prominence);
        props.left_bases.push_back(left_base);
        props.right_bases.push_back(right_base);
      }
    }
  }

  return props;
}

static peak_properties _peak_widths(std::vector<double> x,
                                    peak_properties props_,
                                    const double min_width,
                                    const double max_width,
                                    const double rel_height) {
  peak_properties props;

  auto peaks = props_.peaks;
  auto prominences = props_.prominences;
  auto left_bases = props_.left_bases;
  auto right_bases = props_.right_bases;

  for (int p = 0; p < (int)peaks.size(); p++) {
    int peak = peaks[p];
    int i_min = left_bases[p];
    int i_max = right_bases[p];
    int i = 0;

    double height = x[peak] - prominences[p] * rel_height;

    // Find intersection point on left side
    i = peak;
    while (i_min < i && height < x[i]) {
      i--;
    }
    double left_ip = i;
    if (x[i] < height) {
      // Interpolate if true intersection height is between samples
      left_ip += (height - x[i]) / (x[i + 1] - x[i]);
    }

    // Find intersection point on right side
    i = peak;
    while (i < i_max && height < x[i]) {
      i++;
    }
    double right_ip = i;
    if (x[i] < height) {
      // Interpolate if true intersection height is between samples
      right_ip -= (height - x[i]) / (x[i - 1] - x[i]);
    }

    double width = right_ip - left_ip;

    if (min_width <= width || min_width < 0) {
      if (width <= max_width || max_width < 0) {
        props.peaks.push_back(peak);
        props.widths.push_back(width);
      }
    }
  }

  return props;
}

std::vector<int> signalgo::find_peaks(std::vector<double> src,
                                      const double min_prominence,
                                      const double max_prominence,
                                      const double min_width,
                                      const double max_width, const int wlen,
                                      const double rel_height) {
  peak_properties props;
  props = _local_maxima_1d(src);

  if (0 <= min_prominence || 0 <= min_width || 0 <= max_width) {
    props = _peak_prominences(src, props.peaks, min_prominence, max_prominence,
                              wlen);
  }
  if (0 <= min_width || 0 <= max_width) {
    props = _peak_widths(src, props, min_width, max_width, rel_height);
  }

  return props.peaks;
}

int signalgo::get_peaks_count(int size, double *data,
                              const double min_prominence,
                              const double max_prominence,
                              const double min_width, const double max_width,
                              const int wlen, const double rel_height) {
  std::vector<double> sig;
  for (int i = 0; i < size; i++) {
    sig.push_back(data[i]);
  }
  /* preprocess */
  // auto sig2 = signalgo::subtract_ema(sig, 0.01);
  // auto sig3 = signalgo::ema(sig2, 0.05);
  auto peaks = signalgo::find_peaks(sig, min_prominence, max_prominence,
                                    min_width, max_width, wlen, rel_height);
  return peaks.size();
}