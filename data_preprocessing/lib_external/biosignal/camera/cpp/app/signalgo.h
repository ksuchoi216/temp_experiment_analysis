#ifndef SIGNALGO_H
#define SIGNALGO_H

#include <vector>

namespace signalgo {
std::vector<double> ema(std::vector<double> src, const double alpha);
std::vector<double> subtract_ema(std::vector<double> src, const double alpha);

std::vector<int> find_peaks(std::vector<double> src,
                            const double min_prominence = -1.0,
                            const double max_prominence = -1.0,
                            const double min_width = -1.0,
                            const double max_width = -1.0, const int wlen = -1,
                            const double rel_height = 0.5);

int get_peaks_count(int size, double *data, const double min_prominence = 75.0,
                    const double max_prominence = -1.0,
                    const double min_width = 15.0,
                    const double max_width = 250.0, const int wlen = 250,
                    const double rel_height = 0.5);
} // namespace signalgo

#endif