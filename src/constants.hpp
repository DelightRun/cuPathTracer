#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

#include <stdlib.h>

namespace crt {

constexpr float kEpsilon = 1e-5;
constexpr float kPi = 3.1415926535897932384626422832795028841971;
constexpr float kTwoPi = 2 * kPi;
constexpr float kArcPerAngle = kPi / 180;
constexpr float kSQRTOfOneThird = 0.5773502691896257645091487805019574556476;

constexpr float kAirIoR = 1.0f;

constexpr size_t kMaximumSize = (size_t)-1;
constexpr size_t kInvalidIndex = kMaximumSize;

constexpr size_t kThreadsPerBlock = 256;

}  // namespace crt

#endif