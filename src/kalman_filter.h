#ifndef KALMAN_FILTER_H
#define KALMAN_FILTER_H

#include <tuple>
#include "Eigen/Dense"

namespace tomi92 {
namespace kalman_filter {

/**
 * Extended Kálmán Filter Predict
 *
 * x: State Vector
 * P: State Covariance Matrix
 * dt: Elapsed time since last measurement
 * dg_f: (x, dt) -> x': State Transition Function
 * dg_F: (x, dt) -> F: State Transition Matrix generator
 * dg_Q: (dt) -> Q: Process Covariance Matrix generator
 */
template <typename Dg_f, typename Dg_F, typename Dg_Q>
const auto EkfPredict(const Eigen::VectorXd &x, const Eigen::MatrixXd &P,
                      const double dt, const Dg_f &dg_f, const Dg_F &dg_F,
                      const Dg_Q &dg_Q) {
  const Eigen::MatrixXd &F = dg_F(x, dt);

  return std::tuple<Eigen::VectorXd, Eigen::MatrixXd>(
      dg_f(x, dt), F * P * F.transpose() + dg_Q(dt));
}

/**
 * Kálmán Filter Predict
 *
 * x: State Vector
 * P: State Covariance Matrix
 * dt: Elapsed time since last measurement
 * dg_F: (dt) -> F: State Transition Matrix generator
 * dg_Q: (dt) -> Q: Process Covariance Matrix generator
 */
template <typename Dg_F, typename Dg_Q>
const auto KfPredict(const Eigen::VectorXd &x, const Eigen::MatrixXd &P,
                     const double dt, const Dg_F &dg_F, const Dg_Q &dg_Q) {
  const Eigen::MatrixXd &F = dg_F(dt);

  const auto dg_f = [&F](const auto &x, const double dt) -> Eigen::VectorXd {
    return F * x;
  };

  const auto dg_F2 = [&F](
      const auto &x, const double dt) -> const Eigen::MatrixXd & { return F; };

  return EkfPredict(x, P, dt, dg_f, dg_F2, dg_Q);
}

/**
 * Extended Kálmán Filter Measure
 *
 * x: State Vector
 * P: State Covariance Matrix
 * dt: Elapsed time since last measurement
 * z: Raw measurement
 * dg_h: (x, z) -> y: "Measurement function"
 * dg_H: (x) -> H: Measurement Matrix generator
 * R: Measurement Covariance Matrix
 */
template <typename Dg_h, typename Dg_H>
const auto EkfMeasure(const Eigen::VectorXd &x, const Eigen::MatrixXd &P,
                      const double dt, const Eigen::VectorXd &z,
                      const Dg_h &dg_h, const Dg_H &dg_H,
                      const Eigen::MatrixXd &R) {
  const Eigen::MatrixXd &H = dg_H(x);
  const Eigen::MatrixXd Ht = H.transpose();

  Eigen::MatrixXd y = dg_h(x, z);
  const Eigen::MatrixXd S = H * P * Ht + R;
  const Eigen::MatrixXd K = P * Ht * S.inverse();

  const size_t x_size = x.size();
  const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(x_size, x_size);

  return std::tuple<Eigen::VectorXd, Eigen::MatrixXd>(x + K * y, (I - K * H) * P);
}

/**
 * Kálmán Filter Measure
 *
 * x: State Vector
 * P: State Covariance Matrix
 * dt: Elapsed time since last measurement
 * z: Raw measurement
 * H: Measurement Matrix
 * R: Measurement Covariance Matrix
 */
inline const auto KfMeasure(const Eigen::VectorXd &x, const Eigen::MatrixXd &P,
                            const double dt, const Eigen::VectorXd &z,
                            const Eigen::MatrixXd &H,
                            const Eigen::MatrixXd &R) {
  const auto dg_h = [&H](const auto &x,
                         const auto &z) -> const Eigen::MatrixXd {
    return z - H * x;
  };

  const auto dg_H = [&H](const auto &x) -> const Eigen::MatrixXd & {
    return H;
  };

  return EkfMeasure(x, P, dt, z, dg_h, dg_H, R);
}

}  // namespace kalman_filter
}  // namespace tomi92

#endif  // KALMAN_FILTER_H
