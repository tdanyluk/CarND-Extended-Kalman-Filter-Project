#ifndef FUSION_EKF_1_H
#define FUSION_EKF_1_H

#include <cstdint>
#include "measurement_package.h"

namespace tomi92 {
namespace kalman_filter {

class FusionEKF {
 public:
  FusionEKF();

  virtual ~FusionEKF();

  void ProcessMeasurement(const MeasurementPackage& measurement_pack);

  const Eigen::VectorXd& GetX() { return x_; }

 private:
  // Variable members
  Eigen::VectorXd x_;
  Eigen::MatrixXd P_;
  bool is_initialized_;
  int64_t previous_timestamp_;

  // Constant members
  double noise_ax_;
  double noise_ay_;
  Eigen::MatrixXd R_laser_;
  Eigen::MatrixXd R_radar_;
  Eigen::MatrixXd H_laser_;

  static const double Normalize(const double rad);
  static const Eigen::MatrixXd CalculateF(const double dt);
  static const Eigen::MatrixXd CalculateQ(const double noise_ax,
                                          const double noise_ay,
                                          const double dt);
  static const Eigen::MatrixXd CalculateHjRadar(const Eigen::VectorXd& x_state);
  static const Eigen::VectorXd hRadar(const Eigen::VectorXd& x,
                                      const Eigen::VectorXd& z);
  static const Eigen::VectorXd RadarToPV(const Eigen::VectorXd& radar_data);
  static const Eigen::VectorXd LaserToPV(const Eigen::VectorXd& laser_data);
};

}  // namespace kalman_filter
}  // namespace tomi92

#endif  // FUSION_EKF_1_H