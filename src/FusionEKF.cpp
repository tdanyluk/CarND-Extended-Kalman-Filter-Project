#include "FusionEKF.h"

#include <functional>
#include <iostream>
#include "kalman_filter.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace tomi92 {
namespace kalman_filter {

FusionEKF::FusionEKF()
    : x_(4),
      P_(4, 4),
      is_initialized_(false),
      previous_timestamp_(0),
      noise_ax_(9),
      noise_ay_(9),
      R_laser_(2, 2),
      R_radar_(3, 3),
      H_laser_(2, 4) {
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;
}

FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage& measurement_pack) {
  if (!is_initialized_) {
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      x_ << RadarToPV(measurement_pack.raw_measurements_);
    } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      x_ << LaserToPV(measurement_pack.raw_measurements_);
    }
    P_ << 1, 0, 0, 0,
          0, 1, 0, 0,
          0, 0, 1000,
          0, 0, 0, 0, 1000;
    previous_timestamp_ = measurement_pack.timestamp_;
    is_initialized_ = true;
    return;
  }

  double dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  // Just to prevent strange behaviour when changing datasets
  if (dt < 0 || dt > 1) {
    dt = 1;
  }
  previous_timestamp_ = measurement_pack.timestamp_;

  auto calcQ =
      bind(FusionEKF::CalculateQ, noise_ax_, noise_ay_, placeholders::_1);
  tie(x_, P_) = KfPredict(x_, P_, dt, FusionEKF::CalculateF, calcQ);

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    tie(x_, P_) = EkfMeasure(x_, P_, dt, measurement_pack.raw_measurements_,
                             FusionEKF::hRadar, FusionEKF::CalculateHjRadar, R_radar_);
  } else {
    tie(x_, P_) = KfMeasure(x_, P_, dt, measurement_pack.raw_measurements_,
                            H_laser_, R_laser_);
  }

  cout << "x_ = " << x_ << endl;
  cout << "P_ = " << P_ << endl;
}

const double FusionEKF::Normalize(double rad) {
  constexpr double Pi = 3.14159265358979323846;
  constexpr double TwoPi = 2 * Pi;

  while (rad <= -Pi) {
    rad += TwoPi;
  }
  while (rad > Pi) {
    rad -= TwoPi;
  }

  return rad;
}

const Eigen::MatrixXd FusionEKF::CalculateF(const double dt) {
  MatrixXd F(4, 4);  // Will change with time
  F << 1, 0, dt, 0,
       0, 1, 0, dt, 
       0, 0, 1, 0, 
       0, 0, 0, 1;

  return F;
}

const Eigen::MatrixXd FusionEKF::CalculateQ(const double noise_ax,
                                            const double noise_ay,
                                            const double dt) {
  const auto dt2 = dt * dt;
  const auto dt3 = dt2 * dt;
  const auto dt4 = dt3 * dt;

  MatrixXd Q(4, 4);
  Q << dt4 / 4 * noise_ax, 0, dt3 / 2 * noise_ax, 0,
       0, dt4 / 4 * noise_ay, 0, dt3 / 2 * noise_ay,
       dt3 / 2 * noise_ax, 0, dt2 * noise_ax, 0,
       0, dt3 / 2 * noise_ay, 0, dt2 * noise_ay;

  return Q;
}

const Eigen::MatrixXd FusionEKF::CalculateHjRadar(const VectorXd& x_state) {
  MatrixXd Hj(3, 4);

  const double px = x_state(0);
  const double py = x_state(1);
  const double vx = x_state(2);
  const double vy = x_state(3);

  const double n = px * px + py * py;
  const double n1_2 = sqrt(n);
  const double n3_2 = n1_2 * n1_2 * n1_2;

  if (n < 0.0001) {
    return MatrixXd::Zero(3, 4);
  }

  Hj(0, 0) = px / n1_2;
  Hj(0, 1) = py / n1_2;
  Hj(0, 2) = 0;
  Hj(0, 3) = 0;

  Hj(1, 0) = -py / n;
  Hj(1, 1) = px / n;
  Hj(1, 2) = 0;
  Hj(1, 3) = 0;

  Hj(2, 0) = py * (vx * py - vy * px) / n3_2;
  Hj(2, 1) = px * (vy * px - vx * py) / n3_2;
  Hj(2, 2) = px / n1_2;
  Hj(2, 3) = py / n1_2;

  return Hj;
}

const Eigen::VectorXd FusionEKF::hRadar(const Eigen::VectorXd& x,
                                        const Eigen::VectorXd& z) {
  const double px = x(0);
  const double py = x(1);
  const double vx = x(2);
  const double vy = x(3);

  VectorXd hx(3);
  const double n = sqrt(px * px + py * py);

  if (n < 0.0001) {
    hx << n, 0, 0;
  } else {
    hx << n, atan2(py, px), (px * vx + py * vy) / n;
  }

  VectorXd y = z - hx;
  y(1) = Normalize(y(1));
  return y;
}

const Eigen::VectorXd FusionEKF::RadarToPV(const VectorXd& radar_data) {
  const double rho = radar_data(0);
  const double phi = radar_data(1);
  const double rho_dot = radar_data(2);

  const double px = rho * cos(phi);
  const double py = rho * sin(phi);
  const double vx = 0;
  const double vy = 0;

  VectorXd pv(4);
  pv << px, py, vx, vy;
  return pv;
}

const Eigen::VectorXd FusionEKF::LaserToPV(const VectorXd& laser_data) {
  const double px = laser_data(0);
  const double py = laser_data(1);

  VectorXd pv(4);
  pv << px, py, 0, 0;
  return pv;
}

}  // namespace kalman_filter
}  // namespace tomi92
