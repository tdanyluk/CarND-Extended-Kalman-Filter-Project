#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd>& estimations,
                              const vector<VectorXd>& ground_truth) {
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  // check the validity of the following inputs:
  //  * the estimation vector size should not be zero
  //  * the estimation vector size should equal ground truth vector size
  if (estimations.size() != ground_truth.size() || estimations.size() == 0) {
    cout << "Invalid estimation or ground_truth data" << endl;
    return rmse;
  }

  // accumulate squared residuals
  for (unsigned int i = 0; i < estimations.size(); ++i) {
    VectorXd residual = estimations[i] - ground_truth[i];

    // coefficient-wise multiplication
    residual = residual.array() * residual.array();
    rmse += residual;
  }

  // calculate the mean
  rmse = rmse / estimations.size();

  // calculate the squared root
  rmse = rmse.array().sqrt();

  // return the result
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  MatrixXd Hj(3, 4);
  // recover state parameters
  double px = x_state(0);
  double py = x_state(1);
  double vx = x_state(2);
  double vy = x_state(3);

  double n = px * px + py * py;
  double n1_2 = sqrt(n);
  double n3_2 = n1_2 * n1_2 * n1_2;

  if (fabs(n) < 0.001 ||  fabs(n1_2) < 0.001 || fabs(n3_2) < 0.001  ) {
    cout << "----------------- Error Zero DIV" << endl;

    return Hj;
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

VectorXd Tools::RadarToPV(const VectorXd& radar_data) {
  double rho = radar_data(0);      // Meter
  double phi = radar_data(1);      // Radian
  double rho_dot = radar_data(2);  // Meter / Second

  double px = rho * cos(phi);
  double py = rho * sin(phi);
  double vx = 0;
  double vy = 0;

  VectorXd pv(4);
  pv << px, py, vx, vy;
  return pv;
}