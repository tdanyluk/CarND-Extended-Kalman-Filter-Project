#include "FusionEKF.h"

#include <iostream>
#include "Eigen/Dense"
#include "tools.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  // measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  // measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  VectorXd x(4);
  x << 0, 0, 0, 0;

  MatrixXd F(4, 4);  // Will change with time
  F << 1, 0, 1, 0,
       0, 1, 0, 1,
       0, 0, 1, 0,
       0, 0, 0, 1;

  MatrixXd H(2, 4);
  H << 1, 0, 0, 0,
       0, 1, 0, 0;

  MatrixXd Q(4, 4);  // Init later

  noise_ax_ = 9;
  noise_ay_ = 9;

  MatrixXd P(4, 4);
  P << 1, 0, 0, 0,
       0, 1, 0, 0,
      0, 0, 1000, 0,
      0, 0, 0, 1000;

  ekf_.Init(x, P, F, H, R_laser_, R_radar_, Q);
  /**
  TODO:
    * Finish initializing the FusionEKF.
    * Set the process and measurement noises
  */
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    /**
    TODO:
      * Initialize the state ekf_.x_ with the first measurement.
      * Create the covariance matrix.
      * Remember: you'll need to convert radar from polar to cartesian
    coordinates.
    */
    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      ekf_.x_ << tools.RadarToPV(measurement_pack.raw_measurements_);
    } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
      ekf_.x_ << measurement_pack.raw_measurements_(0), measurement_pack.raw_measurements_(1), 0, 0;
    }

    last_time_ = measurement_pack.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;

    return;
  }

  double dt = (measurement_pack.timestamp_ - last_time_) / 1000000.0;
  last_time_ = measurement_pack.timestamp_;

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  /**
   TODO:
     * Update the state transition matrix F according to the new elapsed time.
      - Time is measured in seconds.
     * Update the process noise covariance matrix.
     * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */

  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;

  double dt2 = dt * dt;
  double dt3 = dt2 * dt;
  double dt4 = dt3 * dt;
  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.Q_ << dt4 / 4 * noise_ax_, 0, dt3 / 2 * noise_ax_, 0,
             0, dt4 / 4 * noise_ay_, 0, dt3 / 2 * noise_ay_,
             dt3 / 2 * noise_ax_, 0, dt2 * noise_ax_, 0,
             0, dt3 / 2 * noise_ay_, 0, dt2 * noise_ay_;

  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
   TODO:
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    ekf_.Hj_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    // Laser updates
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
