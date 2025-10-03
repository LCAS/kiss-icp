#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <geometry_msgs/msg/quaternion.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>

class KalmanFilter {
public:
    KalmanFilter() {
        // state = [x, y, theta, vx, vy, omega]
        x_.setZero();
        P_ = Eigen::MatrixXd::Identity(6, 6) * 0.2;
        Q_ = Eigen::MatrixXd::Identity(6, 6) * 0.1;
        R_ = Eigen::Matrix3d::Identity() * 0.1;
    }

    nav_msgs::msg::Odometry Update(const nav_msgs::msg::Odometry::ConstSharedPtr &msg) {
        // ---- Time update ----
        rclcpp::Time now(msg->header.stamp);
        if (last_time_.nanoseconds() == 0) {
            last_time_ = now;
        }
        double dt = (now - last_time_).seconds();
        if (dt <= 0.0) dt = 0.1;
        last_time_ = now;

        // State transition (constant velocity model)
        Eigen::MatrixXd F = Eigen::MatrixXd::Identity(6, 6);
        F(0, 3) = dt;
        F(1, 4) = dt;
        F(2, 5) = dt;

        x_ = F * x_;
        P_ = F * P_ * F.transpose() + Q_;

        // ---- Measurement (pose only: x, y, yaw) ----
        Eigen::Vector3d z;
        z(0) = msg->pose.pose.position.x;
        z(1) = msg->pose.pose.position.y;
        z(2) = yawFromQuaternion(msg->pose.pose.orientation);

        // H maps pose measurements into the 6-state vector
        Eigen::Matrix<double, 3, 6> H;
        H.setZero();
        H(0, 0) = 1;
        H(1, 1) = 1;
        H(2, 2) = 1;

        // normalize innovation for angle
        auto normalizeAngle = [](double a) {
            while (a > M_PI) a -= 2.0 * M_PI;
            while (a <= -M_PI) a += 2.0 * M_PI;
            return a;
        };

        Eigen::Vector3d y = z - H * x_;
        y(2) = normalizeAngle(y(2)); // wrap yaw innovation

        // measurement noise for pose (3x3)
        Eigen::Matrix3d R_pose = R_; // reuse R_ initialized earlier

        Eigen::Matrix3d S = H * P_ * H.transpose() + R_pose;
        Eigen::Matrix<double, 6, 3> K = P_ * H.transpose() * S.inverse();

        x_ = x_ + K * y;
        P_ = (Eigen::MatrixXd::Identity(6, 6) - K * H) * P_;

        // ---- Prepare output odom ----
        nav_msgs::msg::Odometry odom_out;
        odom_out.header = msg->header;
        odom_out.child_frame_id = msg->child_frame_id;

        odom_out.pose.pose.position.x = x_(0);
        odom_out.pose.pose.position.y = x_(1);
        odom_out.pose.pose.position.z = 0.0;
        odom_out.pose.pose.orientation = quaternionFromYaw(x_(2));

        // Fill pose covariance (top-left 3x3 of P_)
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                odom_out.pose.covariance[i * 6 + j] = P_(i, j);
            }
        }

        // ---- Estimate twist from state (vx, vy, omega) ----
        odom_out.twist.twist.linear.x = x_(3);
        odom_out.twist.twist.linear.y = x_(4);
        odom_out.twist.twist.linear.z = 0.0;
        odom_out.twist.twist.angular.x = 0.0;
        odom_out.twist.twist.angular.y = 0.0;
        odom_out.twist.twist.angular.z = x_(5);

        // Fill twist covariance from P_ submatrix [3:5,3:5] into 6x6 twist.covariance top-left
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                odom_out.twist.covariance[i * 6 + j] = P_(3 + i, 3 + j);
            }
        }

        return odom_out;
    }

private:
    Eigen::Matrix<double, 6, 1> x_;
    Eigen::MatrixXd P_, Q_;
    Eigen::Matrix3d R_;
    rclcpp::Time last_time_;

    // ---- Helper functions ----
    static double yawFromQuaternion(const geometry_msgs::msg::Quaternion &q) {
        // yaw (theta) from quaternion
        return std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
    }

    static geometry_msgs::msg::Quaternion quaternionFromYaw(double yaw) {
        // convert yaw to quaternion (roll=pitch=0)
        geometry_msgs::msg::Quaternion q;
        double half_yaw = yaw * 0.5;
        q.w = std::cos(half_yaw);
        q.x = 0.0;
        q.y = 0.0;
        q.z = std::sin(half_yaw);
        return q;
    }
};
