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

        P_ = Eigen::MatrixXd::Identity(6, 6) * 0.1;
        Q_ = Eigen::MatrixXd::Identity(6, 6) * 0.01;
        R_ = Eigen::Matrix3d::Identity() * 0.05;
    }

    nav_msgs::msg::Odometry Update(const nav_msgs::msg::Odometry::ConstSharedPtr &msg) {
        // ---- Time update ----
        rclcpp::Time now(msg->header.stamp);  // convert from msg timestamp
        if (last_time_.nanoseconds() == 0) {
            last_time_ = now;  // first message
        }
        double dt = (now - last_time_).seconds();
        if (dt <= 0.0) dt = 0.1;
        last_time_ = now;

        Eigen::MatrixXd F = Eigen::MatrixXd::Identity(6, 6);
        F(0, 3) = dt;
        F(1, 4) = dt;
        F(2, 5) = dt;

        x_ = F * x_;
        P_ = F * P_ * F.transpose() + Q_;

        // ---- Measurement ----
        Eigen::Vector3d z;
        z(0) = msg->pose.pose.position.x;
        z(1) = msg->pose.pose.position.y;
        z(2) = yawFromQuaternion(msg->pose.pose.orientation);

        Eigen::MatrixXd H(3, 6);
        H.setZero();
        H(0, 0) = 1;
        H(1, 1) = 1;
        H(2, 2) = 1;

        Eigen::Vector3d y = z - H * x_;
        Eigen::Matrix3d S = H * P_ * H.transpose() + R_;
        Eigen::MatrixXd K = P_ * H.transpose() * S.inverse();

        x_ = x_ + K * y;
        P_ = (Eigen::MatrixXd::Identity(6, 6) - K * H) * P_;

        // ---- Fill output ----
        nav_msgs::msg::Odometry odom_out;
        odom_out.header = msg->header;
        odom_out.child_frame_id = msg->child_frame_id;

        odom_out.pose.pose.position.x = x_(0);
        odom_out.pose.pose.position.y = x_(1);
        odom_out.pose.pose.position.z = 0.0;

        odom_out.pose.pose.orientation = quaternionFromYaw(x_(2));

        for (int i = 0; i < 6; i++)
            for (int j = 0; j < 6; j++)
                odom_out.pose.covariance[i * 6 + j] = (i < 3 && j < 3) ? P_(i, j) : 0.0;

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
