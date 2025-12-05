#include <Eigen/Dense>
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/vector3.hpp"


using Eigen::Matrix3d;
using Eigen::Vector3d;

class KalmanFilterNode : public rclcpp::Node
{
public:
    KalmanFilterNode() : Node("kalman_filter_node")
    {
        cmd_sub_ = this->create_subscription<geometry_msgs::msg::Vector3>(
            "/broyden_controller/command", 10, std::bind(&KalmanFilterNode::cmd_callback, this, std::placeholders::_1));
        meas_sub_ = this->create_subscription<geometry_msgs::msg::Point>(
            "/tracked_tip", 10, std::bind(&KalmanFilterNode::meas_callback, this, std::placeholders::_1));
        est_pub_ = this->create_publisher<geometry_msgs::msg::Point>("/kalman_estimate", 10);

        // Timer for regular publishing (10Hz)
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&KalmanFilterNode::timer_callback, this));

        // Kalman filter initialization
        J_ = Matrix3d::Identity();
        H_ = Matrix3d::Identity();
        Q_ = 1e-1 * Matrix3d::Identity();
        R_ = 1e-2 * Matrix3d::Identity();
        P_ = 1e-2 * Matrix3d::Identity();
        S_prev_ = Vector3d::Zero();

        dt_ = 0.1;
        cov_threshold_ = 54e-4;
        prediction_allowed_ = true;
        initialized_ = false;
        meas_counter_ = 0;

    }

private:
    // ROS interfaces
    rclcpp::Subscription<geometry_msgs::msg::Vector3>::SharedPtr cmd_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr meas_sub_;
    rclcpp::Publisher<geometry_msgs::msg::Point>::SharedPtr est_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    // Kalman state
    Matrix3d J_, H_, Q_, R_, P_;
    Vector3d S_prev_; // current state estimate
    double dt_;
    double cov_threshold_;
    bool prediction_allowed_;
    bool initialized_;
    int meas_counter_;

    // To keep 2 most recent for Broyden
    Vector3d prev_hand_, prev_tip_;
    bool have_prev_hand_ = false, have_prev_tip_ = false;

    // Velocity command callback
    void cmd_callback(const geometry_msgs::msg::Vector3::SharedPtr msg) {
        if (!prediction_allowed_) {
            RCLCPP_WARN(this->get_logger(), "Uncertainty too high, waiting for measurement...");
            return; // Don't predict further until correction!
        }

        Vector3d hand;
        hand << msg->x, msg->y, msg->z;
        // If we have a previous hand and tip, do Broyden update
        if (have_prev_hand_ && have_prev_tip_) {
            J_ = broyden_update(J_, prev_hand_, hand, prev_tip_, S_prev_);
        }
        prev_hand_ = hand*dt_; 
        have_prev_hand_ = true;

        // Kalman prediction
        Vector3d delta_p = hand;
        Vector3d S_prior = S_prev_ + J_ * delta_p;
        Matrix3d P_pred = P_ + Q_ * dt_;

        S_prev_ = S_prior;
        P_ = P_pred;

        // If uncertainty explode stop predicting
        if (P_.trace() > cov_threshold_) {
            prediction_allowed_ = false;
            RCLCPP_WARN(this->get_logger(), "Uncertainty exceeded threshold, pausing predictions until next measurement.");
        }
    }

    // Measurement callback (correction)
    void meas_callback(const geometry_msgs::msg::Point::SharedPtr msg) {
        Vector3d tip;
        tip << msg->x, msg->y, msg->z;
        // For Broyden, save last tip (for next command callback)
        if (have_prev_tip_) {
            prev_tip_ = tip;
        }
        have_prev_tip_ = true;

        // Correction (posterior update)
        Matrix3d K = P_ * H_.transpose() * (H_ * P_ * H_.transpose() + R_).inverse();
        Vector3d S_post = S_prev_ + K * (tip - H_ * S_prev_);
        P_ = (Matrix3d::Identity() - K * H_) * P_;
        S_prev_ = S_post;

        prediction_allowed_ = true; // Allow prediction again
        
        //
        double error = (S_prev_ - tip).norm();
        RCLCPP_INFO(this->get_logger(),
        "Corrected: x=%f y=%f z=%f | Meas: x=%f y=%f z=%f | Error=%.6f",
        S_prev_(0), S_prev_(1), S_prev_(2),
        tip(0), tip(1), tip(2),
        error);
        //


    meas_counter_++;



    }

    // Timer callback: regularly publish the state estimate (10Hz)
    void timer_callback() {
        geometry_msgs::msg::Point est_msg;
        est_msg.x = S_prev_(0);
        est_msg.y = S_prev_(1);
        est_msg.z = S_prev_(2);
        est_pub_->publish(est_msg);

    }

    // Broyden's update
    Matrix3d broyden_update(const Matrix3d& J_old,
                            const Vector3d& q_old, const Vector3d& q_new,
                            const Vector3d& x_old, const Vector3d& x_new) {
        Vector3d dq = q_new - q_old;
        Vector3d dx = x_new - x_old;
        double dq_norm_sq = dq.squaredNorm();
        if (dq_norm_sq < 1e-12)
            return J_old;
        return J_old + ((dx - J_old * dq) / dq_norm_sq) * dq.transpose();
    }
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<KalmanFilterNode>());
    rclcpp::shutdown();
    return 0;
}
