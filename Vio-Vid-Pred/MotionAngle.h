#pragma once

#include <iostream>
#include <vector>
#include <ctime>
#include <cmath>

#include <opencv.hpp>

using namespace std;
using namespace cv;

#define MAX_TIME_DELTA 0.5
#define MIN_TIME_DELTA 0.05

class MotionAngle{

public:
	MotionAngle(void);
	MotionAngle(const double m_mhi_duration, const unsigned int diff_threshold = 30);
	~MotionAngle(void);

public:
	void motion_angle_feature(const cv::Mat& src, std::vector<double>& ma_feature, bool isPicture);

private:
	void draw_picture(cv::Mat& dst, const cv::Rect comp_rect, const cv::Mat& silh,
		bool isWhole, double angle);
	void cal_feature();

private:
	cv::Mat m_src; //源图像
	cv::Mat m_dst; //目标图像

	std::vector<cv::Mat> m_buffer; //图像缓存
	int m_last;		//图像开始位置

	std::vector<double> m_angle_vector; //角度向量
	std::vector<double> m_ma_feature; //角度特征

	unsigned int m_diff_threshold;	//掩码阈值
	unsigned int m_buffer_length;	//缓冲区长度
	double m_mhi_duration; //mhi持续时间

	cv::Mat	m_mhi; //运动历史图像(Motion History Image)
	cv::Mat	m_orient; //梯度方向角度图像
	cv::Mat	m_mask; //梯度有效掩码图像
	cv::Mat	m_segmask; //运动区域图像

	std::vector<cv::Rect> m_storage; //运动组件位置

};