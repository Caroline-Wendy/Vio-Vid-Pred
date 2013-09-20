#pragma once

#include <iostream>
#include <vector>
#include <ctime>

#include <opencv.hpp>

using namespace std;
using namespace cv;

class MotionFlow{

public:
	MotionFlow(void);
	MotionFlow(int maxCount, double qualityLevel = 0.01, double minDist = 30);

public:
	void cal_motion_vector(const cv::Mat& src, std::vector<double>& motion_vector,  bool isPicture=false);

private:
	void motion_tracking(cv::Mat& pre_frame, cv::Mat& next_frame,
		vector<Point2f>& points_old, vector<Point2f>& points_new);

private:
	cv::Mat m_src; //源图像
	std::vector<cv::Mat> m_buffer; //图像缓存

	unsigned int m_buffer_length; //缓存长度
	int m_last; //起始缓存位置
	unsigned int m_point_record; //上一个interest point数

	int m_maxCount; //corners的最大个数
	double m_qualityLevel; //特征检测的等级
	double m_minDist; //两特征点之间的最小距离

};