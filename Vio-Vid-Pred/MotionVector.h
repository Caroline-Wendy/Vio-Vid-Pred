#pragma once

#include <iostream>
#include <vector>
#include <ctime>

#include <opencv.hpp>

using namespace std;
using namespace cv;

class MotionVector{

public:
	MotionVector(void);
	MotionVector(double minDist, int maxCount = 100, double m_qualityLevel = 0.01);

public:
	void cal_motion_vector(const cv::Mat& src, std::vector<double>& motion_vector,  bool isPicture=false);

private:
	void motion_tracking(cv::Mat& pre_frame, cv::Mat& next_frame,
		vector<Point2f>& points_old, vector<Point2f>& points_new);

private:
	cv::Mat m_src;	//Դͼ��
	std::vector<cv::Mat> m_buffer;	//ͼ�񻺴�

	int m_last;		//����λ��
	int m_maxCount;					// corners��������

	double m_qualityLevel;			// �������ĵȼ�
	double m_minDist;				// ��������֮�����С����

	unsigned int m_motion_num;

	unsigned int m_buffer_length;
};