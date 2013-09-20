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
	cv::Mat m_src; //Դͼ��
	cv::Mat m_dst; //Ŀ��ͼ��

	std::vector<cv::Mat> m_buffer; //ͼ�񻺴�
	int m_last;		//ͼ��ʼλ��

	std::vector<double> m_angle_vector; //�Ƕ�����
	std::vector<double> m_ma_feature; //�Ƕ�����

	unsigned int m_diff_threshold;	//������ֵ
	unsigned int m_buffer_length;	//����������
	double m_mhi_duration; //mhi����ʱ��

	cv::Mat	m_mhi; //�˶���ʷͼ��(Motion History Image)
	cv::Mat	m_orient; //�ݶȷ���Ƕ�ͼ��
	cv::Mat	m_mask; //�ݶ���Ч����ͼ��
	cv::Mat	m_segmask; //�˶�����ͼ��

	std::vector<cv::Rect> m_storage; //�˶����λ��

};