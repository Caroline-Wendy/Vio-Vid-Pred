#pragma once

#include <iostream>
#include <vector>
#include <ctime>

#include <opencv.hpp>

using namespace std;
using namespace cv;

#define MAX_TIME_DELTA 0.5
#define MIN_TIME_DELTA 0.05

class MotionAngle{

public:
	MotionAngle(void);
	MotionAngle(const double m_mhi_duration, 
		const unsigned int diff_threshold = 30, const unsigned int buffer_length = 4);
	~MotionAngle(void);

public:
	void update_mhi(const cv::Mat& src, std::vector<double>& angle_vector, bool isPicture=false);

private:
	void draw_picture(cv::Mat& dst, const cv::Rect comp_rect, 
		const cv::Mat& silh, bool isWhole, double angle);

private:
	cv::Mat m_src;	//Դͼ��
	cv::Mat m_dst;	//Ŀ��ͼ��
	unsigned int m_diff_threshold;	//������ֵ
	unsigned int m_buffer_length;	//����������
	double m_mhi_duration; //mhi����ʱ��
	std::vector<cv::Mat> m_buffer;	//ͼ�񻺴�

	cv::Mat	m_mhi;			//�˶���ʷͼ��(Motion History Image)
	cv::Mat	m_orient;		//�ݶȷ���Ƕ�ͼ��
	cv::Mat	m_mask;		//�ݶ���Ч����ͼ��
	cv::Mat	m_segmask;	//�˶�����ͼ��

	int m_last;		//ͼ��ʼλ��
	std::vector<cv::Rect> m_storage;	//�˶����λ��
};