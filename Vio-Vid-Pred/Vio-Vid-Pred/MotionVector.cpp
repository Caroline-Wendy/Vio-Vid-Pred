#include "stdafx.h"

#include "MotionVector.h"

using namespace std;
using namespace cv;

MotionVector::MotionVector(void){
}

MotionVector::MotionVector(double minDist, int maxCount, double qualityLevel){

	const unsigned int buffer_length = 2;

	m_buffer_length = buffer_length;
	m_maxCount = maxCount;					// corners的最大个数
	m_qualityLevel = qualityLevel;			// 特征检测的等级
	m_minDist = minDist;				//两特征点之间的最小距离

	m_buffer.clear();

	m_motion_num = 10;

	m_last = 0; //图像位置
}

void  MotionVector::cal_motion_vector(const cv::Mat& src, std::vector<double>& motion_vector, bool isPicture){

	if(src.empty()){
		std::cerr << "Sorry, the source image is empty! " << std::endl;
		return;
	}
	src.copyTo(m_src);

	/*初始化缓存*/
	if(m_buffer.empty()){

		for(unsigned int i=0; i<m_buffer_length; i++){
			m_buffer.push_back(cv::Mat::zeros(src.rows, src.cols, CV_32FC1));
		}

		/*初始化第一个图像*/
		cv::cvtColor(m_src, m_buffer[m_last], CV_BGR2GRAY);
		m_last++;

		return;
	}

	/*计算缓存大小*/
	unsigned int buffer_length = m_buffer.size();
	if(buffer_length < 2){
		std::cerr << "Sorry, the buffer length should be more than 2! " << std::endl;
		return;
	}

	int idx1 = m_last;	//last帧索引
	int idx2 = (m_last + 1) % buffer_length; //last下一帧索引

	/*转换当前图像为灰度图像*/
	cv::cvtColor(m_src, m_buffer[m_last], CV_BGR2GRAY);

	m_last = idx2;

	if(m_buffer[m_last].empty() || m_buffer[idx2].empty()){
		return;
	}

	vector<Point2f> points_old;
	vector<Point2f> points_new;

	motion_tracking(m_buffer[idx2], m_buffer[idx1], points_old, points_new);

	/*返回运动向量值*/
	double temp_x(0.0);
	double temp_y(0.0);
	double temp_result(0.0);

	for(unsigned int i=0; i<points_new.size(); i++){
		temp_x = points_new[i].x-points_old[i].x;
		temp_y = points_new[i].y-points_old[i].y;
		temp_result = pow(temp_x, 2.0)+pow(temp_y, 2.0);
		motion_vector.push_back(temp_result);
	}

	/*画图*/
	if(isPicture){
		for (size_t i=0; i < points_new.size(); i++){
			cv::line(m_src, points_old[i], points_new[i], cv::Scalar(0, 255, 255), 3);
			cv::circle(m_src, points_new[i], 3, Scalar(255, 0, 0), -1);
			cv::circle(m_src, points_old[i], 3, Scalar(0, 255, 0), -1);
		}
		cv::imshow( "Motion Vector", m_src);
		if( cv::waitKey(1) >= 0 )
			return;
	}
}

void MotionVector::motion_tracking(cv::Mat& pre_frame, cv::Mat& next_frame,
	vector<Point2f>& points_old, vector<Point2f>& points_new){

		cv::Size frame_size = pre_frame.size();
		cv::resize(next_frame, next_frame, frame_size, 0.0, 0.0, INTER_CUBIC);

		points_old.clear();
		points_new.clear();

		cv::Mat g_pre_frame;
		cv::Mat g_next_frame;

		if(pre_frame.channels() != 1){
			cv::cvtColor(pre_frame, g_pre_frame, CV_BGR2GRAY);
		}else{
			g_pre_frame = pre_frame;
		}

		if(next_frame.channels() != 1){
			cv::cvtColor(next_frame, g_next_frame, CV_BGR2GRAY);
		}else{
			g_next_frame = next_frame;
		}

		std::vector<cv::Point2f> points_old_temp;
		std::vector<cv::Point2f> points_new_temp;

		cv::goodFeaturesToTrack(g_pre_frame, points_old_temp, m_maxCount, m_qualityLevel, m_minDist);

		cv::Size winSize(5, 5);
		cv::Size zeroZone(-1,-1);
		TermCriteria criteria = TermCriteria( TermCriteria::EPS + TermCriteria::COUNT, 40, 0.001 );
		cv::cornerSubPix(g_pre_frame, points_old_temp, winSize, zeroZone, criteria);

		vector<uchar> status;		//状态
		vector<float> err;
		cv::calcOpticalFlowPyrLK(g_pre_frame, g_next_frame, points_old_temp, points_new_temp, status, err);

		/*忽略跟踪失败的点*/
		for(unsigned i=0; i<err.size(); i++){
			if(err[i] > 1.0 && status[i]){

				/*忽略距离小的点*/
				double temp_x = points_new_temp[i].x-points_old_temp[i].x;
				double temp_y = points_new_temp[i].y-points_old_temp[i].y;
				double temp_result = pow(temp_x, 2.0)+pow(temp_y, 2.0);
				if(sqrt(temp_result) > m_minDist){

					/*忽略镜头切换*/
					if(abs(static_cast<int>(m_motion_num-points_new_temp.size())) < m_motion_num/4){
						points_new.push_back(points_new_temp[i]);
						points_old.push_back(points_old_temp[i]);
					}
					m_motion_num = points_new_temp.size();
				}
			}
		}
}