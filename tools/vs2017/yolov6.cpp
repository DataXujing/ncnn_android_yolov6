#include "yolo.h"

//#include "layer.h"
#include "net.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

int main()
{

	cv::Mat m = cv::imread("./test.jpg", 1);
	cv::Mat m1;
	m.copyTo(m1);

	Yolo * yolov6 = new Yolo();


	int target_size = 640;
	float norm_vals[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };

	yolov6->load(target_size, norm_vals);

	std::vector<Object> objects;

	const float prob_threshold = 0.25f;
	const float nms_threshold = 0.45f;

	yolov6->detect(m, objects, prob_threshold, nms_threshold);

	yolov6->draw(m1, objects);

	cv::imshow("image", m1);
	cv::waitKey();

	return 0;
}