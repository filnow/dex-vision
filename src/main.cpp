#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

int main()
{
    cv::Mat img = cv::imread("/home/filnow/fun/dex-vision/image1.jpg");
    cv::imshow("IMAGE", img);
    cv::waitKey(0);
    return 0;
}