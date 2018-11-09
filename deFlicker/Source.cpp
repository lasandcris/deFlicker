#include "opencv2\highgui.hpp"
#include "opencv2\imgproc.hpp"
#include "opencv2\objdetect\objdetect.hpp"
#include "opencv2/video/tracking.hpp"
#include <vector>
#include <stdio.h>
#include <Windows.h>
#include <iostream>
#include <ctime>

using namespace std;
using namespace cv;

string type2str(int type) {
	string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}

vector<string> get_all_files_full_path_within_folder(string folder)
{
	vector<string> names;
	char search_path[200];
	sprintf(search_path, "%s*.*", folder.c_str());
	WIN32_FIND_DATA fd;
	HANDLE hFind = ::FindFirstFile(search_path, &fd);
	if (hFind != INVALID_HANDLE_VALUE)
	{
		do
		{
			// read all (real) files in current folder, delete '!' read other 2 default folder . and ..
			if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
			{

				names.push_back(folder + fd.cFileName);
			}
		} while (::FindNextFile(hFind, &fd));
		::FindClose(hFind);
	}
	return names;
}

String SplitFilename(const string& str)
{
	size_t found;
	found = str.find_last_of("/\\");
	return str.substr(0, found);
}
String fname(const string& str)
{
	size_t found;
	found = str.find_last_of("/\\");
	return str.substr(found+1);
}

int main(int argc, char* argv[])
{
	/*cv::Mat res1, res2, res3;
	cv::Mat img1, img2, img3, img4;
	cv::Mat imgDM1, imgDM2, imgDM3, imgDM4;
	cv::Mat img2Original, img2OriginalC, img1OriginalC;*/
	cv::Mat imgAve;
	vector<string> allFileNames;
	vector<string> allFileNamesDepth;

	//get rgb images
	string filename = argv[1];
	string directory = SplitFilename(filename);
	directory += "/";
	allFileNames = get_all_files_full_path_within_folder(directory);


	//get depth maps
	string filenameDepth = argv[2];
	string directoryDepth = SplitFilename(filenameDepth);
	directoryDepth += "/";
	allFileNamesDepth = get_all_files_full_path_within_folder(directoryDepth);

	std::cout << "Number of RGB images: " << allFileNames.size() << "\n";
	std::cout << "Number of Depth Map images: " << allFileNamesDepth.size() << "\n";

	if (allFileNames.size() != allFileNamesDepth.size()) {
		std::cout << "the frame numbers are different in the RGB and DepthMap directories" << "\n";
		return 0;
	}

	int nFrame = 3;

	for (int i = nFrame; i < allFileNames.size(); i++)
	{
		// timer for debug
		std::clock_t start;
		double duration;
		start = std::clock();

		int original_x = 0;
		int original_y = 0;

		vector<Mat> rgbimgs; vector<Mat> depthimgs; vector<Mat> res;

		for (int n = 0; n <= nFrame; n++) {
			Mat rgb_a = imread(allFileNames[i - n]);
			original_x = rgb_a.cols;
			original_y = rgb_a.rows;
			Mat rgb;
			cv::resize(rgb_a, rgb, cv::Size(original_x*0.5, original_y*0.5), 0, 0, CV_INTER_LINEAR);
			cvtColor(rgb, rgb, COLOR_BGR2GRAY);
			rgbimgs.push_back(rgb);
			Mat depth_a = imread(allFileNamesDepth[i - n]);
			Mat depth_b;
			cv::resize(depth_a, depth_b, cv::Size(original_x*0.5, original_y*0.5), 0, 0, CV_INTER_LINEAR);
			depthimgs.push_back(depth_b);

			if (depth_a.cols != rgb_a.cols || depth_a.rows != rgb_a.rows) {
				std::cout << "The frame sizes are different between the rgb and depth map images: " << "\n";
				std::cout << "Size of " + allFileNames[i - n] + ": " << rgb_a.size() << "\n";
				std::cout << "Size of " + allFileNamesDepth[i - n] + ": " << depth_a.size() << "\n";
				return 0;
			}
		}

		depthimgs[0].copyTo(imgAve);

		for (int n = 0; n < nFrame; n++) {
			Mat outres;
			cv::calcOpticalFlowFarneback(rgbimgs[n], rgbimgs[n + 1], outres, .5, 2, 400, 10, 7, 1.5, OPTFLOW_USE_INITIAL_FLOW);
			res.push_back(outres);
		}

		// current file name
		string currentFName = fname(allFileNames[i]);
		size_t findchar = currentFName.find_last_of(".");
		// current file number
		string getFNumber = currentFName.substr(0, findchar);
		findchar = getFNumber.find_last_of(".");
		// current raw file name
		string rawfilename = getFNumber.substr(0, findchar);
		// current raw file number
		string rawfilenumber = getFNumber.substr(findchar + 1);
		//std::cout << rawfilenumber << "\n";
		
		for (int y = 0; y < rgbimgs[0].rows; y++)
		{
			for (int x = 0; x < rgbimgs[0].cols; x++)
			{
				vector<Point2f> flowxy;
				for (int s = 0; s < nFrame; s++)
					flowxy.push_back(res[s].at<Point2f>(y, x) * 1);

				bool checkboundry = true;
				for (int s = 0; s < nFrame; s++)
				{
					if (y + flowxy[s].y > 0 && y + flowxy[s].y < rgbimgs[0].rows && x + flowxy[s].x > 0 && x + flowxy[s].x < rgbimgs[0].cols)
						continue;	
					else
					{
						checkboundry = false;
						break;
					}
				}

				if (checkboundry == true) {
					vector<float> pixels;
					pixels.push_back(depthimgs[0].at<cv::Vec3b>(y, x)[0]);

					for (int s = 1; s < nFrame + 1; s++)
						pixels.push_back(depthimgs[s].at<cv::Vec3b>(y + flowxy[s - 1].y, x + flowxy[s - 1].x)[0]);

					float mean = 0;
					for (int m = 0; m < pixels.size(); m++)
						mean += pixels[m];
					mean /= pixels.size();

					float stdDev = 0;
					for (int m = 0; m < pixels.size(); m++)
						stdDev += pow(abs(pixels[m] - mean), 2.0);
					stdDev = sqrt(stdDev / pixels.size());

					float newmean = 0;
					int nPixels = 0;
					float h_mean = mean + stdDev;
					float l_mean = mean - stdDev;
					for (int m = 0; m < pixels.size(); m++)
					{
						if (pixels[m] <= h_mean && pixels[m] >= l_mean)
						{
							newmean += pixels[m];
							nPixels += 1;
						}
					}
					newmean /= nPixels;

					imgAve.at<cv::Vec3b>(y, x) = cv::Vec3b(newmean, newmean, newmean);
				}
			}
		}
		//timer for debug
		duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
		std::cout << "printf: " << duration << '\n';

		//get high frequency
		Mat rgb_canny;
		blur(rgbimgs[0], rgb_canny, Size(3, 3));
		Canny(rgb_canny, rgb_canny, 10, 30, 3);

		// dilate result
		int dilation_size = 9;
		Mat element = getStructuringElement(MORPH_RECT, Size(2 * dilation_size + 1, 2 * dilation_size + 1), Point(dilation_size, dilation_size));
		dilate(rgb_canny, rgb_canny, element);
		
		// smooth and invert it
		GaussianBlur(rgb_canny, rgb_canny, Size(251, 251), 0, 0);
		bitwise_not(rgb_canny, rgb_canny);
		cvtColor(rgb_canny, rgb_canny, COLOR_GRAY2BGR);

		Mat depth_blured;
		GaussianBlur(imgAve, depth_blured, Size(25,25),0,0);

		// prepare images for multiplication
		imgAve.convertTo(imgAve, CV_32F);
		cvtColor(imgAve, imgAve, COLOR_BGR2GRAY);
		rgb_canny.convertTo(rgb_canny, CV_32F);
		cvtColor(rgb_canny, rgb_canny, COLOR_BGR2GRAY);
		depth_blured.convertTo(depth_blured, CV_32F); 
		cvtColor(depth_blured, depth_blured, COLOR_BGR2GRAY);

		imgAve /= 255.0; rgb_canny /= 255.0; depth_blured /= 255.0;

		Mat blended, blended2;
		multiply(imgAve, 1.0 - rgb_canny, blended);
		multiply(depth_blured, rgb_canny, blended2);

		std::vector<cv::Mat> imagesbgr(3);
		imagesbgr.at(0) = (blended + blended2) * 255; //for blue channel
		imagesbgr.at(1) = (blended + blended2) * 255;   //for green channel
		imagesbgr.at(2) = (blended + blended2) * 255; //for red channel

		Mat finalBlend;
		cv::merge(imagesbgr, finalBlend);

		cv::resize(finalBlend, finalBlend, cv::Size(original_x, original_y), 0, 0, CV_INTER_LINEAR);
	
		std::cout << "writing: " << rawfilename + "." + rawfilenumber + ".png" << "\n";
		imwrite(rawfilename + "." + rawfilenumber + ".png", finalBlend);
	}
	return 0;
}