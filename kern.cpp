#include <iostream>
#include <string>
#include "device_launch_parameters.h"
#include <iomanip>
#include <memory>
#include <cstdio>
#include <stdio.h>
#include "mpi.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <chrono>


#define BLOCK_SIZE 16
#define CHANNELS 3

int main(int argc, char** argv)
{
	auto start_time = std::chrono::high_resolution_clock::now();
	int rank, size;
	cv::Mat full_image;
	int image_properties[4];

	// initiate MPI
	MPI_Init(&argc, &argv);
	// get the size and rank
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);


	// load the image ONLY in the master process #0:
	if (rank == 0) {
		full_image = cv::imread("C:\\BA_Tests\\PKA_img\\5.jpg", cv::IMREAD_UNCHANGED);
		if (full_image.empty()) {
			std::cout << "!!! Failed imread(): image not found" << std::endl;
			return 1;
		}
		// get the properties of the image, to send to other processes later:
		image_properties[0] = full_image.cols; // width
		image_properties[1] = full_image.rows / size; // height, divide it by number of processes
		image_properties[2] = full_image.type(); // image type (in this case: CV_8UC3 - 3 channel, 8 bit )
		image_properties[3] = full_image.channels(); // number of channels (here: 3)
	}

	//Pause all threads, until the master process is done.
	MPI_Barrier(MPI_COMM_WORLD);
	//Master thread sends the image properties to the other threads
	MPI_Bcast(image_properties, 4, MPI_INT, 0, MPI_COMM_WORLD);
	//Each thread sets the properties of their part of the image
	cv::Mat part_image = cv::Mat(image_properties[1], image_properties[0], image_properties[2]);

	MPI_Barrier(MPI_COMM_WORLD);
	int send_size = image_properties[1] * image_properties[0] * image_properties[3];
	//Master thread scatters the data of the image to the other threads
	MPI_Scatter(full_image.data, send_size, MPI_UNSIGNED_CHAR, part_image.data, send_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD); // from process #0
	
	cv::Mat final_image(full_image.rows, full_image.cols, CV_8UC3);
	cv::Mat outputImage, tempImage, embossImage;
	
	auto start_timeg = std::chrono::high_resolution_clock::now();

	cv::Mat filter = cv::Mat(3, 3, CV_8S);
	filter.at<schar>(0, 0) = -2;
	filter.at<schar>(0, 1) = -1;
	filter.at<schar>(0, 2) =  0;
	filter.at<schar>(1, 0) = -1;
	filter.at<schar>(1, 1) = 1;
	filter.at<schar>(1, 2) = 1;
	filter.at<schar>(2, 0) = 0;
	filter.at<schar>(2, 1) = 1;
	filter.at<schar>(2, 2) = 2;

	cv::cvtColor(part_image, tempImage, cv::COLOR_BGR2GRAY);
	tempImage = 3 * (tempImage -128) + 128;
	cv::filter2D(tempImage, embossImage, -1, filter);

	cv::cvtColor(embossImage, outputImage, cv::COLOR_GRAY2BGR);


	//Gather the parts of the image processed in each mpi thread
	MPI_Gather(outputImage.data, send_size, MPI_UNSIGNED_CHAR,
		final_image.data, send_size, MPI_UNSIGNED_CHAR,
		0, MPI_COMM_WORLD);
	//Master thread saves the final image
	if (rank == 0) {
		auto end_timeg = std::chrono::high_resolution_clock::now();
		auto timeg = end_timeg - start_timeg;
		std::cout << "Process #0 received the gathered image" << std::endl;
		cv::imshow("gathered image", final_image);
		cv::waitKey(0); // will need to press a key in EACH process...
		cv::destroyAllWindows();
		cv::imwrite("imageFinalEmboss2.jpg", final_image);
		std::cout << std::endl << std::endl << std::endl << std::endl <<  "took " << timeg / std::chrono::milliseconds(1) << "ms to run modification.\n" << std::endl << std::endl << std::endl;
	}
	
	//Finalize MPI
	MPI_Finalize();

}