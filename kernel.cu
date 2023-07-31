#include <iostream>
#include <string>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iomanip>
#include <memory>
#include <cstdio>
#include <stdio.h>
#include "mpi.h"
#include <opencv2/opencv.hpp>
#include <chrono>

#define BLOCK_SIZE 16
#define CHANNELS 3

/// <summary>
/// Clamp value to an acceptable value for a color. Between 0 and 255
/// </summary>
__device__ float clamp(float value) {
	float oVal = value;
	if (oVal < 0) return 0;
	if (oVal > 255) return 255;
	return oVal;
}

/// <summary>
/// Kernel to turn an BRG image into a grayscale image.
/// The output image's data has only one channel.
/// </summary>
/// <param name="imageData"> The data from the BRG image's pixels</param>
__global__ void grayscaleKernel(unsigned char* imageData, int width, int height, int channels, unsigned char* outputImage)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < width && y < height)
	{
		int pixelIndex = (y * width + x);
		int pixelBRG = pixelIndex * channels;
		outputImage[pixelIndex] =
			(unsigned char)((float)imageData[pixelBRG] * 0.11) + //b
			(unsigned char)((float)imageData[pixelBRG + 1] * 0.59) + //g
			(unsigned char)((float)imageData[pixelBRG + 2] * 0.3);  //r
	}
}

/// <summary>
/// Kernel to modify the contrast of a grayscale image.
/// </summary>
__global__ void contrastKernel(int width, int height, unsigned char* outputImage) {
	float contrast = 3.5;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < width && y < height)
	{
		int pixelIndex = y * width + x;
		outputImage[pixelIndex] = clamp(contrast * (outputImage[pixelIndex] - 128) + 128);
	}
}

/// <summary>
/// Kernel to add an emboss filter to a grayscale image
/// </summary>
/// <param name="smallEmboss"> There are two filters, this boolean decides which filter will be used</param>
__global__ void embossKernel(int width, int height, unsigned char* outputImage, bool smallEmboss) {
	float emboss = 0;
	int filterSize = smallEmboss ? 3 : 5;
	int offset = (filterSize - 1) / 2;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int pixelIndex = y * width + x;;
	int filter[3][3] = {
	{-2,-1,0},
	{-1,1,1},
	{0,1,2}
	};
	int filter2[5][5] = {
		{4 ,0    ,0  ,0      ,0  },
		{0    ,4 ,0  ,0      ,0  },
		{0    ,0    ,1  ,0      ,0  },
		{0    ,0    ,0  ,-4  ,0  },
		{0    ,0    ,0  ,0      ,-4  }
	};
	if (x <= width - offset && y <= height - offset && x > offset && y > offset) {
		for (int fx = 0; fx < filterSize; fx++) {
			for (int fy = 0; fy < filterSize; fy++) {
				int index = pixelIndex + (fx - offset) + (width * (fy - offset));
				if (filterSize == 5)emboss += outputImage[index] * filter2[fx][fy];
				if (filterSize == 3)emboss += outputImage[index] * filter[fx][fy];
			}
		}
		outputImage[pixelIndex] = clamp(emboss);
	}

}

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
		full_image = cv::imread("C:\\BA_Tests\\PKA_img\\3.jpg", cv::IMREAD_UNCHANGED);
		if (full_image.empty()) {
			std::cout << "!!! Failed imread(): image not found" << std::endl;
			return 1;
		}
		// get the properties of the image, to send to other processes later:
		image_properties[0] = full_image.cols; // width
		image_properties[1] = full_image.rows / size; // height, divide it by number of processes
		image_properties[2] = full_image.type(); // image type (in this case: CV_8UC3 - 3 channel, 8 bit )
		image_properties[3] = full_image.channels(); // number of channels (here: 3)
		cv::imshow("output image" + std::to_string(rank), full_image);
		cv::waitKey(0);
		cv::destroyAllWindows();
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
	cv::Mat embossImage;
	
	size_t partImageSize = part_image.total() * part_image.channels() * sizeof(uchar);
	// Start CUDA preparations
	//We use cuda status, to check if everything is running correctly, after each CUDA operation.
	cudaError_t cudaStatus;
	//Set GPU device, that will be called by the host thread
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!");
		return 1;
	}


	uchar* dev_part_image = new uchar[partImageSize];
	//Allocate memory in GPU for the original image
	cudaStatus = cudaMalloc(&dev_part_image, partImageSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return 1;
	}

	//Copy data from host to gpu
	cudaStatus = cudaMemcpy(dev_part_image, part_image.data, partImageSize, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy 1 failed!");
		return 1;
	}

	cv::Mat grayImage(image_properties[1], image_properties[0], CV_8U);
	int grayImageSize = partImageSize / part_image.channels();
	uchar* dev_grayImage = new uchar[grayImageSize];
	//Allocate a memory equal to a third of the original image, as the number of channels changes from 3 to 1
	cudaStatus = cudaMalloc(&dev_grayImage, grayImageSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return 1;
	}

	int3 imageSize = make_int3(part_image.cols, part_image.rows, part_image.channels());
	// Size of each block in x,y,z
	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	// Number of blocks needed in x,y,z
	dim3 numBlocks(std::ceil(static_cast<double>(part_image.cols) / threadsPerBlock.x),
		std::ceil(static_cast<double>(part_image.rows) / threadsPerBlock.y));
	//Output for information about image, blocks and threads
	std::cout << "Image properties: W=" << imageSize.x << "   H=" << imageSize.y << "   C=" << imageSize.z << "  \n";
	std::cout << "Block size:       X=" << threadsPerBlock.x << "   Y=" << threadsPerBlock.y << "   Z=" << threadsPerBlock.z << "  \n";
	std::cout << "Number of blocks: X=" << numBlocks.x << "   Y=" << numBlocks.y << "   Z=" << numBlocks.z << "  \n";

	auto start_timeg = std::chrono::high_resolution_clock::now();

	grayscaleKernel << < numBlocks, threadsPerBlock >> > 
		(dev_part_image, imageSize.x, imageSize.y, part_image.channels(), dev_grayImage);
	contrastKernel << < numBlocks, threadsPerBlock >> > (imageSize.x, imageSize.y, dev_grayImage);
	embossKernel << < numBlocks, threadsPerBlock >> > (imageSize.x, imageSize.y, dev_grayImage, true);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize failed!");
		return 1;
	}
	//copy the data of the processed image, from the gpu, to the host
	cudaStatus = cudaMemcpy(grayImage.data, dev_grayImage, partImageSize / part_image.channels(), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy 2 failed!");
		return 1;
	}



	cv::cvtColor(grayImage, embossImage, cv::COLOR_GRAY2BGR);



	//cv::imwrite("image_slice_" + std::to_string(rank) + "Output.jpg", embossImage);
	//cv::imshow("input image" + std::to_string(rank), part_image);
	//cv::imshow("output image" + std::to_string(rank), embossImage);
	//cv::waitKey(0);
	//cv::destroyAllWindows();


	if (cudaStatus == cudaSuccess) {
		//Gather the parts of the image processed in each mpi thread
		MPI_Gather(embossImage.data, send_size , MPI_UNSIGNED_CHAR,
			final_image.data, send_size, MPI_UNSIGNED_CHAR,
			0, MPI_COMM_WORLD);
		auto end_timeg = std::chrono::high_resolution_clock::now();
		auto timeg = end_timeg - start_timeg;
		//Master thread saves the final image
		if (rank == 0) {
			std::cout << "Process #0 received the gathered image" << std::endl;
			std::cout << std::endl << std::endl << std::endl << std::endl << "______took " << timeg / std::chrono::nanoseconds(1) << "ns to run modification.\n" ;
			std::cout << "______took " << timeg / std::chrono::milliseconds(1) << "ms to run modification.\n" << std::endl << std::endl << std::endl;
			cv::imshow("gathered image", final_image);
			cv::waitKey(0); // will need to press a key in EACH process...
			cv::destroyAllWindows();
			cv::imwrite("imageFinalEmboss2.jpg", final_image);
		}

	}
	// cudaDeviceReset must be called before exiting
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	//Setting free the memory used for these two variables
	cudaFree(dev_part_image);
	cudaFree(dev_grayImage);
	//Finalize MPI
	MPI_Finalize();

	return cudaStatus;

}