#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<math.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include<time.h>
#include<sys/time.h>

#define SQUARE(x) ((x)*(x))
#define PI 3.14
#define BLOCK_SIZE 16
#define N 4096

__device__ __constant__ float pi=3.14;
__device__ __constant__ int sobelX[9]={1,0,-1,2,0,-2,1,0,-1};
__device__ __constant__ int sobelY[9]={1,2,1,0,0,0,-1,-2,-1};


__global__ void NormalizeGrayGPU(double input[], int width, int height, unsigned char output[], double min, double max)
{
	int index= blockIdx.x*blockDim.x+threadIdx.x;
	
	if(index < (width*height))
	{
		output[index]=(input[index]-min)*255/(max-min);
	}
}

double FindMin(double input[], int width, int height)
{
	double min = input[0];
	
	for (int i = 0; i < width*height; i++)
	{
		if (input[i] < min) min = input[i];
	}
	return min;
}
double FindMax(double input[], int width, int height)
{
	double max = input[0];
	
	for (int i = 0; i < width*height; i++)
	{
		if (input[i] > max) max = input[i];
	}
	return max;
}

__global__ void SobelFilter_gpu(unsigned char* A, double *gradImageX, double *gradImageY, double *gradMag, int width, int height)
{
	

	int row= blockIdx.y*blockDim.y+threadIdx.y;
	int col= blockIdx.x*blockDim.x+threadIdx.x;
	double tempx=0;
	double tempy=0;	
		if(row < height && col < width){


			tempx = 0;
			tempy = 0;
			for (int r2=-1; r2<=1; r2++){
				for (int c2=-1; c2<=1; c2++)
				{	
					tempx += A[(row+r2)*width+(col+c2)]*sobelX[(r2+1)*3+c2+1];
					tempy += A[(row+r2)*width+(col+c2)]*sobelY[(r2+1)*3+c2+1];
				}
			}


			gradImageX[(row*width)+col]=tempx;
			gradImageY[(row*width)+col]=tempy;
			gradMag[(row*width)+col]= sqrt((double) (tempx*tempx)+(tempy*tempy));

	
		}
		
}
		


__global__ void theta_gpu(double *gradImageY, double *gradImageX, double *gradPhase, int width, int height){
	
	int index= blockIdx.x*blockDim.x+threadIdx.x;
	if(index<(width*height)){
		float theta = atan2(gradImageY[index],gradImageX[index]);
		theta=theta*180/pi;
		gradPhase[index]=theta;
	}

}




int main(int argc, char *argv[])
{

	FILE *fptr;
	char *inputHeader, *testHeader;
	int inputCols, inputRows, inputBytes;
	int testCols, testRows, testBytes;
	char Header_1[320], Header_2[320];
	unsigned char *inputImage, *testImage;
	unsigned char *normalGradMag, *normalGrad_x, *normalGrad_y, *normalGradPhase;
	unsigned char *normaltestMag, *normaltest_x, *normaltest_y, *normaltestPhase;
	double *gradPhase, *gradMag;
	double *testgradPhase, *testgradMag;
	double max=0;
	double min=0;
	float gpu_time_1 = 0;
	float gpu_time_2 = 0;
	float gpu_time_3 = 0;

	//GPU variables
	double *d_gradImageX, *d_gradImageY, *d_gradPhase, *d_gradMag;
	unsigned char *d_inputImage, *d_normalGradMag, *d_normalGradX, *d_normalGradY, *d_normalGradPhase;
	unsigned char *d_testImage;
	double *d_testgradImageX, *d_testgradImageY, *d_testgradMag, *d_testgradPhase;
	unsigned char *d_testnormalGradMag, *d_testnormalGradX, *d_testnormalGradY, *d_testnormalGradPhase;
	cudaError_t err;
	struct timeval cstart1, cstart2, cstart3, cend1, cend2, cend3;
	cudaEvent_t start1, start2, start3, stop1, stop2, stop3;
	
	printf("Initialization done!\n");
	
	gettimeofday(&cstart1, NULL);
	if ((fptr=fopen(argv[1],"r"))==NULL)
	{
		printf("Unable to open input file for reading\n");
		exit(0);
	}

	//Open and load input image
	fptr = fopen(argv[1], "r");
	fscanf(fptr,"%s %d %d %d",&inputHeader, &inputCols, &inputRows, &inputBytes);	
	Header_1[0]=fgetc(fptr);	/* read white-space character that separates header */
	inputImage = (unsigned char*)calloc(inputCols*inputRows,sizeof(unsigned char));
	fread(inputImage, 1, inputCols*inputRows, fptr);
	fclose(fptr);
	printf("Input file opened!\n");

	if ((fptr = fopen(argv[2], "r")) == NULL)
	{
		printf("Unable to open test file for reading\n");
		exit(0);
	}
	//Open and load test image
	fptr = fopen(argv[2], "rb");
	fscanf(fptr, "%s %d %d %d", &testHeader, &testCols, &testRows, &testBytes);
	Header_2[0] = fgetc(fptr);	/* read white-space character that separates header */
	testImage = (unsigned char*)calloc(testCols*testRows, sizeof(unsigned char));
	fread(testImage, 1, testCols*testRows, fptr);
	fclose(fptr);
	printf("Test file opened!\n");

	gettimeofday(&cend1, NULL);
	

	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);

	cudaEventRecord(start1);
	cudaEventSynchronize(start1);

	//cudaMalloc for Input image
	err=cudaMalloc(&d_inputImage,(inputRows*inputCols*sizeof(unsigned char)));
	if(err != cudaSuccess) printf("/n Error in cudaMalloc d_inputImage");

	err=cudaMalloc(&d_gradImageX,(inputRows*inputCols*sizeof(double)));
	if(err != cudaSuccess) printf("/n Error in cudaMalloc d_gradImageX");

	err=cudaMalloc(&d_gradImageY,(inputRows*inputCols*sizeof(double)));
	if(err != cudaSuccess) printf("/n Error in cudaMalloc d_gradImageY");

	err=cudaMalloc(&d_gradPhase,(inputRows*inputCols*sizeof(double)));
	if(err != cudaSuccess) printf("/n Error in cudaMalloc d_gradPhase");

	err=cudaMalloc(&d_gradMag,(inputRows*inputCols*sizeof(double)));
	if(err != cudaSuccess) printf("/n Error in cudaMalloc d_gradMag");

	err=cudaMalloc(&d_normalGradMag,(inputRows*inputCols*sizeof(unsigned char)));
	if(err != cudaSuccess) printf("/n Error in cudaMalloc d_normalGradMag");

	err=cudaMalloc(&d_normalGradX,(inputRows*inputCols*sizeof(unsigned char)));
	if(err != cudaSuccess) printf("/n Error in cudaMalloc d_normalGradX");
	
	err=cudaMalloc(&d_normalGradY,(inputRows*inputCols*sizeof(unsigned char)));
	if(err != cudaSuccess) printf("/n Error in cudaMalloc d_normalGradY");
	
	err=cudaMalloc(&d_normalGradPhase,(inputRows*inputCols*sizeof(unsigned char)));
	if(err != cudaSuccess) printf("/n Error in cudaMalloc d_normalGradPhase");

	//cudaMalloc for test image
	err=cudaMalloc(&d_testImage,(testRows*testCols*sizeof(unsigned char)));
	if(err != cudaSuccess) printf("/n Error in cudaMalloc d_testImage");

	err=cudaMalloc(&d_testgradImageX,(testRows*testCols*sizeof(double)));
	if(err != cudaSuccess) printf("/n Error in cudaMalloc d_testgradImageX");

	err=cudaMalloc(&d_testgradImageY,(testRows*testCols*sizeof(double)));
	if(err != cudaSuccess) printf("/n Error in cudaMalloc d_testgradImageY");

	err=cudaMalloc(&d_testgradPhase,(testRows*testCols*sizeof(double)));
	if(err != cudaSuccess) printf("/n Error in cudaMalloc d_testgradPhase");

	err=cudaMalloc(&d_testgradMag,(testRows*testCols*sizeof(double)));
	if(err != cudaSuccess) printf("/n Error in cudaMalloc d_testgradMag");

	err=cudaMalloc(&d_testnormalGradMag,(testRows*testCols*sizeof(unsigned char)));
	if(err != cudaSuccess) printf("/n Error in cudaMalloc d_testnormalGradMag");

	err=cudaMalloc(&d_testnormalGradX,(testRows*testCols*sizeof(unsigned char)));
	if(err != cudaSuccess) printf("/n Error in cudaMalloc d_testnormalGradX");
	
	err=cudaMalloc(&d_testnormalGradY,(testRows*testCols*sizeof(unsigned char)));
	if(err != cudaSuccess) printf("/n Error in cudaMalloc d_testnormalGradY");
	
	err=cudaMalloc(&d_testnormalGradPhase,(testRows*testCols*sizeof(unsigned char)));
	if(err != cudaSuccess) printf("/n Error in cudaMalloc d_testnormalGradPhase");
	
	cudaEventSynchronize(stop1);
	cudaEventRecord(stop1);

	gettimeofday(&cstart2, NULL);
	//Normalized input gradient images
	normalGradMag = (unsigned char*)calloc(inputCols*inputRows, sizeof(unsigned char));
	normalGrad_x = (unsigned char*)calloc(inputCols*inputRows, sizeof(unsigned char));
	normalGrad_y = (unsigned char*)calloc(inputCols*inputRows, sizeof(unsigned char));
	normalGradPhase = (unsigned char*)calloc(inputCols*inputRows, sizeof(unsigned char));
	gradPhase = (double*)calloc(inputCols*inputRows, sizeof(double));
	gradMag = (double*)calloc(inputCols*inputRows, sizeof(double));
	
	//Normalized test gradient images
	normaltestMag = (unsigned char*)calloc(testCols*testRows, sizeof(unsigned char));
	normaltest_x = (unsigned char*)calloc(testCols*testRows, sizeof(unsigned char));
	normaltest_y = (unsigned char*)calloc(testCols*testRows, sizeof(unsigned char));
	normaltestPhase = (unsigned char*)calloc(testCols*testRows, sizeof(unsigned char));
	testgradPhase = (double*)calloc(testCols*testRows, sizeof(double));
	testgradMag = (double*)calloc(testCols*testRows, sizeof(double));
	
	gettimeofday(&cend2, NULL);	
	
	cudaEventCreate(&start2);
	cudaEventCreate(&stop2);

	cudaEventRecord(start2);
	cudaEventSynchronize(start2);
	//Compute gradients and phase for input image
	err=cudaMemcpy(d_inputImage, inputImage, (inputRows*inputCols*sizeof(unsigned char)), cudaMemcpyHostToDevice);
	if(err != cudaSuccess) printf("/n Error in cudaMemcpy of d_inputImage");

	/* Launch Kernel*/
	 dim3 dimGrid(ceil((float)(N+2)/BLOCK_SIZE), ceil((float)(N+2)/BLOCK_SIZE),1);
	 dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	
	SobelFilter_gpu<<<dimGrid,dimBlock>>>(d_inputImage, d_gradImageX, d_gradImageY, d_gradMag, inputCols, inputRows);
	cudaDeviceSynchronize();

	dim3 BlockDim = dim3(1024,1,1);
	dim3 GridDim = dim3(10000,1,1);
	theta_gpu<<<GridDim,BlockDim>>>(d_gradImageY,d_gradImageX, d_gradPhase, inputCols, inputRows);


	//Compute gradients and phase for test image
	err=cudaMemcpy(d_testImage, testImage, (testRows*testCols*sizeof(unsigned char)), cudaMemcpyHostToDevice);
	if(err != cudaSuccess) printf("/n Error in cudaMemcpy of d_testImage");

	/* Launch Kernel*/
	SobelFilter_gpu<<<dimGrid,dimBlock>>>(d_testImage, d_testgradImageX, d_testgradImageY, d_testgradMag, testCols, testRows);
	cudaDeviceSynchronize();
	theta_gpu<<<GridDim,BlockDim>>>(d_testgradImageY,d_testgradImageX, d_testgradPhase, testCols, testRows);
	
	cudaMemcpy(gradMag, d_gradMag,(inputCols*inputRows*sizeof(double)),cudaMemcpyDeviceToHost);
	if(err != cudaSuccess) printf("/n Error in cudaMemcpy of normalGrad_x");
	min = FindMin(gradMag, inputCols, inputRows);
	max = FindMax(gradMag, inputCols, inputRows);
	NormalizeGrayGPU<<<GridDim,BlockDim>>>(d_gradMag, inputCols, inputRows, d_normalGradMag, min, max);
	cudaDeviceSynchronize();
	
	cudaMemcpy(testgradMag, d_testgradMag,(inputCols*inputRows*sizeof(double)),cudaMemcpyDeviceToHost);
	if(err != cudaSuccess) printf("/n Error in cudaMemcpy of normalGrad_x");
	min = FindMin(testgradMag, testCols, testRows);
	max = FindMax(testgradMag, testCols, testRows);
	NormalizeGrayGPU<<<GridDim,BlockDim>>>(d_testgradMag, testCols, testRows, d_testnormalGradMag, min, max);
	cudaDeviceSynchronize();

	cudaMemcpy(gradPhase, d_gradPhase,(inputCols*inputRows*sizeof(double)),cudaMemcpyDeviceToHost);
	if(err != cudaSuccess) printf("/n Error in cudaMemcpy of gradPhase");
	
	cudaMemcpy(testgradPhase, d_testgradPhase,(testCols*testRows*sizeof(double)),cudaMemcpyDeviceToHost);
	if(err != cudaSuccess) printf("/n Error in cudaMemcpy of testgradPhase");
	
	cudaMemcpy(normalGradMag, d_normalGradMag,(inputCols*inputRows*sizeof(unsigned char)),cudaMemcpyDeviceToHost);
	if(err != cudaSuccess) printf("/n Error in cudaMemcpy of normalGradMag");
	
	cudaMemcpy(normaltestMag, d_testnormalGradMag,(testCols*testRows*sizeof(unsigned char)),cudaMemcpyDeviceToHost);
	if(err != cudaSuccess) printf("/n Error in cudaMemcpy of normaltestMag");
	
	cudaEventRecord(stop2);
	cudaEventSynchronize(stop2);
	
	gettimeofday(&cstart3, NULL);	
	int histo[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	int testhisto[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	int difference[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };

	//Compute histogram of gradient orientations of input image
	double angle = 0;
	for (int i = 0; i < inputRows*inputCols; i++)
	{
		if (normalGradMag[i] > 25)
		{
			angle = fabs(gradPhase[i]);
			if (angle > 0 && angle < 21) histo[0]++;
			else if (angle > 21 && angle < 41) histo[1]++;
			else if (angle > 41 && angle < 61) histo[2]++;
			else if (angle > 61 && angle < 81) histo[3]++;
			else if (angle > 81 && angle < 101) histo[4]++;
			else if (angle > 101 && angle < 121) histo[5]++;
			else if (angle > 121 && angle < 141) histo[6]++;
			else if (angle > 141 && angle < 161) histo[7]++;
			else histo[8]++;
		}
	}

	printf("here6\n");
	//Compute histogram of gradient orientations of test image
	angle = 0;
	for (int i = 0; i < testRows*testCols; i++)
	{
		if (normaltestMag[i] > 25)
		{
			angle = fabs(testgradPhase[i]);
			if (angle > 0 && angle < 21) testhisto[0]++;
			else if (angle > 21 && angle < 41) testhisto[1]++;
			else if (angle > 41 && angle < 61) testhisto[2]++;
			else if (angle > 61 && angle < 81) testhisto[3]++;
			else if (angle > 81 && angle < 101) testhisto[4]++;
			else if (angle > 101 && angle < 121) testhisto[5]++;
			else if (angle > 121 && angle < 141) testhisto[6]++;
			else if (angle > 141 && angle < 161) testhisto[7]++;
			else testhisto[8]++;
		}
	}

	printf("here7\n");
	//Check the dissimilarity in histogram of gradient orientations
	int sumDiff = 0;
	for (int i = 0; i < 9; i++)
	{
		difference[i] = abs(histo[i] - testhisto[i]);
		printf("diff[%d] = %d\n", i, difference[i]);
		sumDiff += difference[i];
	}
	//float mismatch = (float)sumDiff*100/(testCols*testRows);
	printf("HOG mismatch = %d\n", sumDiff);
	

	fptr=fopen("input_grad_mag.pgm","w");
	fprintf(fptr,"P5 %d %d 255\n",inputCols,inputRows);
	fwrite(normalGradMag,inputCols*inputRows,1,fptr);
	fclose(fptr);

	fptr=fopen("test_grad_mag.pgm","w");
	fprintf(fptr,"P5 %d %d 255\n",testCols,testRows);
	fwrite(normaltestMag,testCols*testRows,1,fptr);
	fclose(fptr);
	
	//Free allocated memory
	free(normalGradMag);
	free(normalGradPhase);
	free(normalGrad_x);
	free(normalGrad_y);
	free(normaltestMag);
	free(normaltestPhase);
	free(normaltest_x);
	free(normaltest_y);
	
	gettimeofday(&cend3, NULL);
	
	//Free Allocated memory on the device. Don't forget. 
	cudaEventCreate(&start3);
	cudaEventCreate(&stop3);

	cudaEventRecord(start3);
	cudaEventSynchronize(start3);

	cudaFree(d_gradImageX);
	cudaFree(d_gradImageY);
	cudaFree(d_gradPhase);
	cudaFree(d_gradMag);
	cudaFree(d_inputImage);
	cudaFree(d_normalGradMag);
	cudaFree(d_normalGradX);
	cudaFree(d_normalGradY);
	cudaFree(d_normalGradPhase);	

	cudaFree(d_testgradImageX);
	cudaFree(d_testgradImageY);
	cudaFree(d_testgradPhase);
	cudaFree(d_testgradMag);
	cudaFree(d_testImage);
	cudaFree(d_testnormalGradMag);
	cudaFree(d_testnormalGradX);
	cudaFree(d_testnormalGradY);
	cudaFree(d_testnormalGradPhase);
	
	cudaEventRecord(stop3);
	cudaEventSynchronize(stop3);
	
	//Calculate time tiaken
//	float gpu_time_1 = 0;
//	float gpu_time_2 = 0;
//	float gpu_time_3 = 0;
	cudaEventElapsedTime(&gpu_time_1, start1, stop1);
	cudaEventElapsedTime(&gpu_time_2, start2, stop2);
	cudaEventElapsedTime(&gpu_time_3, start3, stop3);
	printf("gpu_time_1 = %f\t gpu_time_2 = %f\t gpu_time_3 = %f\n",gpu_time_1, gpu_time_2, gpu_time_3);
	printf("Total GPU time = %f\n", gpu_time_1+gpu_time_2+gpu_time_3);
	
	float cpu_time_1 = (((cend1.tv_sec * 1000000 + cend1.tv_usec) - (cstart1.tv_sec * 1000000 + cstart1.tv_usec))/1000.0);
	float cpu_time_2 = (((cend2.tv_sec * 1000000 + cend2.tv_usec) - (cstart2.tv_sec * 1000000 + cstart2.tv_usec))/1000.0);
	float cpu_time_3 = (((cend3.tv_sec * 1000000 + cend3.tv_usec) - (cstart3.tv_sec * 1000000 + cstart3.tv_usec))/1000.0);

	printf("cpu_time_1 = %f\t cpu_time_2 = %f\t cpu_time_3 = %f\n",cpu_time_1, cpu_time_2, cpu_time_3);
	printf("Total CPU time = %f\n", cpu_time_1+cpu_time_2+cpu_time_3);

	printf(" Total time = %f\n", gpu_time_1+gpu_time_2+gpu_time_3+ cpu_time_1+cpu_time_2+cpu_time_3);
	
	return 0;
}
