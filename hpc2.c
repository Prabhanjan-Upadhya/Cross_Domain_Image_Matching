#include<math.h>
#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<sys/time.h>

#define SQUARE(x) ((x)*(x))
#define PI 3.14


void NormalizeGray(double input[], int inputCols, int inputRows, unsigned char output[], int level)
{
	double min = input[0];
	double max = input[0];

	for (int i = 1; i < inputCols*inputRows; i++)
	{
		if (input[i] > max)
		{
			max = input[i];
		}
		else if (input[i] < min)
		{
			min = input[i];
		}
	}

	for (int i = 0; i < inputCols*inputRows; i++)
	{
		output[i] = (input[i] - min)*level / (max - min);

	}
}

void ComputeGradient(unsigned char input[], int inputCols, int inputRows, double outx[], double outy[], double outmag[])
{
	double sumX, sumY;
	int sobelX[9] = { 1, 0, -1, 2, 0, -2, 1, 0, -1 };
	int sobelY[9] = { 1, 2, 1, 0, 0, 0, -1, -2, -1 };

	for (int r = 1; r < inputRows-1; r++)
	{
		for (int c = 1; c < inputCols-1; c++)
		{
			sumX = 0;
			sumY = 0;
			for (int r2 = -1; r2 <= 1; r2++)
			{
				for (int c2 = -1; c2 <= 1; c2++)
				{
					sumX += input[(r + r2)*inputCols + (c + c2)] * sobelX[(r2 + 1) * 3 + c2 + 1];
					sumY += input[(r + r2)*inputCols + (c + c2)] * sobelY[(r2 + 1) * 3 + c2 + 1];
				}
			}
			outx[(r*inputCols) + c] = sumX;
			outy[(r*inputCols) + c] = sumY;
			outmag[(r*inputCols) + c] = sqrt((SQUARE(sumX) + SQUARE(sumY)));
		}
	}
}

void ComputePhase(int inputCols, int inputRows, double gradx[], double grady[], double gradPhase[])
{
	for (int i = 0; i < inputCols*inputRows; i++)
	{
		float theta = atan2(grady[i], gradx[i]);
		theta = theta * 180 / PI;
		gradPhase[i] = theta;
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
	double *gradXImage, *gradYImage, *gradMag, *gradPhase;
	double *testgradXImage, *testgradYImage, *testgradPhase, *testgradMag;
	struct timeval cstart, cend;

	gettimeofday(&cstart, NULL);
	printf("Initialization done!\n");

	if ((fptr = fopen(argv[1], "r")) == NULL)
	{
		printf("Unable to open input file for reading\n");
		exit(0);
	}
	//Open and load input image
	fptr = fopen(argv[1], "rb");
	fscanf(fptr, "%s %d %d %d", &inputHeader, &inputCols, &inputRows, &inputBytes);
	Header_1[0] = fgetc(fptr);	/* read white-space character that separates header */
	inputImage = (unsigned char*)calloc(inputCols*inputRows, sizeof(unsigned char));
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

	fptr = fopen("inputcheck.pgm", "wb");
	fprintf(fptr, "P5 %d %d 255\n", inputCols, inputRows);
	fwrite(inputImage, inputCols*inputRows, 1, fptr);
	fclose(fptr);
	fptr = fopen("testcheck.pgm", "wb");
	fprintf(fptr, "P5 %d %d 255\n", testCols, testRows);
	fwrite(testImage, testCols*testRows, 1, fptr);
	fclose(fptr);

	//printf("here1\n");
	//Allocate memory to all images
	gradXImage = (double*)calloc(inputCols*inputRows, sizeof(double));
	gradYImage = (double*)calloc(inputCols*inputRows, sizeof(double));
	gradMag = (double*)calloc(inputCols*inputRows, sizeof(double));
	gradPhase = (double*)calloc(inputCols*inputRows, sizeof(double));
	testgradXImage = (double*)calloc(testCols*testRows, sizeof(double));
	testgradYImage = (double*)calloc(testCols*testRows, sizeof(double));
	testgradPhase = (double*)calloc(testCols*testRows, sizeof(double));
	testgradMag = (double*)calloc(testCols*testRows, sizeof(double));

	normalGradMag = (unsigned char*)calloc(inputCols*inputRows, sizeof(unsigned char));
	normalGrad_x = (unsigned char*)calloc(inputCols*inputRows, sizeof(unsigned char));
	normalGrad_y = (unsigned char*)calloc(inputCols*inputRows, sizeof(unsigned char));
	normalGradPhase = (unsigned char*)calloc(inputCols*inputRows, sizeof(unsigned char));
	normaltestMag = (unsigned char*)calloc(testCols*testRows, sizeof(unsigned char));
	normaltest_x = (unsigned char*)calloc(testCols*testRows, sizeof(unsigned char));
	normaltest_y = (unsigned char*)calloc(testCols*testRows, sizeof(unsigned char));
	normaltestPhase = (unsigned char*)calloc(testCols*testRows, sizeof(unsigned char));

	//Compute gradients, phase and magnitude
	ComputeGradient(inputImage, inputCols, inputRows, gradXImage, gradYImage, gradMag);
	ComputeGradient(testImage, testCols, testRows, testgradXImage, testgradYImage, testgradMag);

	//Compute phase
	ComputePhase(inputCols, inputRows, gradXImage, gradYImage, gradPhase);
	ComputePhase(testCols, testRows, testgradXImage, testgradYImage, testgradPhase);

	NormalizeGray(gradXImage, inputCols, inputRows, normalGrad_x, 255);
	NormalizeGray(gradYImage, inputCols, inputRows, normalGrad_y, 255);
	NormalizeGray(gradMag, inputCols, inputRows, normalGradMag, 255);
	NormalizeGray(gradPhase, inputCols, inputRows, normalGradPhase, 255);
	NormalizeGray(testgradXImage, testCols, testRows, normaltest_x, 255);
	NormalizeGray(testgradYImage, testCols, testRows, normaltest_y, 255);
	NormalizeGray(testgradMag, testCols, testRows, normaltestMag, 255);
	NormalizeGray(testgradPhase, testCols, testRows, normaltestPhase, 255);

	int histo[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	int testhisto[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	int difference[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };

	//Compute histogram of gradient orientations of input image
	double angle = 0;
	for (int i = 0; i < inputRows*inputCols; i++)
	{
		if (normalGradMag[i] > 40)
		{
			if (gradPhase[i] > 0)
			{
				angle = gradPhase[i];
			}
			else
			{
				angle = fabs(gradPhase[i]+180);
			}
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

	//Compute histogram of gradient orientations of test image
	angle = 0;
	for (int i = 0; i < testRows*testCols; i++)
	{
		if (normaltestMag[i] > 40)
		{
			if (testgradPhase[i] > 0)
			{
				angle = testgradPhase[i];
			}
			else
			{
				angle = fabs(testgradPhase[i]+180);
			}
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

	//Check the dissimilarity in histogram of gradient orientations
	int sumDiff = 0;
	for (int i = 0; i < 9; i++)
	{
		difference[i] = abs(histo[i] - testhisto[i]);
		sumDiff += difference[i];
	}

	//cosine similarity
	double adotb = 0;
	double moda = 0;
	double modb = 0;
	for (int i=0; i<9;i++)
	{
		adotb += histo[i]*testhisto[i];
		moda += histo[i]*histo[i]; 
		modb += testhisto[i]*testhisto[i];
	}
	double similarity = adotb/(sqrt(moda)*sqrt(modb));
	printf("Similarity = %f\n",similarity);

	//Write out the output images
	fptr = fopen("norm_grad_x.pgm", "wb");
	fprintf(fptr, "P5 %d %d 255\n", inputCols, inputRows);
	fwrite(normalGrad_x, inputCols*inputRows, 1, fptr);
	fclose(fptr);

	fptr = fopen("norm_grad_y.pgm", "wb");
	fprintf(fptr, "P5 %d %d 255\n", inputCols, inputRows);
	fwrite(normalGrad_y, inputCols*inputRows, 1, fptr);
	fclose(fptr);

	fptr = fopen("norm_grad_mag.pgm", "wb");
	fprintf(fptr, "P5 %d %d 255\n", inputCols, inputRows);
	fwrite(normalGradMag, inputCols*inputRows, 1, fptr);
	fclose(fptr);

	fptr = fopen("norm_grad_phase.pgm", "wb");
	fprintf(fptr, "P5 %d %d 255\n", inputCols, inputRows);
	fwrite(normalGradPhase, inputCols*inputRows, 1, fptr);
	fclose(fptr);

	// test image files
	fptr = fopen("test_grad_x.pgm", "wb");
	fprintf(fptr, "P5 %d %d 255\n", testCols, testRows);
	fwrite(normaltest_x, testCols*testRows, 1, fptr);
	fclose(fptr);

	fptr = fopen("test_grad_y.pgm", "wb");
	fprintf(fptr, "P5 %d %d 255\n", testCols, testRows);
	fwrite(normaltest_y, testCols*testRows, 1, fptr);
	fclose(fptr);

	fptr = fopen("test_grad_mag.pgm", "wb");
	fprintf(fptr, "P5 %d %d 255\n", testCols, testRows);
	fwrite(normaltestMag, testCols*testRows, 1, fptr);
	fclose(fptr);

	fptr = fopen("test_grad_phase.pgm", "wb");
	fprintf(fptr, "P5 %d %d 255\n", testCols, testRows);
	fwrite(normaltestPhase, testCols*testRows, 1, fptr);
	fclose(fptr);

	//Free allocated memory
	free(gradXImage);
	free(gradYImage);
	free(gradMag);
	free(gradPhase);
	free(testgradXImage);
	free(testgradYImage);
	free(testgradPhase);
	free(testgradMag);
	free(normalGradMag);
	free(normalGrad_x);
	free(normalGrad_y);
	free(normalGradPhase);
	free(normaltestMag);
	free(normaltest_x);
	free(normaltest_y);
	free(normaltestPhase);

	free(inputImage);
	free(testImage);

	//Calculate time taken
	gettimeofday(&cend, NULL);
	float cpu_time = (((cend.tv_sec * 1000000 + cend.tv_usec) - (cstart.tv_sec * 1000000 + cstart.tv_usec)) / 1000.0);
	printf("cpu time = %f ms\n", cpu_time);

	return 0;
}

