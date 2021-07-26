//arguments are image, gaussian mu, gaussian sigma, gaussian size, weak edge threshold, strong edge threshold

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <math.h>
#include <cstdarg>

#include <iomanip>

#define pi 3.14159265359

using namespace cv;
using namespace std;

class edgePixel {
public:
	int row = 0;
	int col = 0;
	edgePixel* prev;
};

/*
class stack {
public:
	edgePixel* top;
	edgePixel* bottom;
	int size;

	stack() {
		top = NULL;
		bottom = NULL;
		size = 0;
	}

	void push(int row, int col, bool left, bool right) {
		edgePixel* temp = new edgePixel();
		temp->row = row;
		temp->col = col;
		temp->left = left;
		temp->right = right;
		if (size++ == 0) {
			bottom = temp;
			top = temp;
		}
		else {

		}

	}
};
*/

vector<vector<double> > gaussian2d(double mu, double cov, int n);
template<typename T>
Mat correlate(Mat& image, vector<vector<double> >& kernel);
template<typename T>
vector<vector<double> > correlate2(Mat& image, vector<vector<double> >& kernel);
template<typename T>
void cannyEdgeDetection(Mat& image, char** argv);
double matScalarMult(double a, double b);
double matScalarAdd(double a, double b);
template<typename T>
Mat toMat(vector<vector<double> >& mat, int type, bool adjust);
template<typename T>
void printMat(Mat& image, int xRange, int yRange);
void printMat(vector<vector<double> >& mat, int xStart, int xEnd, int yStart, int yEnd);
vector<vector<double> > nonMaxSupression(vector<vector<double> >& imgDX, vector<vector<double> >& imgDY, vector<vector<double> >& imgGM);
double bilInterp(vector<vector<double> >& imgD, double row, double col);
double linInterp(vector<vector<double> >& imgD, double row, double col, double point);
double interp(vector<vector<double> >& imgD, double row, double col);
void sideTrace(vector<vector<double> >& imgDX, vector<vector<double> >& imgDY, int row, int col, double moveRow, double moveCol, bool cont, ...);
template<typename T>
vector<vector<double> > edgeLink(vector<vector<double> >& imgGMS, vector<vector<double> >& imgDX, vector<vector<double> >& imgDY, int type, double threshCont, double threshStart);
void normalize(vector<vector<double> >& mat);

int main(int argc, char** argv) //arguments are image, gaussian mu, gaussian sigma, gaussian size, weak edge threshold, strong edge threshold
{
	Mat image;
	image = imread(argv[1], IMREAD_COLOR); // Read the file
	if (image.empty()) // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	int type = image.type();
	//cout << "type: " << type << endl;
	if (type % 8 == 0) {
		cannyEdgeDetection<uint8_t>(image, argv); //should this be an unsigned char?
	}
	else if (type % 8 == 1) {
		cannyEdgeDetection<int8_t>(image, argv);
	}
	else if (type % 8 == 2) {
		cannyEdgeDetection<uint16_t>(image, argv);
	}
	else if (type % 8 == 3) {
		cannyEdgeDetection<int16_t>(image, argv);
	}
	else if (type % 8 == 4) {
		cannyEdgeDetection<int32_t>(image, argv);
	}
	else if (type % 8 == 5) {
		cannyEdgeDetection<float>(image, argv);
	}
	else if (type % 8 == 6) {
		cannyEdgeDetection<double>(image, argv);
	}
	else {
		cout << "Image type not supported" << endl;
	}
	
	return 0;
}


template<typename T>
void cannyEdgeDetection(Mat& image, char** argv) {
	namedWindow("Original Image", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Original Image", image); // Show our image inside it.

	Mat imageGray;

	cvtColor(image, imageGray, COLOR_BGR2GRAY, 0);

	//cout << imageGray.type() << endl;

	vector<vector<double> > gausKern = gaussian2d(stod(string(argv[2])), stod(string(argv[3])), stoi(string(argv[4])));

	//printMat(gausKern, 0, 20, 0, 20);

	Mat imageBlurred = correlate<T>(imageGray, gausKern);

	//Mat imageBlurred = imageGray;

	vector<vector<double> > DX(3);
	DX[0] = vector<double>{ -0.25, 0, 0.25 };
	DX[1] = vector<double>{ -0.5, 0, 0.5 };
	DX[2] = vector<double>{ -0.25, 0, 0.25 };
	//DX[0] = vector<double>{ -1, 0, 1 };
	//DX[1] = vector<double>{ -2, 0, 2 };
	//DX[2] = vector<double>{ -1, 0, 1 };
	/*auto normalizeKern = [&](vector<vector<double> >& dx, double n) {
		for (vector<double>& row : dx) {
			for (double& col : row) {
				col /= n;
			}
		}

		printMat(dx, 4, 4);
	};*/

	//normalizeKern(DX, 4);

	vector<vector<double> > DY(3);
	DY[0] = vector<double>{ -0.25, -0.5, -0.25 };
	DY[1] = vector<double>{ 0, 0, 0 };
	DY[2] = vector<double>{ 0.25, 0.5, 0.25 };

	//normalizeKern(DY, 4);

	vector<vector<double> > imgDX = correlate2<T>(imageBlurred, DX);
	vector<vector<double> > imgDY = correlate2<T>(imageBlurred, DY);

	vector<vector<double> > imgGM(imgDX.size());
	for (int i = 0; i < imgGM.size(); i++) {
		imgGM[i] = vector<double>(imgDX[0].size());
		for (int j = 0; j < imgGM[0].size(); j++) {
			imgGM[i][j] = sqrt((pow(imgDX[i][j], 2) + pow(imgDY[i][j], 2)) / 2);
		}
	}
	/*
	double largest = 0;
	for (int i = 5; i < imgDX.size() - 5; i++) {
		for (int j = 5; j < imgDX[0].size() - 5; j++) {
			if (abs(imgDX[i][j]) > largest) {
				largest = abs(imgDX[i][j]);
			}
		}
	}

	cout << "largest imgDX: " << largest << endl;

	largest = 0;
	for (int i = 5; i < imgDY.size() - 5; i++) {
		for (int j = 5; j < imgDY[0].size() - 5; j++) {
			if (abs(imgDY[i][j]) > largest) {
				largest = abs(imgDY[i][j]);
			}
		}
	}


	cout << "largest imgDY: " << largest << endl;

	largest = 0;
	for (int i = 5; i < imgGM.size() - 5; i++) {
		for (int j = 5; j < imgGM[0].size() - 5; j++) {
			if (imgGM[i][j] > largest) {
				largest = imgGM[i][j];
			}
		}
	}

	cout << "largest imgGM: " << largest << endl;
	*/
	vector<vector<double> > imgGMS = nonMaxSupression(imgDX, imgDY, imgGM);
	//printMat(imgGM, 8, 18, 5, 10);
	//cout << "========================================" << endl;
	//printMat(imgGMS, 8, 18, 5, 50);

	vector<vector<double> > imgEdges = edgeLink<T>(imgGMS, imgDX, imgDY, imageBlurred.type(), stod(string(argv[5])), stod(string(argv[6])));

	vector<vector<double> > imgEdgesBright(imgEdges.size());
	for (int i = 0; i < imgEdgesBright.size(); i++) {
		imgEdgesBright[i] = vector<double>(imgEdges[0].size(), 0);
	}
	for (int i = 0; i < imgEdges.size(); i++) {
		for (int j = 0; j < imgEdges[0].size(); j++) {
			if (imgEdges[i][j] > 0) {
				imgEdgesBright[i][j] = 255;
			}
		}
	}

	vector<vector<double> > imgGMSBright(imgGMS.size());
	for (int i = 0; i < imgGMSBright.size(); i++) {
		imgGMSBright[i] = vector<double>(imgGMS[0].size(), 0);
	}
	for (int i = 0; i < imgGMS.size(); i++) {
		for (int j = 0; j < imgGMS[0].size(); j++) {
			if (imgGMS[i][j] > 0) {
				imgGMSBright[i][j] = 255;
			}
		}
	}

	//normalize(imgGM);
	//normalize(imgGMS);
	//normalize(imgEdges);

	//Mat imgGMO = toMat<T>(imgGM, imageBlurred.type(), false);
	//Mat imgDXO = toMat<T>(imgDX, imageBlurred.type(), true);
	//Mat imgDYO = toMat<T>(imgDY, imageBlurred.type(), true);
	//Mat imgGMSO = toMat<T>(imgGMS, imageBlurred.type(), false);
	//Mat imgEdgesO = toMat<T>(imgEdges, imageBlurred.type(), false);
	Mat imgEdgesBrightO = toMat<T>(imgEdgesBright, imageBlurred.type(), false);
	//Mat imgGMSBrightO = toMat<T>(imgGMSBright, imageBlurred.type(), false);

	//printMat(imgGMS, 0, 360, 0, 260);

	//namedWindow("imgGray", WINDOW_AUTOSIZE);
	//imshow("imgGray", imageGray);

	//namedWindow("imgBlurred", WINDOW_AUTOSIZE);
	//imshow("imgBlurred", imageBlurred);

	//namedWindow("imgGM", WINDOW_AUTOSIZE);
	//imshow("imgGM", imgGMO);

	//namedWindow("imgDX", WINDOW_AUTOSIZE);
	//imshow("imgDX", imgDXO);

	//namedWindow("imgDY", WINDOW_AUTOSIZE);
	//imshow("imgDY", imgDYO);

	//namedWindow("imgGMS", WINDOW_AUTOSIZE);
	//imshow("imgGMS", imgGMSO);

	//namedWindow("imgEdges", WINDOW_AUTOSIZE);
	//imshow("imgEdges", imgEdgesO);

	namedWindow("imgEdgesBright", WINDOW_AUTOSIZE);
	imshow("imgEdgesBright", imgEdgesBrightO);

	//namedWindow("imgGMSBright", WINDOW_AUTOSIZE);
	//imshow("imgGMSBright", imgGMSBrightO);
	waitKey(0);
}

vector<vector<double> > gaussian2d(double mu, double cov, int n) {
	int size = (2 * n) + 1;
	double* gaus1D = new double[size];

	for (int i = 0; i < size; i++) {
		int x = i - n;
		gaus1D[i] = (1 / sqrt(2 * pi * pow(cov, 2))) * exp(-1 * (pow(x - mu, 2) / (2 * pow(cov, 2))));
	}

	vector<vector<double> > gaus(size);
	for (int i = 0; i < size; i++) {
		gaus[i] = vector<double>(size);
	}
	double sum = 0;
	for (int row = 0; row < size; row++) {
		for (int col = 0; col < size; col++) {
			gaus[row][col] = gaus1D[row] * gaus1D[col];
			sum += gaus[row][col];
		}
	}
	delete[] gaus1D;

	for (int row = 0; row < size; row++) {
		for (int col = 0; col < size; col++) {
			gaus[row][col] /= sum;
		}
	}
	return gaus;
}

template<typename T>
Mat correlate(Mat& image, vector<vector<double> >& kernel) {
	/*
	U retImage;
	int type = 0;
	if (typeid(U) == typeid(vector<vector<double> >)) {
		retImage.~U();
		new(&retImage) vector<vector<double> >(image.rows);
		for (int i = 0; i < retImage.size(); i++) {
			retImage[i] = vector<double>(image.cols);
		}
	}
	else {
		retImage.~U();
		new(&retImage) Mat(image.rows, image.cols, image.type());
		type = 1;
	}
	*/
	Mat retImage(image.rows, image.cols, image.type());
	
	int n = (kernel.size() - 1) / 2;

	for (int row = 0; row < image.rows; row++) {
		for (int col = 0; col < image.cols; col++) {
			double sum = 0;
			for (int kRow = 0; kRow < kernel.size(); kRow++) {
				for (int kCol = 0; kCol < kernel[0].size(); kCol++) {
					if ((kRow - n + row >= 0) && (kRow - n + row < image.rows) && (kCol - n + col >= 0) && (kCol - n + col < image.cols)) {
						
						sum += image.at<T>(row + kRow - n, col + kCol - n) * kernel[kRow][kCol];
					}
				}
			}
			//if (type == 0) {
				//retImage[row][col] = sum;
			//}
			//else {
			retImage.at<T>(row, col) = (T)sum;
			//}
		}
	}
	
	return retImage;
}

template<typename T>
vector<vector<double> > correlate2(Mat& image, vector<vector<double> >& kernel) {
	vector<vector<double> > retImage(image.rows);
	for (int i = 0; i < retImage.size(); i++) {
		retImage[i] = vector<double>(image.cols);
	}
	
	int n = (kernel.size() - 1) / 2;

	for (int row = 0; row < image.rows; row++) {
		for (int col = 0; col < image.cols; col++) {
			double sum = 0;
			for (int kRow = 0; kRow < kernel.size(); kRow++) {
				for (int kCol = 0; kCol < kernel[0].size(); kCol++) {
					if ((kRow - n + row >= 0) && (kRow - n + row < image.rows) && (kCol - n + col >= 0) && (kCol - n + col < image.cols)) {

						sum += image.at<T>(row + kRow - n, col + kCol - n) * kernel[kRow][kCol];
					}
				}
			}
			retImage[row][col] = sum;
		}
	}

	return retImage;
}

void matScalarEWOP(vector<vector<double> >& mat, double n, double (*op)(double, double)) {
	for (int i = 0; i < mat.size(); i++) {
		for (int j = 0; j < mat[0].size(); j++) {
			mat[i][j] = op(mat[i][j], n);
		}
	}
}

double matScalarMult(double a, double b) {
	return a * b;
}

double matScalarAdd(double a, double b) {
	return a + b;
}

template<typename T>
Mat toMat(vector<vector<double> >& mat, int type, bool adjust) {
	if (adjust) {
		matScalarEWOP(mat, round(pow(2, (sizeof(T) * 8)) - 1), &matScalarAdd);
		matScalarEWOP(mat, 0.5, &matScalarMult);
	}
	Mat retMat(mat.size(), mat[0].size(), type);
	for (int i = 0; i < mat.size(); i++) {
		for (int j = 0; j < mat[0].size(); j++) {
			retMat.at<T>(i, j) = (T)mat[i][j];
		}
	}

	return retMat;
}

template<typename T>
void printMat(Mat& image, int xRange, int yRange) {
	for (int i = 0; (i < image.rows) && (i < yRange); i++) {
		cout << "row: " << i << " | ";
		for (int j = 0; (j < image.cols) && (j < xRange); j++) {
			cout << (double)image.at<T>(i, j) << " ";
		}
		cout << endl;
	}
}

void printMat(vector<vector<double> >& mat, int xStart, int xEnd, int yStart, int yEnd) {
	cout << fixed;
	for (int i = yStart; (i < mat.size()) && (i < yEnd); i++) {
		cout << "row: " << i << " | ";
		for (int j = xStart; (j < mat[0].size()) && (j < xEnd); j++) {
			//if (mat[i][j] == 0) {
				cout << setprecision(3) << mat[i][j] << " ";
			//}
			//else {
				//cout << setprecision(3) << "col: " << j << " " << mat[i][j] << " ";
			//}
		}
		cout << endl;
	}
}


vector<vector<double> > nonMaxSupression(vector<vector<double> >& imgDX, vector<vector<double> >& imgDY, vector<vector<double> >& imgGM) {
	vector<vector<double> > imgGMS(imgGM.size());
	double step = 0.25;


	for (int row = 0; row < imgGMS.size(); row++) {
		imgGMS[row] = vector<double>(imgGM[0].size(), 0);
	}

	for (int row = 0; row < imgGMS.size(); row++) {
		for (int col = 0; col < imgGMS[0].size(); col++) {
			double moveRow = imgDY[row][col];
			double moveCol = imgDX[row][col];

			//scale so that the hypotenuse length is equal to step size
			double scale = step / sqrt(pow(moveCol, 2) + pow(moveRow, 2));
			moveRow *= scale;
			moveCol *= scale;

			double gm1 = -1;
			double gm2 = -1;

			sideTrace(imgDX, imgDY, row, col, moveRow, moveCol, false, &gm1);
			sideTrace(imgDX, imgDY, row, col, -moveRow, -moveCol, false, &gm2);

			double* largest;
			if (gm1 > gm2) {
				largest = &gm1;
			}
			else {
				largest = &gm2;
			}
			if (imgGM[row][col] > *largest) {
				imgGMS[row][col] = imgGM[row][col];
			}
			else {
				double lRow = 0;
				double lCol = 0;
				if (largest == &gm2) {
					moveRow = -moveRow;
					moveCol = -moveCol;
				}
				sideTrace(imgDX, imgDY, row, col, moveRow, moveCol, true, *largest, &lRow, &lCol);
				
				//cout << "nonMaxSupression | lRow: " << lRow << " | lCol: " << lCol << endl;
				imgGMS[round(lRow)][round(lCol)] = imgGM[round(lRow)][round(lCol)];
			}
		}
	}

	return imgGMS;
}

void sideTrace(vector<vector<double> >& imgDX, vector<vector<double> >& imgDY, int row, int col, double moveRow, double moveCol, bool cont, ...) { //if 1 arg then will be gm, else will be gmPrev, retrunPointRow, returnPointCol
	va_list args;
	va_start(args, cont);

	double moveRowTotal = moveRow;
	double moveColTotal = moveCol;
	bool done = false;
	double gmPrev = 0;
	double* retRow = NULL;
	double* retCol = NULL;
	if (cont) {
		gmPrev = va_arg(args, double);
		moveRowTotal += moveRow;
		moveColTotal += moveCol;
		retRow = va_arg(args, double*);
		retCol = va_arg(args, double*);
	}
	do {
		if ((row + moveRowTotal > -0.5) && (row + moveRowTotal < imgDX.size() - 0.5) && (col + moveColTotal > -0.5) && (col + moveColTotal < imgDX[0].size() - 0.5)) {
			double gdx = interp(imgDX, row + moveRowTotal, col + moveColTotal);
			double gdy = interp(imgDY, row + moveRowTotal, col + moveColTotal);
			double gm = sqrt((pow(gdx, 2) + pow(gdy, 2)) / 2);

			if (!cont) {
				*(va_arg(args, double*)) = gm;
			}
			else if (gm >= gmPrev) {
				gmPrev = gm;
				moveRowTotal += moveRow;
				moveColTotal += moveCol;
			}
			else {
				*retRow = row + moveRowTotal;
				*retCol = col + moveColTotal;
				done = true;
			}
		}
		else {
			done = true;
			if (cont) {
				*retRow = row + moveRowTotal - moveRow;
				*retCol = col + moveColTotal - moveCol;
			}
		}
	} while (!done && cont);
}

double interp(vector<vector<double> >& imgD, double row, double col) {
	if ((row < 0) && (col < 0)) {
		return imgD[0][0];
	}
	if ((row > imgD.size() - 1) && (col > imgD[0].size() - 1)) {
		return imgD[imgD.size() - 1][imgD[0].size() - 1];
	}
	if ((row < 0) && (col > imgD[0].size() - 1)) {
		return imgD[0][imgD[0].size() - 1];
	}
	if ((row > imgD.size() - 1) && (col < 0)) {
		return imgD[imgD.size() - 1][0];
	}
	if ((row > 0) && (row < imgD.size() - 1) && (col > 0) && (col < imgD[0].size() - 1)) {
		return bilInterp(imgD, row, col);
	}
	else if (row <= 0) {
		return linInterp(imgD, 0, col, col);
	}
	else if (row >= imgD.size() - 1) {
		return linInterp(imgD, imgD.size() - 1, col, col);
	}
	else if (col <= 0) {
		return linInterp(imgD, row, 0, row);
	}
	else {
		return linInterp(imgD, row, imgD[0].size() - 1, row);
	}
}


double linInterp(vector<vector<double> >& imgD, double row, double col, double point) {
	if (ceil(point) == floor(point)) {
		return imgD[row][col];
	}
	
	return ((point - floor(point)) * imgD[ceil(row)][ceil(col)]) + ((ceil(point) - point) * imgD[floor(row)][floor(col)]);
}

double bilInterp(vector<vector<double> >& imgD, double row, double col) {
	double& x1y1 = imgD[floor(row)][floor(col)];
	double& x2y1 = imgD[floor(row)][ceil(col)];
	double& x1y2 = imgD[ceil(row)][floor(col)];
	double& x2y2 = imgD[ceil(row)][ceil(col)];

	if (floor(row) == ceil(row)) {
		return linInterp(imgD, row, col, col);
	}
	if (floor(col) == ceil(col)) {
		return linInterp(imgD, row, col, row);
	}

	double x1y1x2y1 = ((col - floor(col)) * x2y1) + ((ceil(col) - col) * x1y1);
	double x1y2x2y2 = ((col - floor(col)) * x2y2) + ((ceil(col) - col) * x1y2);

	double ret = ((row - floor(row)) * x1y2x2y2) + ((ceil(row) - row) * x1y1x2y1);
	return ret;

}

//done with depth first search using loops instead of recursive function calls so that stack size doesnt get too large
template<typename T>
vector<vector<double> > edgeLink(vector<vector<double> >& imgGMS, vector<vector<double> >& imgDX, vector<vector<double> >& imgDY, int type, double threshCont, double threshStart) {
	//cout << "threshStart: " << threshStart << " | threshCont: " << threshCont << endl;
	//double threshStart = 60;
	//double threshCont = 1;
	vector<vector<double> > linkedEdges(imgGMS.size());
	for (int i = 0; i < imgGMS.size(); i++) {
		linkedEdges[i] = vector<double>(imgGMS[0].size(), 0);
	}
	vector<vector<vector<bool> > > edgeTracker(imgGMS.size());
	for (int i = 0; i < imgGMS.size(); i++) {
		edgeTracker[i] = vector<vector<bool> >(imgGMS[0].size());
		for (int j = 0; j < imgGMS[0].size(); j++) {
			edgeTracker[i][j] = vector<bool>(2, 0); //0 index is left, 1 index is right
		}
	}
	for (int row = 0; row < imgGMS.size(); row++) {
		for (int col = 0; col < imgGMS[0].size(); col++) {
			
			if ((imgGMS[row][col] >= threshStart) && (linkedEdges[row][col] == 0)) {
				//cout << "edgeLink | starting at row: " << row << " | col: " << col << " | imgGMS: " << imgGMS[row][col] << endl;
				//double moveRowTotal = 0;
				//double moveColTotal = 0;
				double moveRow = 0;
				double moveCol = 0;
				//edgePixel* prev = NULL;
				edgePixel* ep = new edgePixel();
				ep->prev = NULL;
				ep->row = row;
				ep->col = col;
				edgeTracker[row][col][1] = true;
				linkedEdges[row][col] = imgGMS[row][col];
				bool done = false;
				bool dir = true; //false is left, true is right
				while (!done) {
					//moveRow = imgDY[row + moveRowTotal][col + moveColTotal];
					//moveCol = imgDX[row + moveRowTotal][col + moveColTotal];
					int& epRow = ep->row;
					int& epCol = ep->col;
					if ((epRow == 7) && (epCol == 13)) {
						int x = 0;
					}
					moveRow = imgDY[epRow][epCol];
					moveCol = imgDX[epRow][epCol];

					if ((moveRow == 0) && (moveCol == 0)) {
						int x = 0;
					}

					//cout << "edgeLink | moveRow: " << moveRow << " | moveCol: " << moveCol << endl;

					if (abs(moveRow) > abs(moveCol)) {
						moveCol /= moveRow;
						moveRow = 1;
					}
					else {
						moveRow /= moveCol;
						moveCol = 1;
					}

					if (dir) {
						//clockwise direction (right) to norm
						double temp = moveRow;
						moveRow = moveCol;
						moveCol = temp;
						moveCol = -moveCol;
						//cout << "clockwise direction: moveRow: " << moveRow << " | moveCol: " << moveCol << endl;
					}
					else {
						//counterclockwise direction (left) to norm
						double temp = moveRow;
						moveRow = moveCol;
						moveCol = temp;
						moveRow = -moveRow;
						//cout << "counterclockwise direction: moveRow: " << moveRow << " | moveCol: " << moveCol << endl;
					}

					moveRow = round(moveRow);
					moveCol = round(moveCol);

					bool elseFlag = true;
					//cout << "edgeLink | checking row: " << epRow + moveRow << " | checking col: " << epCol + moveCol << endl;
					if ((epRow + moveRow >= 0) && (epRow + moveRow <= imgGMS.size() - 1) && (epCol + moveCol >= 0) && (epCol + moveCol <= imgGMS[0].size() - 1)) {
						//cout << "within bounds" << endl;
						//depth first search starting with right side, so check if left side has been traveled as well as threshold
						if (imgGMS[epRow + moveRow][epCol + moveCol] >= threshCont && (!edgeTracker[epRow + moveRow][epCol + moveCol][0])) {
							
							//cout << "edgeLink | Adding pixel(row: " << epRow + moveRow << ", col: " << epCol + moveCol << ") to linkedEdges" << endl;
							elseFlag = false;
							if (edgeTracker[epRow + moveRow][epCol + moveCol][1]) { //if the right path has already been traveled
								dir = false;
								edgeTracker[epRow + moveRow][epCol + moveCol][0] = true;
							}
							else {
								dir = true;
							}
							edgePixel* temp = ep;
							ep = new edgePixel();
							ep->prev = temp;
							//ep->row = row + round(moveRow) + moveRowTotal;
							//ep->col = col + round(moveCol) + moveColTotal;
							//edgeTracker[row + round(moveRow) + moveRowTotal][col + round(moveCol) + moveColTotal][1] = true;
							//moveRowTotal += round(moveRow);
							//moveColTotal += round(moveCol);
							ep->row = epRow + moveRow;
							ep->col = epCol + moveCol;
							edgeTracker[epRow + moveRow][epCol + moveCol][1] = true;
							linkedEdges[epRow + moveRow][epCol + moveCol] = imgGMS[epRow + moveRow][epCol + moveCol];
						}
					}

					if (elseFlag) { //at the end of the current path
						if (dir) {
							dir = false;
							edgeTracker[epRow][epCol][0] = true;
						}
						else {
							edgePixel* temp = ep->prev;
							delete ep;
							ep = temp;
							if (ep == NULL) {
								done = true;
							}
						}
					}
					
				}

				/*
				do {
					moveRowTotal += round(moveRow);
					moveColTotal += round(moveCol);
					linkedEdges[row + moveRowTotal][col + moveColTotal] = imgGMS[row + moveRowTotal][col + moveColTotal];
					moveRow = imgDY[row][col];
					moveCol = imgDX[row][col];
					if (moveRow > moveCol) {
						moveCol /= moveRow;
						moveRow = 1;
					}
					else {
						moveRow /= moveCol;
						moveCol = 1;
					}

					//clockwise direction to norm
					double temp = moveRow;
					moveRow = moveCol;
					moveCol = temp;
					moveCol = -moveCol;
				} while (imgGMS[row + moveRowTotal + round(moveRow)][col + moveColTotal + round(moveCol)] >= threshCont);
				*/
			}
			//Mat frame = toMat<T>(linkedEdges, type, false);
			//cout << "new frame | row: " << row << " | col: " << col << endl;
			//imshow("edge linker", frame);
			//char c = (char)waitKey(1);
			//if (c == 27) break;
		}
	}
	return linkedEdges;
}

void normalize(vector<vector<double> >& mat) {
	double largest = 0;

	for (int i = 5; i < mat.size() - 5; i++) {
		for (int j = 5; j < mat[0].size() - 5; j++) {
			if (mat[i][j] > largest) {
				largest = mat[i][j];
			}
		}
	}

	for (int i = 5; i < mat.size() - 5; i++) {
		for (int j = 5; j < mat[0].size() - 5; j++) {
			
			mat[i][j] = (mat[i][j] / largest) * 255;
			
		}
	}
}