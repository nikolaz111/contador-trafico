//poner las areas con el mouse las q faltan



/* --Sparse Optical Flow Demo Program--
* Written by David Stavens (david.stavens@ai.stanford.edu)
*/
#include <iostream>
#include <stdio.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <math.h>
#include "cvplot.h"
//#include "cvplot.cpp"


#define MAX_NUM_FEATURES_POR_ZONA 10000
#define PI 3.14159265358979323846

using namespace std;
using namespace cv;


//--------Areas para hacer la deteccion
Rect areaOrigen;
bool areaOrigenSet = false;
Rect areaGiroIzq;
bool areaGiroIzqSet = false;
Rect areaGiroDer;
bool areaGiroDerSet = false;

//Parametros zona giro izquierda
int maxDistPuntosCarro_izq_trackbar = 200;
double maxDistPuntosCarro_izq = 20;
int maxAnguloPuntosCarro_izq_trackbar = (PI/4)*100;
double maxAnguloPuntosCarro_izq = PI/4;
int minPuntosCarro_izq = 8;

//Parametros zona giro derecha
int maxDistPuntosCarro_der_trackbar = 200;
double maxDistPuntosCarro_der = 20;
int maxAnguloPuntosCarro_der_trackbar = (PI/4)*100;
double maxAnguloPuntosCarro_der = PI/4;
int minPuntosCarro_der = 8;


//Cosas para el tracking
std::vector<Point> carrosTrack;
std::vector<int> carrosNoNewData;
int radioBusqueda = 20;
int maxFramesContarCarro = 5;


//Parametros generales
int minNorm = 2;
int maxNorm = 200;
int maxIter = 20;
int epsilon_trackbar = 300;
double epsilon = 0.3;
int maxError_trackbar = 100;
double maxError = 10;
int frame_video_trackbar = 0;
long current_frame = 0;
int tamanoVentana = 5;
int maxDistPuntosCarro_trackbar = 200;
double maxDistPuntosCarro = 20;
int maxAnguloPuntosCarro_trackbar = (PI/4)*100;
double maxAnguloPuntosCarro = PI/4;
int minPuntosCarro = 8;

bool selectObject = false;
Rect selection;
Point origin;
int trackObject = 0;


//-----Valores Canny
int threshold1 = 100;
int threshold2 = 200;
int apertureSize = 3;

//-----Cosas resta
#define numFramesPlot 1000


void on_trackbar_epsilon( int, void* )
{
	epsilon = epsilon_trackbar/1000;
}
void on_trackbar_error( int, void* )
{
	maxError = maxError_trackbar/10;
}
void on_trackbar_frame_video( int, void* )
{
	current_frame = frame_video_trackbar;
}
void on_trackbar_maxDistPuntosCarro( int, void* )
{
	maxDistPuntosCarro = maxDistPuntosCarro_trackbar/10;
}
void on_trackbar_maxAnguloPuntosCarro( int, void* )
{
	maxAnguloPuntosCarro = maxAnguloPuntosCarro_trackbar/100;
}

//Trackbars de controles giro izquierda
void on_trackbar_maxDistPuntosCarro_izq( int, void* )
{
	maxDistPuntosCarro_izq = maxDistPuntosCarro_izq_trackbar/10;
}
void on_trackbar_maxAnguloPuntosCarro_izq( int, void* )
{
	maxAnguloPuntosCarro_izq = maxAnguloPuntosCarro_izq_trackbar/100;
}

//Trackbars de controles giro derecha
void on_trackbar_maxDistPuntosCarro_der( int, void* )
{
	maxDistPuntosCarro_der = maxDistPuntosCarro_der_trackbar/10;
}
void on_trackbar_maxAnguloPuntosCarro_der( int, void* )
{
	maxAnguloPuntosCarro_der = maxAnguloPuntosCarro_der_trackbar/100;
}



static void onMouse( int event, int x, int y, int, void* )
{
	if( selectObject )
	{
		if( !areaOrigenSet )
		{
			areaOrigen.x = MIN(x, origin.x);
			areaOrigen.y = MIN(y, origin.y);
			areaOrigen.width = std::abs(x - origin.x);
			areaOrigen.height = std::abs(y - origin.y);
		}
		else if( !areaGiroIzqSet )
		{
			areaGiroIzq.x = MIN(x, origin.x);
			areaGiroIzq.y = MIN(y, origin.y);
			areaGiroIzq.width = std::abs(x - origin.x);
			areaGiroIzq.height = std::abs(y - origin.y);
		}
		else if( !areaGiroDerSet )
		{
			areaGiroDer.x = MIN(x, origin.x);
			areaGiroDer.y = MIN(y, origin.y);
			areaGiroDer.width = std::abs(x - origin.x);
			areaGiroDer.height = std::abs(y - origin.y);
		}

		selection.x = MIN(x, origin.x);
		selection.y = MIN(y, origin.y);
		selection.width = std::abs(x - origin.x);
		selection.height = std::abs(y - origin.y);


		//selection &= Rect(0, 0, image.cols, image.rows);
	}

	switch( event )
	{
	case CV_EVENT_LBUTTONDOWN:
		origin = Point(x,y);
		selection = Rect(x,y,0,0);
		selectObject = true;
		break;
	case CV_EVENT_LBUTTONUP:
		selectObject = false;
		if( selection.width > 0 && selection.height > 0 )
			trackObject = -1;
		if(!areaOrigenSet)
			areaOrigenSet = true;
		else if(!areaGiroIzqSet)
			areaGiroIzqSet = true;
		else if(!areaGiroDerSet)
			areaGiroDerSet = true;
		break;
	}
}

inline static double square(int a)
{
	return a * a;
}

inline static void puntosCercanosAlineados(std::vector<Point>* puntos, std::vector<int>* puntosPintar, int puntoInteres, CvPoint2D32f* frame1_features, CvPoint2D32f* frame2_features, char* optical_flow_found_feature, float* optical_flow_feature_error, bool* puntosMarcados, int numFeatures, int zona)
{
	if(!puntosMarcados[puntoInteres])
	{

		Point pPuntoInteres,qPuntoInteres;
		pPuntoInteres.x = (int) frame1_features[puntoInteres].x;
		pPuntoInteres.y = (int) frame1_features[puntoInteres].y;
		qPuntoInteres.x = (int) frame2_features[puntoInteres].x;
		qPuntoInteres.y = (int) frame2_features[puntoInteres].y;

		/* Criterios para safar punto
		* If Pyramidal Lucas Kanade didn't really find the feature, skip it. */
		if ( optical_flow_found_feature[puntoInteres] == 0 )	return;
		if ( optical_flow_feature_error[puntoInteres] > maxError )	return;

		Point pNuevoPuntoInteres = pPuntoInteres;
		Point qNuevoPuntoInteres = qPuntoInteres;

		Point vecPuntoInteres = qNuevoPuntoInteres - pNuevoPuntoInteres;
		if(norm(vecPuntoInteres) < minNorm)	return;
		if(norm(vecPuntoInteres) > maxNorm)	return;
		/* Fin de criterios para safar punto*/

		double anglePuntoInteres;		anglePuntoInteres = atan2( (double) pPuntoInteres.y - qPuntoInteres.y, (double) pPuntoInteres.x - qPuntoInteres.x );
		//double hypotenusePuntoInteres;	hypotenusePuntoInteres = sqrt( square(pPuntoInteres.y - qPuntoInteres.y) + square(pPuntoInteres.x - qPuntoInteres.x) );

		for(int i = 0; i < (numFeatures); i++)
		{
			if(!puntosMarcados[i] && puntoInteres != i)
			{

				Point p,q;
				p.x = (int) frame1_features[i].x;
				p.y = (int) frame1_features[i].y;
				q.x = (int) frame2_features[i].x;
				q.y = (int) frame2_features[i].y;

				/* Criterios para safar punto
				* If Pyramidal Lucas Kanade didn't really find the feature, skip it. */
				if ( optical_flow_found_feature[i] == 0 )	continue;
				if ( optical_flow_feature_error[i] > maxError )	continue;

				Point pNuevo = p;
				Point qNuevo = q;

				Point vec = qNuevo - pNuevo;
				if(norm(vec) < minNorm)	continue;
				if(norm(vec) > maxNorm)	continue;
				/* Fin de criterios para safar punto*/

				double angle;		angle = atan2( (double) p.y - q.y, (double) p.x - q.x );
				//double hypotenuse;	hypotenuse = sqrt( square(p.y - q.y) + square(p.x - q.x) );
				double dist = norm(pPuntoInteres - p);

				//criterios segun zona
				double maxAng;
				double maxDist;

				if(zona == 0)
				{
					maxAng = maxAnguloPuntosCarro;
					maxDist = maxDistPuntosCarro;
				}
				else if(zona == 1)
				{
					maxAng = maxAnguloPuntosCarro_izq;
					maxDist = maxDistPuntosCarro_izq;
				}
				else if(zona == 2)
				{
					maxAng = maxAnguloPuntosCarro_der;
					maxDist = maxDistPuntosCarro_der;
				}
				if(abs(angle-anglePuntoInteres)>maxAng || dist>maxDist) 
					continue;
				else
				{
					puntos->push_back(p);
					puntosMarcados[i] = true;
					puntosPintar->push_back(i);
					puntosCercanosAlineados(puntos, puntosPintar, i, frame1_features, frame2_features, optical_flow_found_feature, optical_flow_feature_error, puntosMarcados, numFeatures, zona);
				}
			}
		}

		puntos->push_back(pPuntoInteres);
		puntosPintar->push_back(puntoInteres);

	}
}

/* This is just an inline that allocates images.  I did this to reduce clutter in the
* actual computer vision algorithmic code.  Basically it allocates the requested image
* unless that image is already non-NULL.  It always leaves a non-NULL image as-is even
* if that image's size, depth, and/or channels are different than the request.
*/
inline static void allocateOnDemand( IplImage **img, CvSize size, int depth, int channels )
{
	if ( *img != NULL )	return;

	*img = cvCreateImage( size, depth, channels );
	if ( *img == NULL )
	{
		fprintf(stderr, "Error: Couldn't allocate image.  Out of memory?\n");
		exit(-1);
	}
}

int nosotros(long number_of_frames, CvSize frame_size, CvCapture* input_video)
{

	static IplImage *frame = NULL, *frame1 = NULL, *frame1_1C = NULL, *frame2_1C = NULL;
	//piramides para las 3 diferentes zonas
	static IplImage *pyramid1Origen = NULL, *pyramid2Origen = NULL, *pyramid1Izq = NULL, *pyramid2Izq = NULL, *pyramid1Der = NULL, *pyramid2Der = NULL;
	//cosas para las 3 diferentes zonas
	static IplImage *eig_imageOrigen = NULL, *temp_imageOrigen = NULL, *eig_imageIzq = NULL, *temp_imageIzq = NULL, *eig_imageDer = NULL, *temp_imageDer = NULL; 
	//mascaras
	static IplImage *maskOrigen = NULL, *maskIzq = NULL, *maskDer = NULL;		

	// Go to the frame we want.  Important if multiple frames are queried in
	// the loop which they of course are for optical flow.  Note that the very
	// first call to this is actually not needed. (Because the correct position
	// is set outsite the for() loop.)
	//
	cvSetCaptureProperty( input_video, CV_CAP_PROP_POS_FRAMES, current_frame );

	// Get the next frame of the video.
	// IMPORTANT!  cvQueryFrame() always returns a pointer to the _same_
	// memory location.  So successive calls:
	// frame1 = cvQueryFrame();
	// frame2 = cvQueryFrame();
	// frame3 = cvQueryFrame();
	// will result in (frame1 == frame2 && frame2 == frame3) being true.
	// The solution is to make a copy of the cvQueryFrame() output.	
	frame = cvQueryFrame( input_video );
	if (frame == NULL)
	{
		// Why did we get a NULL frame?  We shouldn't be at the end. */
		fprintf(stderr, "Error: Hmm. The end came sooner than we thought.\n");
		return -1;
	}
	// Allocate another image if not already allocated.
	// Image has ONE channel of color (ie: monochrome) with 8-bit "color" depth.
	// This is the image format OpenCV algorithms actually operate on (mostly).
	allocateOnDemand( &frame1_1C, frame_size, IPL_DEPTH_8U, 1 );

	// Convert whatever the AVI image format is into OpenCV's preferred format.
	// AND flip the image vertically.  Flip is a shameless hack.  OpenCV reads
	// in AVIs upside-down by default.  (No comment :-))
	cvConvertImage(frame, frame1_1C);


	// We'll make a full color backup of this frame so that we can draw on it.
	// (It's not the best idea to draw on the static memory space of cvQueryFrame().)
	allocateOnDemand( &frame1, frame_size, IPL_DEPTH_8U, 3 );
	cvConvertImage(frame, frame1);

	// Get the second frame of video.  Same principles as the first. */
	frame = cvQueryFrame( input_video );
	if (frame == NULL)
	{
		fprintf(stderr, "Error: Hmm. The end came sooner than we thought.\n");
		return -1;
	}
	allocateOnDemand( &frame2_1C, frame_size, IPL_DEPTH_8U, 1 );
	cvConvertImage(frame, frame2_1C);

	// Shi and Tomasi Feature Tracking! */

	// Preparation: Allocate the necessary storage. */
	allocateOnDemand( &eig_imageOrigen, frame_size, IPL_DEPTH_32F, 1 );
	allocateOnDemand( &temp_imageOrigen, frame_size, IPL_DEPTH_32F, 1 );

	allocateOnDemand( &eig_imageIzq, frame_size, IPL_DEPTH_32F, 1 );
	allocateOnDemand( &temp_imageIzq, frame_size, IPL_DEPTH_32F, 1 );

	allocateOnDemand( &eig_imageDer, frame_size, IPL_DEPTH_32F, 1 );
	allocateOnDemand( &temp_imageDer, frame_size, IPL_DEPTH_32F, 1 );

	// Preparation: This array will contain the features found in frame 1. 
	CvPoint2D32f frame1Origen_features[MAX_NUM_FEATURES_POR_ZONA];
	CvPoint2D32f frame1Izq_features[MAX_NUM_FEATURES_POR_ZONA];
	CvPoint2D32f frame1Der_features[MAX_NUM_FEATURES_POR_ZONA];

	// Preparation: BEFORE the function call this variable is the array size
	// (or the maximum number of features to find).  AFTER the function call
	// this variable is the number of features actually found.
	int number_of_featuresOrigen;
	int number_of_featuresIzq;
	int number_of_featuresDer;

	// I'm hardcoding this at 400.  But you should make this a #define so that you can
	// change the number of features you use for an accuracy/speed tradeoff analysis.
	//
	number_of_featuresOrigen = MAX_NUM_FEATURES_POR_ZONA;
	number_of_featuresIzq = MAX_NUM_FEATURES_POR_ZONA;
	number_of_featuresDer = MAX_NUM_FEATURES_POR_ZONA;

	//----------------Mascara para la zona de origen
	allocateOnDemand( &maskOrigen, frame_size, IPL_DEPTH_8U, 1 );
	cvSet(maskOrigen, cvScalar(0.0));
	int xIniOrigen = areaOrigen.x, yIniOrigen = areaOrigen.y, xFinOrigen = areaOrigen.x+areaOrigen.width, yFinOrigen = areaOrigen.y+areaOrigen.height;
	for(int i = xIniOrigen ; i<xFinOrigen ; i++)
	{
		for(int j = yIniOrigen ; j<yFinOrigen ; j++)
		{
			cvSet2D(maskOrigen, j, i, cvScalar(255.0));
		}
	}
	CvScalar line_color_origen;			
	line_color_origen = CV_RGB(0,255,0);
	CvPoint ptOrigen1, ptOrigen2;
	ptOrigen1.x = xIniOrigen;
	ptOrigen1.y = yIniOrigen;
	ptOrigen2.x = xFinOrigen;
	ptOrigen2.y = yFinOrigen;
	cvRectangle(frame1, ptOrigen1, ptOrigen2, line_color_origen, 1, CV_AA, 0 );

	//----------------Mascara para la zona de giro a la izquierda
	allocateOnDemand( &maskIzq, frame_size, IPL_DEPTH_8U, 1 );
	cvSet(maskIzq, cvScalar(0.0));
	int xIniIzq = areaGiroIzq.x, yIniIzq = areaGiroIzq.y, xFinIzq = areaGiroIzq.x+areaGiroIzq.width, yFinIzq = areaGiroIzq.y+areaGiroIzq.height;
	for(int i = xIniIzq ; i<xFinIzq ; i++)
	{
		for(int j = yIniIzq ; j<yFinIzq ; j++)
		{
			cvSet2D(maskIzq, j, i, cvScalar(255.0));
		}
	}
	CvScalar line_color_izq;			
	line_color_izq = CV_RGB(0,0,255);
	CvPoint ptIzq1, ptIzq2;
	ptIzq1.x = xIniIzq;
	ptIzq1.y = yIniIzq;
	ptIzq2.x = xFinIzq;
	ptIzq2.y = yFinIzq;
	cvRectangle(frame1, ptIzq1, ptIzq2, line_color_izq, 1, CV_AA, 0 );

	//----------------Mascara para la zona de giro a la derecha
	allocateOnDemand( &maskDer, frame_size, IPL_DEPTH_8U, 1 );
	cvSet(maskDer, cvScalar(0.0));
	int xIniDer = areaGiroDer.x, yIniDer = areaGiroDer.y, xFinDer = areaGiroDer.x+areaGiroDer.width, yFinDer = areaGiroDer.y+areaGiroDer.height;
	for(int i = xIniDer ; i<xFinDer ; i++)
	{
		for(int j = yIniDer ; j<yFinDer ; j++)
		{
			cvSet2D(maskDer, j, i, cvScalar(255.0));
		}
	}
	CvScalar line_color_der;			
	line_color_der = CV_RGB(0,255,255);
	CvPoint ptDer1, ptDer2;
	ptDer1.x = xIniDer;
	ptDer1.y = yIniDer;
	ptDer2.x = xFinDer;
	ptDer2.y = yFinDer;
	cvRectangle(frame1, ptDer1, ptDer2, line_color_der, 1, CV_AA, 0 );


	// Actually run the Shi and Tomasi algorithm!!
	// "frame1_1C" is the input image.
	// "eig_image" and "temp_image" are just workspace for the algorithm.
	// The first ".01" specifies the minimum quality of the features (based on the eigenvalues).
	// The second ".01" specifies the minimum Euclidean distance between features.
	// "NULL" means use the entire input image.  You could point to a part of the image.
	// WHEN THE ALGORITHM RETURNS:
	// "frame1_features" will contain the feature points.
	// "number_of_features" will be set to a value <= 400 indicating the number of feature points found.
	//
	cvGoodFeaturesToTrack(frame1_1C, eig_imageOrigen, temp_imageOrigen, frame1Origen_features, &number_of_featuresOrigen, .01, .01, maskOrigen);
	cvGoodFeaturesToTrack(frame1_1C, eig_imageIzq, temp_imageIzq, frame1Izq_features, &number_of_featuresIzq, .01, .01, maskIzq);
	cvGoodFeaturesToTrack(frame1_1C, eig_imageDer, temp_imageDer, frame1Der_features, &number_of_featuresDer, .01, .01, maskDer);

	/*Uniwer los 3 arreglo
	CvPoint2D32f frame1Todo_features[MAX_NUM_FEATURES_POR_ZONA*3];
	for (int i=0 ; i<(number_of_featuresOrigen+number_of_featuresIzq+number_of_featuresDer) ; i++)
	{
		if(i<number_of_featuresOrigen)
		{
			frame1Todo_features[i] = frame1Origen_features[i]; 
		}
		else if(i<number_of_featuresOrigen+number_of_featuresIzq)
		{
			frame1Todo_features[i] = frame1Izq_features[i-number_of_featuresOrigen]; 
		}
		else if(i<number_of_featuresOrigen+number_of_featuresIzq+number_of_featuresDer)
		{
			frame1Todo_features[i] = frame1Der_features[i-number_of_featuresOrigen-number_of_featuresIzq]; 
		}
	}
	*/

	// Pyramidal Lucas Kanade Optical Flow! */

	// This array will contain the locations of the points from frame 1 in frame 2. */
	CvPoint2D32f frame2Origen_features[MAX_NUM_FEATURES_POR_ZONA];
	CvPoint2D32f frame2Izq_features[MAX_NUM_FEATURES_POR_ZONA];
	CvPoint2D32f frame2Der_features[MAX_NUM_FEATURES_POR_ZONA];

	// The i-th element of this array will be non-zero if and only if the i-th feature of
	// frame 1 was found in frame 2.
	//
	char optical_flow_found_featureOrigen[MAX_NUM_FEATURES_POR_ZONA];
	char optical_flow_found_featureIzq[MAX_NUM_FEATURES_POR_ZONA];
	char optical_flow_found_featureDer[MAX_NUM_FEATURES_POR_ZONA];

	// The i-th element of this array is the error in the optical flow for the i-th feature
	// of frame1 as found in frame 2.  If the i-th feature was not found (see the array above)
	// I think the i-th entry in this array is undefined.
	//
	float optical_flow_feature_errorOrigen[MAX_NUM_FEATURES_POR_ZONA];
	float optical_flow_feature_errorIzq[MAX_NUM_FEATURES_POR_ZONA];
	float optical_flow_feature_errorDer[MAX_NUM_FEATURES_POR_ZONA];

	// This is the window size to use to avoid the aperture problem (see slide "Optical Flow: Overview"). */
	CvSize optical_flow_window = cvSize(tamanoVentana,tamanoVentana);

	// This termination criteria tells the algorithm to stop when it has either done 20 iterations or when
	// epsilon is better than .3.  You can play with these parameters for speed vs. accuracy but these values
	// work pretty well in many situations.
	//
	CvTermCriteria optical_flow_termination_criteria
		= cvTermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, maxIter, epsilon );

	// This is some workspace for the algorithm.
	// (The algorithm actually carves the image into pyramids of different resolutions.)
	//
	allocateOnDemand( &pyramid1Origen, frame_size, IPL_DEPTH_8U, 1 );
	allocateOnDemand( &pyramid2Origen, frame_size, IPL_DEPTH_8U, 1 );

	allocateOnDemand( &pyramid1Izq, frame_size, IPL_DEPTH_8U, 1 );
	allocateOnDemand( &pyramid2Izq, frame_size, IPL_DEPTH_8U, 1 );

	allocateOnDemand( &pyramid1Der, frame_size, IPL_DEPTH_8U, 1 );
	allocateOnDemand( &pyramid2Der, frame_size, IPL_DEPTH_8U, 1 );

	// Actually run Pyramidal Lucas Kanade Optical Flow!!
	// "frame1_1C" is the first frame with the known features.
	// "frame2_1C" is the second frame where we want to find the first frame's features.
	// "pyramid1" and "pyramid2" are workspace for the algorithm.
	// "frame1_features" are the features from the first frame.
	// "frame2_features" is the (outputted) locations of those features in the second frame.
	// "number_of_features" is the number of features in the frame1_features array.
	// "optical_flow_window" is the size of the window to use to avoid the aperture problem.
	// "5" is the maximum number of pyramids to use.  0 would be just one level.
	// "optical_flow_found_feature" is as described above (non-zero iff feature found by the flow).
	// "optical_flow_feature_error" is as described above (error in the flow for this feature).
	// "optical_flow_termination_criteria" is as described above (how long the algorithm should look).
	// "0" means disable enhancements.  (For example, the second array isn't pre-initialized with guesses.)
	//
	cvCalcOpticalFlowPyrLK(frame1_1C, frame2_1C, pyramid1Origen, pyramid2Origen, frame1Origen_features, frame2Origen_features, number_of_featuresOrigen, optical_flow_window, 5, optical_flow_found_featureOrigen, optical_flow_feature_errorOrigen, optical_flow_termination_criteria, 0 );
	cvCalcOpticalFlowPyrLK(frame1_1C, frame2_1C, pyramid1Izq, pyramid2Izq, frame1Izq_features, frame2Izq_features, number_of_featuresIzq, optical_flow_window, 5, optical_flow_found_featureIzq, optical_flow_feature_errorIzq, optical_flow_termination_criteria, 0 );
	cvCalcOpticalFlowPyrLK(frame1_1C, frame2_1C, pyramid1Der, pyramid2Der, frame1Der_features, frame2Der_features, number_of_featuresDer, optical_flow_window, 5, optical_flow_found_featureDer, optical_flow_feature_errorDer, optical_flow_termination_criteria, 0 );
	//cout << (number_of_featuresOrigen+number_of_featuresIzq+number_of_featuresDer) << "\n";

	//------------------------Zona origen-----------------
	std::vector<Point> posicionesCarrosOrigen;
	bool puntosAPintarDiferenteOrigen[MAX_NUM_FEATURES_POR_ZONA];
	for(int r = 0; r<MAX_NUM_FEATURES_POR_ZONA ; r++)
		puntosAPintarDiferenteOrigen[r] = false;
	bool puntosMarcadosOrigen[MAX_NUM_FEATURES_POR_ZONA*3];
	for(int r = 0; r<MAX_NUM_FEATURES_POR_ZONA ; r++)
		puntosMarcadosOrigen[r] = false;
	for(int i = 0; i < number_of_featuresOrigen; i++)
	{

		std::vector<Point> puntos;
		std::vector<int> puntosPintar;
		//ZONA 0
		puntosCercanosAlineados(&puntos, &puntosPintar, i, frame1Origen_features, frame2Origen_features, optical_flow_found_featureOrigen, optical_flow_feature_errorOrigen, puntosMarcadosOrigen, number_of_featuresOrigen, 0 );

		//criterio segun zona
		if(puntos.size()>=minPuntosCarro)
		{
			Point puntoProm;
			for(int k = 0 ; k<puntos.size(); k++)
			{
				puntoProm.x += puntos[k].x;
				puntoProm.y += puntos[k].y;
				puntosAPintarDiferenteOrigen[puntosPintar[k]] = true;
			}

			puntoProm.x = puntoProm.x/puntos.size();
			puntoProm.y = puntoProm.y/puntos.size();

			posicionesCarrosOrigen.push_back(puntoProm);
			cvCircle(frame1, puntoProm, 10, line_color_der, 10, CV_AA, 0);
		}
	}
	// For fun (and debugging :)), let's draw the flow field. */
	for(int i = 0; i < number_of_featuresOrigen; i++)
	{
		// If Pyramidal Lucas Kanade didn't really find the feature, skip it. */
		if ( optical_flow_found_featureOrigen[i] == 0 )	continue;
		if ( optical_flow_feature_errorOrigen[i] > maxError )	continue;
		//cout << optical_flow_feature_error[i] << "\n";

		int line_thickness;				line_thickness = 1;
		// CV_RGB(red, green, blue) is the red, green, and blue components
		// of the color you want, each out of 255.
		//	
		CvScalar line_color;
		if(puntosAPintarDiferenteOrigen[i])
		{
			line_color = CV_RGB(255,255,255);
		}
		else
		{
			line_color = CV_RGB(255,0,0);
		}
		// Let's make the flow field look nice with arrows. */

		// The arrows will be a bit too short for a nice visualization because of the high framerate
		// (ie: there's not much motion between the frames).  So let's lengthen them by a factor of 3.
		//
		CvPoint p,q;
		p.x = (int) frame1Origen_features[i].x;
		p.y = (int) frame1Origen_features[i].y;
		q.x = (int) frame2Origen_features[i].x;
		q.y = (int) frame2Origen_features[i].y;

		Point pNuevo = p;
		Point qNuevo = q;

		Point vec = qNuevo - pNuevo;

		if(norm(vec) < minNorm) 
			continue;
		if(norm(vec) > maxNorm) 
			continue;

		double angle;		angle = atan2( (double) p.y - q.y, (double) p.x - q.x );
		double hypotenuse;	hypotenuse = sqrt( square(p.y - q.y) + square(p.x - q.x) );

		// Here we lengthen the arrow by a factor of three. */
		q.x = (int) (p.x - 3 * hypotenuse * cos(angle));
		q.y = (int) (p.y - 3 * hypotenuse * sin(angle));

		// Now we draw the main line of the arrow. */
		// "frame1" is the frame to draw on.
		// "p" is the point where the line begins.
		// "q" is the point where the line stops.
		// "CV_AA" means antialiased drawing.
		// "0" means no fractional bits in the center cooridinate or radius.
		//
		cvLine( frame1, p, q, line_color, line_thickness, CV_AA, 0 );
		// Now draw the tips of the arrow.  I do some scaling so that the
		// tips look proportional to the main line of the arrow.
		//			
		p.x = (int) (q.x + 9 * cos(angle + PI / 4));
		p.y = (int) (q.y + 9 * sin(angle + PI / 4));
		cvLine( frame1, p, q, line_color, line_thickness, CV_AA, 0 );
		p.x = (int) (q.x + 9 * cos(angle - PI / 4));
		p.y = (int) (q.y + 9 * sin(angle - PI / 4));
		cvLine( frame1, p, q, line_color, line_thickness, CV_AA, 0 );
	}
	

	for(int i = 0 ; i<carrosTrack.size(); i++)
	{
		Point p = carrosTrack[i];
		for(int j = 0 ; j<posicionesCarrosOrigen.size(); j++)
		{
			Point q = posicionesCarrosOrigen[j];

			Point vec = q - p;

			if(norm(vec) < radioBusqueda) 
			{
				carrosTrack[i] = q;
				posicionesCarrosOrigen.erase(j);
			}
		}
	}

	for(int i = 0 ; i<carrosTrack.size(); i++)
	{
		Point p = carrosTrack[i];
		for(int j = 0 ; j<posicionesCarrosOrigen.size(); j++)
		{
			Point q = posicionesCarrosOrigen[j];

			Point vec = q - p;

			if(norm(vec) < radioBusqueda) 
			{
				carrosTrack[i] = q;
				posicionesCarrosOrigen.erase(j);
			}
		}
	}






	//------------------------Zona Izquierda-----------------
	std::vector<Point> posicionesCarrosIzq;
	bool puntosAPintarDiferenteIzq[MAX_NUM_FEATURES_POR_ZONA];
	for(int r = 0; r<MAX_NUM_FEATURES_POR_ZONA ; r++)
		puntosAPintarDiferenteIzq[r] = false;
	bool puntosMarcadosIzq[MAX_NUM_FEATURES_POR_ZONA*3];
	for(int r = 0; r<MAX_NUM_FEATURES_POR_ZONA ; r++)
		puntosMarcadosIzq[r] = false;
	for(int i = 0; i < number_of_featuresIzq; i++)
	{

		std::vector<Point> puntos;
		std::vector<int> puntosPintar;
		//ZONA 0
		puntosCercanosAlineados(&puntos, &puntosPintar, i, frame1Izq_features, frame2Izq_features, optical_flow_found_featureIzq, optical_flow_feature_errorIzq, puntosMarcadosIzq, number_of_featuresIzq, 1 );

		//criterio segun zona
		if(puntos.size()>=minPuntosCarro_izq)
		{
			Point puntoProm;
			for(int k = 0 ; k<puntos.size(); k++)
			{
				puntoProm.x += puntos[k].x;
				puntoProm.y += puntos[k].y;
				puntosAPintarDiferenteIzq[puntosPintar[k]] = true;
			}

			puntoProm.x = puntoProm.x/puntos.size();
			puntoProm.y = puntoProm.y/puntos.size();

			posicionesCarrosIzq.push_back(puntoProm);
			cvCircle(frame1, puntoProm, 10, line_color_der, 10, CV_AA, 0);
		}
	}
	// For fun (and debugging :)), let's draw the flow field. */
	for(int i = 0; i < number_of_featuresIzq; i++)
	{
		// If Pyramidal Lucas Kanade didn't really find the feature, skip it. */
		if ( optical_flow_found_featureIzq[i] == 0 )	continue;
		if ( optical_flow_feature_errorIzq[i] > maxError )	continue;
		//cout << optical_flow_feature_error[i] << "\n";

		int line_thickness;				line_thickness = 1;
		// CV_RGB(red, green, blue) is the red, green, and blue components
		// of the color you want, each out of 255.
		//	
		CvScalar line_color;
		if(puntosAPintarDiferenteIzq[i])
		{
			line_color = CV_RGB(255,255,255);
		}
		else
		{
			line_color = CV_RGB(255,0,0);
		}
		// Let's make the flow field look nice with arrows. */

		// The arrows will be a bit too short for a nice visualization because of the high framerate
		// (ie: there's not much motion between the frames).  So let's lengthen them by a factor of 3.
		//
		CvPoint p,q;
		p.x = (int) frame1Izq_features[i].x;
		p.y = (int) frame1Izq_features[i].y;
		q.x = (int) frame2Izq_features[i].x;
		q.y = (int) frame2Izq_features[i].y;

		Point pNuevo = p;
		Point qNuevo = q;

		Point vec = qNuevo - pNuevo;

		if(norm(vec) < minNorm) 
			continue;
		if(norm(vec) > maxNorm) 
			continue;

		double angle;		angle = atan2( (double) p.y - q.y, (double) p.x - q.x );
		double hypotenuse;	hypotenuse = sqrt( square(p.y - q.y) + square(p.x - q.x) );

		// Here we lengthen the arrow by a factor of three. */
		q.x = (int) (p.x - 3 * hypotenuse * cos(angle));
		q.y = (int) (p.y - 3 * hypotenuse * sin(angle));

		// Now we draw the main line of the arrow. */
		// "frame1" is the frame to draw on.
		// "p" is the point where the line begins.
		// "q" is the point where the line stops.
		// "CV_AA" means antialiased drawing.
		// "0" means no fractional bits in the center cooridinate or radius.
		//
		cvLine( frame1, p, q, line_color, line_thickness, CV_AA, 0 );
		// Now draw the tips of the arrow.  I do some scaling so that the
		// tips look proportional to the main line of the arrow.
		//			
		p.x = (int) (q.x + 9 * cos(angle + PI / 4));
		p.y = (int) (q.y + 9 * sin(angle + PI / 4));
		cvLine( frame1, p, q, line_color, line_thickness, CV_AA, 0 );
		p.x = (int) (q.x + 9 * cos(angle - PI / 4));
		p.y = (int) (q.y + 9 * sin(angle - PI / 4));
		cvLine( frame1, p, q, line_color, line_thickness, CV_AA, 0 );
	}

	//------------------------Zona Derecha-----------------
	std::vector<Point> posicionesCarrosDer;
	bool puntosAPintarDiferenteDer[MAX_NUM_FEATURES_POR_ZONA];
	for(int r = 0; r<MAX_NUM_FEATURES_POR_ZONA ; r++)
		puntosAPintarDiferenteDer[r] = false;
	bool puntosMarcadosDer[MAX_NUM_FEATURES_POR_ZONA*3];
	for(int r = 0; r<MAX_NUM_FEATURES_POR_ZONA ; r++)
		puntosMarcadosDer[r] = false;
	for(int i = 0; i < number_of_featuresDer; i++)
	{

		std::vector<Point> puntos;
		std::vector<int> puntosPintar;
		//ZONA 0
		puntosCercanosAlineados(&puntos, &puntosPintar, i, frame1Der_features, frame2Der_features, optical_flow_found_featureDer, optical_flow_feature_errorDer, puntosMarcadosDer, number_of_featuresDer, 2 );

		//criterio segun zona
		if(puntos.size()>=minPuntosCarro_der)
		{
			Point puntoProm;
			for(int k = 0 ; k<puntos.size(); k++)
			{
				puntoProm.x += puntos[k].x;
				puntoProm.y += puntos[k].y;
				puntosAPintarDiferenteDer[puntosPintar[k]] = true;
			}

			puntoProm.x = puntoProm.x/puntos.size();
			puntoProm.y = puntoProm.y/puntos.size();

			posicionesCarrosDer.push_back(puntoProm);
			cvCircle(frame1, puntoProm, 10, line_color_der, 10, CV_AA, 0);
		}
	}
	// For fun (and debugging :)), let's draw the flow field. */
	for(int i = 0; i < number_of_featuresDer; i++)
	{
		// If Pyramidal Lucas Kanade didn't really find the feature, skip it. */
		if ( optical_flow_found_featureDer[i] == 0 )	continue;
		if ( optical_flow_feature_errorDer[i] > maxError )	continue;
		//cout << optical_flow_feature_error[i] << "\n";

		int line_thickness;				line_thickness = 1;
		// CV_RGB(red, green, blue) is the red, green, and blue components
		// of the color you want, each out of 255.
		//	
		CvScalar line_color;
		if(puntosAPintarDiferenteDer[i])
		{
			line_color = CV_RGB(255,255,255);
		}
		else
		{
			line_color = CV_RGB(255,0,0);
		}
		// Let's make the flow field look nice with arrows. */

		// The arrows will be a bit too short for a nice visualization because of the high framerate
		// (ie: there's not much motion between the frames).  So let's lengthen them by a factor of 3.
		//
		CvPoint p,q;
		p.x = (int) frame1Der_features[i].x;
		p.y = (int) frame1Der_features[i].y;
		q.x = (int) frame2Der_features[i].x;
		q.y = (int) frame2Der_features[i].y;

		Point pNuevo = p;
		Point qNuevo = q;

		Point vec = qNuevo - pNuevo;

		if(norm(vec) < minNorm) 
			continue;
		if(norm(vec) > maxNorm) 
			continue;

		double angle;		angle = atan2( (double) p.y - q.y, (double) p.x - q.x );
		double hypotenuse;	hypotenuse = sqrt( square(p.y - q.y) + square(p.x - q.x) );

		// Here we lengthen the arrow by a factor of three. */
		q.x = (int) (p.x - 3 * hypotenuse * cos(angle));
		q.y = (int) (p.y - 3 * hypotenuse * sin(angle));

		// Now we draw the main line of the arrow. */
		// "frame1" is the frame to draw on.
		// "p" is the point where the line begins.
		// "q" is the point where the line stops.
		// "CV_AA" means antialiased drawing.
		// "0" means no fractional bits in the center cooridinate or radius.
		//
		cvLine( frame1, p, q, line_color, line_thickness, CV_AA, 0 );
		// Now draw the tips of the arrow.  I do some scaling so that the
		// tips look proportional to the main line of the arrow.
		//			
		p.x = (int) (q.x + 9 * cos(angle + PI / 4));
		p.y = (int) (q.y + 9 * sin(angle + PI / 4));
		cvLine( frame1, p, q, line_color, line_thickness, CV_AA, 0 );
		p.x = (int) (q.x + 9 * cos(angle - PI / 4));
		p.y = (int) (q.y + 9 * sin(angle - PI / 4));
		cvLine( frame1, p, q, line_color, line_thickness, CV_AA, 0 );
	}



	// Now display the image we drew on.  Recall that "Optical Flow" is the name of
	// the window we created above.
	//
	cvShowImage("Optical Flow", frame1);


	//--------------------probando marcacion en pantalla
	Mat image = frame1; 
	//Mat image = cvarrToMat(frame1).clone();
	if( selectObject && selection.width > 0 && selection.height > 0 )
	{
		Mat roi(image, selection);
		bitwise_not(roi, roi);
	}

	imshow( "Optical Flow", image );

	/* And wait for the user to press a key (so the user has time to look at the image).
	* If the argument is 0 then it waits forever otherwise it waits that number of milliseconds.
	* The return value is the key the user pressed.
	*/
	int key_pressed;
	key_pressed = cvWaitKey(1);

	/* If the users pushes "b" or "B" go back one frame.
	* Otherwise go forward one frame.
	*/
	if (key_pressed == 'b' || key_pressed == 'B')	current_frame--;
	else											current_frame++;
	/* Don't run past the front/end of the AVI. */
	if (current_frame < 0)						current_frame = 0;
	if (current_frame >= number_of_frames - 1)	current_frame = number_of_frames - 2;
}

int canny(long number_of_frames, CvSize frame_size, CvCapture* input_video)
{
	static IplImage *frame = NULL, *frame1 = NULL, *frame1_1C = NULL, *frame2_1C = NULL, *eig_image = NULL, *temp_image = NULL, *pyramid1 = NULL, *pyramid2 = NULL;
	//mascaras
	static IplImage *maskOrigen = NULL, *maskIzq = NULL, *maskDer = NULL;		

	
	cvSetCaptureProperty( input_video, CV_CAP_PROP_POS_FRAMES, current_frame );


	frame = cvQueryFrame( input_video );
	if (frame == NULL)
	{
		// Why did we get a NULL frame?  We shouldn't be at the end. */
		fprintf(stderr, "Error: Hmm. The end came sooner than we thought.\n");
		return -1;
	}
	

	allocateOnDemand( &frame1_1C, frame_size, IPL_DEPTH_8U, 1 );

	
	cvConvertImage(frame, frame1_1C);

	
	allocateOnDemand( &frame1, frame_size, IPL_DEPTH_8U, 3 );
	cvConvertImage(frame, frame1);

	
	frame = cvQueryFrame( input_video );
	if (frame == NULL)
	{
		fprintf(stderr, "Error: Hmm. The end came sooner than we thought.\n");
		return -1;
	}
	allocateOnDemand( &frame2_1C, frame_size, IPL_DEPTH_8U, 1 );
	cvConvertImage(frame, frame2_1C);

	// Shi and Tomasi Feature Tracking! */

	// Preparation: Allocate the necessary storage. */
	allocateOnDemand( &eig_image, frame_size, IPL_DEPTH_32F, 1 );
	allocateOnDemand( &temp_image, frame_size, IPL_DEPTH_32F, 1 );




	cvCanny(frame1_1C, frame2_1C, threshold1, threshold2, apertureSize );

	Mat image = frame2_1C; 
	imshow( "Optical Flow", image );

	/* And wait for the user to press a key (so the user has time to look at the image).
	* If the argument is 0 then it waits forever otherwise it waits that number of milliseconds.
	* The return value is the key the user pressed.
	*/
	int key_pressed;
	key_pressed = cvWaitKey(1);

	/* If the users pushes "b" or "B" go back one frame.
	* Otherwise go forward one frame.
	*/
	if (key_pressed == 'b' || key_pressed == 'B')	current_frame--;
	else											current_frame++;
	/* Don't run past the front/end of the AVI. */
	if (current_frame < 0)						current_frame = 0;
	if (current_frame >= number_of_frames - 1)	current_frame = number_of_frames - 2;

}

int restar(long number_of_frames, CvSize frame_size, CvCapture* input_video, int* valoresPlot)
{
	static IplImage *frame = NULL, *frame1 = NULL, *frame1_1C = NULL, *frame2_1C = NULL, *eig_image = NULL, *temp_image = NULL, *pyramid1 = NULL, *pyramid2 = NULL;
	//mascaras
	static IplImage *maskOrigen = NULL, *maskIzq = NULL, *maskDer = NULL;		

	static IplImage *outImg = NULL;		

	
	cvSetCaptureProperty( input_video, CV_CAP_PROP_POS_FRAMES, current_frame );


	frame = cvQueryFrame( input_video );
	if (frame == NULL)
	{
		// Why did we get a NULL frame?  We shouldn't be at the end. */
		fprintf(stderr, "Error: Hmm. The end came sooner than we thought.\n");
		return -1;
	}
	

	allocateOnDemand( &frame1_1C, frame_size, IPL_DEPTH_8U, 1 );

	allocateOnDemand( &outImg, frame_size, IPL_DEPTH_8U, 1 );

	
	cvConvertImage(frame, frame1_1C);

	
	allocateOnDemand( &frame1, frame_size, IPL_DEPTH_8U, 3 );
	cvConvertImage(frame, frame1);

	
	frame = cvQueryFrame( input_video );
	if (frame == NULL)
	{
		fprintf(stderr, "Error: Hmm. The end came sooner than we thought.\n");
		return -1;
	}
	allocateOnDemand( &frame2_1C, frame_size, IPL_DEPTH_8U, 1 );
	cvConvertImage(frame, frame2_1C);

	// Shi and Tomasi Feature Tracking! */

	// Preparation: Allocate the necessary storage. */
	allocateOnDemand( &eig_image, frame_size, IPL_DEPTH_32F, 1 );
	allocateOnDemand( &temp_image, frame_size, IPL_DEPTH_32F, 1 );




	cvAbsDiff( frame1_1C, frame2_1C, outImg );

	Mat frame2_1C_mat = frame2_1C;
	Scalar suma = sum(frame2_1C_mat);



	const int sumaplot = suma[0];
	int width = (frame_size.height*frame_size.width*255)/2;

	static int pos;
	valoresPlot[pos] = sumaplot;
	pos++;

	cout << pos << "\n";

	if (pos >= numFramesPlot-1)
	{
		for(int r = 0; r<numFramesPlot ; r++)
			valoresPlot[r] = 0;
		pos=0;
	}

	CvPlot::plot("RGB", valoresPlot, numFramesPlot, 1, 255, 0, 0);
	CvPlot::label("B");
	
	CvFont font;
	cvInitFont(&font,CV_FONT_HERSHEY_PLAIN,0.55,0.7, 0,1,CV_AA);


	int number  = 5;
	char text[255]; 
	sprintf(text, "suma: %d", (int)sumaplot);

	cvPutText(outImg, text,  cvPoint(30,30), &font, cvScalar(255,255,0));


	Mat image = outImg; 
	imshow( "Optical Flow", image );

	/* And wait for the user to press a key (so the user has time to look at the image).
	* If the argument is 0 then it waits forever otherwise it waits that number of milliseconds.
	* The return value is the key the user pressed.
	*/
	int key_pressed;
	key_pressed = cvWaitKey(0);

	/* If the users pushes "b" or "B" go back one frame.
	* Otherwise go forward one frame.
	*/
	if (key_pressed == 'b' || key_pressed == 'B')	current_frame--;
	else											current_frame++;
	/* Don't run past the front/end of the AVI. */
	if (current_frame < 0)						current_frame = 0;
	if (current_frame >= number_of_frames - 1)	current_frame = number_of_frames - 2;


	CvPlot::clear("RGB");
}

int main(void)
{
	/* Create an object that decodes the input video stream. */
	CvCapture *input_video = cvCaptureFromFile(
		"C:\\Users\\Nicolas\\Desktop\\Videos Imagenes y vi\\GOPR4426.MP4"
		);
	if (input_video == NULL)
	{
		/* Either the video didn't exist OR it uses a codec OpenCV
		* doesn't support.
		*/
		fprintf(stderr, "Error: Can't open video.\n");
		return -1;
	}

	/* Read the video's frame size out of the AVI. */
	CvSize frame_size;
	frame_size.height =
		(int) cvGetCaptureProperty( input_video, CV_CAP_PROP_FRAME_HEIGHT );
	frame_size.width =
		(int) cvGetCaptureProperty( input_video, CV_CAP_PROP_FRAME_WIDTH );

	/* Determine the number of frames in the AVI. */
	long number_of_frames;
	/* Go to the end of the AVI (ie: the fraction is "1") */
	cvSetCaptureProperty( input_video, CV_CAP_PROP_POS_AVI_RATIO, 1. );
	/* Now that we're at the end, read the AVI position in frames */
	number_of_frames = (int) cvGetCaptureProperty( input_video, CV_CAP_PROP_POS_FRAMES );
	/* Return to the beginning */
	cvSetCaptureProperty( input_video, CV_CAP_PROP_POS_FRAMES, 0. );

	/* Create a windows called "Optical Flow" for visualizing the output.
	* Have the window automatically change its size to match the output.
	*/
	namedWindow("Optical Flow", CV_WINDOW_AUTOSIZE);

	//------Ventana para los controles de la aplicacion
	namedWindow("Controls", CV_WINDOW_NORMAL);
	resizeWindow("Controls", 500, 400);
	createTrackbar( "Min Norm", "Controls", &minNorm, 200, NULL);
	createTrackbar( "Max Norm", "Controls", &maxNorm, 500, NULL);
	createTrackbar( "Max Iter", "Controls", &maxIter, 100, NULL);
	createTrackbar( "epsilon", "Controls", &epsilon_trackbar, 1000, on_trackbar_epsilon);
	createTrackbar( "Max Error", "Controls", &maxError_trackbar, 1000, on_trackbar_error);
	createTrackbar( "Video", "Controls", &frame_video_trackbar, number_of_frames, on_trackbar_frame_video);
	createTrackbar( "Kernel", "Controls", &tamanoVentana, 30, NULL);
	createTrackbar( "MaxDistPuntos", "Controls", &maxDistPuntosCarro_trackbar, 10000, on_trackbar_maxDistPuntosCarro);
	createTrackbar( "MinNumPuntos", "Controls", &minPuntosCarro, 100, NULL);
	createTrackbar( "MaxAngPuntos", "Controls", &maxAnguloPuntosCarro_trackbar, 300, on_trackbar_maxDistPuntosCarro);

	//------Ventana para controles de Canny
	/*namedWindow("Controls Canny", CV_WINDOW_NORMAL);
	resizeWindow("Controls", 500, 400);
	createTrackbar( "Thres1", "Controls Canny", &threshold1, 255, NULL);
	createTrackbar( "Thres2", "Controls Canny", &threshold1, 255, NULL);
	createTrackbar( "Aperture", "Controls Canny", &apertureSize, 50, NULL);
	*/

	//------Ventana para controles de giro izquierda
	namedWindow("Controles Giro Izquierda", CV_WINDOW_NORMAL);
	createTrackbar( "MaxDistPuntos", "Controles Giro Izquierda", &maxDistPuntosCarro_izq_trackbar, 10000, on_trackbar_maxDistPuntosCarro_izq);
	createTrackbar( "MinNumPuntos", "Controles Giro Izquierda", &minPuntosCarro_izq, 100, NULL);
	createTrackbar( "MaxAngPuntos", "Controles Giro Izquierda", &maxAnguloPuntosCarro_izq_trackbar, 300, on_trackbar_maxDistPuntosCarro_izq);

	//------Ventana para controles de giro derecha
	namedWindow("Controles Giro Derecha", CV_WINDOW_NORMAL);
	createTrackbar( "MaxDistPuntos", "Controles Giro Derecha", &maxDistPuntosCarro_der_trackbar, 10000, on_trackbar_maxDistPuntosCarro_der);
	createTrackbar( "MinNumPuntos", "Controles Giro Derecha", &minPuntosCarro_der, 100, NULL);
	createTrackbar( "MaxAngPuntos", "Controles Giro Derecha", &maxAnguloPuntosCarro_der_trackbar, 300, on_trackbar_maxDistPuntosCarro_der);
	

	setMouseCallback( "Optical Flow", onMouse, 0 );

	current_frame = 0;

	int valoresPlot[numFramesPlot];
	for(int r = 0; r<numFramesPlot ; r++)
		valoresPlot[r] = 0;

	
	while(true)
	{


		//restar(number_of_frames, frame_size, input_video, valoresPlot);
		nosotros(number_of_frames, frame_size, input_video);


	}
}

