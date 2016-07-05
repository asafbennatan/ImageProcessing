#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include "Cluster.h"
#include <algorithm> 
#include <map>

#include <unordered_set>
using namespace std;
using namespace cv;

vector<uchar> getIdentificationVector(Mat image, int gridWidth, int gridHeight);
Vec<uchar, 128> calculateGrid(Mat cell);
double compare(Vec<uchar,128> a, Vec<uchar,128> b);
double euclideanDist(Point2f& p, Point2f& q);
pair<double, unordered_set<Cluster *>> buildClusters(Mat img,double vectorDirectionThreshold,double optionalDistance);
double joinClustersFixedPoint(unordered_set<Cluster *>& clusters, double vectorDirectionThreshold, double optionalDistance);
void initClusters(vector<KeyPoint> keypoints, Mat descriptors, unordered_set<Cluster*>& clusters);
bool mergable(Cluster * a,Cluster *b,double vectorDirectionThreshold, double optionalDistance);
//double * calculateDirection(Vec<uchar, 128> v);
pair<double,double> calculateDirection(Vec<uchar, 128> v);
void findSimilars(vector<Cluster *> clusters, Cluster * c,vector<Cluster*>& similars, double threshold);
double getMinimalDistanceBetweenTriplets(Cluster * origin, Cluster * other);
void calculateTrigularityList(unordered_set<Cluster *> clusters, Size imgSize);
Cluster *findCluster(Point2f p, unordered_set<Cluster*> clusters);
bool compareFloats(float a, float b);