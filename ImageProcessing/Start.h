#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include "Cluster.h"
#include <algorithm>   

#include <unordered_set>
using namespace std;
using namespace cv;

vector<uchar> getIdentificationVector(Mat image, int gridWidth, int gridHeight);
Vec<uchar, 128> calculateGrid(Mat cell);
double compare(Vec<uchar,128> a, Vec<uchar,128> b);
double euclideanDist(Point2f& p, Point2f& q);
vector<Cluster *> joinClusters(vector<Cluster *>& clusters, double distance);
pair<double, vector<Cluster *>> buildClusters(Mat img, int numberOfClusters);
double joinClustersFixedPoint(vector<Cluster *>& clusters, int numberOfClusters);
double joinClusters(vector<Cluster *>& clusters);
vector<Cluster *> joinClustersFixedPoint(vector<Cluster *>& clusters, double distance);
void initClusters(vector<KeyPoint> keypoints, Mat descriptors, vector<Cluster*>& clusters);
bool sameDirection(Vec<uchar, 128> v1, Vec<uchar, 128> v2);
int calculateDirection(Vec<uchar, 128> v);
void findSimilars(vector<Cluster *> clusters, Cluster * c,vector<Cluster*>& similars, double threshold);
vector<Cluster *> buildClusters(Mat img, double distance);
double getMinimalDistanceBetweenTriplets(Cluster * origin, Cluster * other);
void calculateNeighbours(vector<Cluster *> clusters);
void setNearestTwoNeighbours(vector<Cluster *> clusters, Cluster *c);
