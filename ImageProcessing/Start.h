using namespace std;
using namespace cv;

vector<uchar> getIdentificationVector(Mat image, int gridWidth, int gridHeight);
Vec<uchar, 128> calculateGrid(Mat cell);
double compare(Vec<uchar,128> a, Vec<uchar,128> b);
double euclideanDist(Point& p, Point& q);
vector<Cluster *> joinClusters(vector<Cluster *> clusters, double distance);
vector<Cluster *> joinClustersFixedPoint(vector<Cluster *> clusters, double distance);
vector<Cluster *> initClusters(vector<KeyPoint> keypoints, Mat descriptors);
bool sameDirection(Vec<uchar, 128> v1, Vec<uchar, 128> v2);
int calculateDirection(Vec<uchar, 128> v);
vector<Cluster * > findSimilars(vector<Cluster *> clusters, Cluster * c, double threshold);
vector<Cluster *> buildClusters(Mat img, double distance);