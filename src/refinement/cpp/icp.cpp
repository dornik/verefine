/* Author: Dominik Bauer
 * Vision for Robotics Group, Automation and Control Institute (ACIN)
 * TU Wien, Vienna */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/common.h>
#include <pcl/registration/icp.h>
#include <pcl/recognition/ransac_based/auxiliary.h>
#include <pcl/recognition/ransac_based/trimmed_icp.h>

using namespace pcl;

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;


PointCloudT::Ptr cloud_to_pcd(double* points, unsigned long num_points)
{
    PointCloudT in_cloud;
    for (unsigned long i = 0; i < num_points; i++)
    {
        unsigned long point_idx = i*3;
        PointT point;
        point.x = (float) points[point_idx];
        point.y = (float) points[point_idx+1];
        point.z = (float) points[point_idx+2];
        in_cloud.push_back(point);
    }
    PointCloudT::Ptr cloud = boost::shared_ptr<PointCloudT>(new PointCloudT(in_cloud));
    return cloud;
}

Eigen::Matrix4f performTrICP(double* points_obs, unsigned long num_points_obs,
                  double* points_ren, unsigned long num_points_ren,
                  float trim)
{
    // parse input buffer to point cloud
    PointCloudT::Ptr cloud_obs = cloud_to_pcd(points_obs, num_points_obs);
    PointCloudT::Ptr cloud_ren = cloud_to_pcd(points_ren, num_points_ren);

    // initialize trimmed icp
    pcl::recognition::TrimmedICP<PointT, float> tricp;
    tricp.init(cloud_ren);
    tricp.setNewToOldEnergyRatio(1.f);

    // compute trafo
    Eigen::Matrix4f tform;
    tform.setIdentity();
    float num_points = trim * num_points_obs;
    tricp.align(*cloud_obs, abs(num_points), tform);

    return tform.inverse();
}

Eigen::Matrix4f performICP(double* points_obs, unsigned long num_points_obs,
                  double* points_ren, unsigned long num_points_ren,
                  unsigned int max_iterations, float p_correspondence_distance)
{
    // parse input buffer to point cloud
    PointCloudT::Ptr cloud_obs = cloud_to_pcd(points_obs, num_points_obs);
    PointCloudT::Ptr cloud_ren = cloud_to_pcd(points_ren, num_points_ren);

    Eigen::Vector4f min;
    Eigen::Vector4f max;
    pcl::getMinMax3D (*cloud_obs, min, max);
    Eigen::Vector4f size = max-min;
    float max_size = size.maxCoeff();

    // initialize ICP
    pcl::IterativeClosestPoint<PointT, PointT> icp;
    icp.setMaxCorrespondenceDistance (p_correspondence_distance * max_size);
    icp.setMaximumIterations(max_iterations);
    icp.setTransformationEpsilon (1e-6);
    icp.setEuclideanFitnessEpsilon (1);
    icp.setRANSACOutlierRejectionThreshold(1.5);

    icp.setInputCloud(cloud_obs);
    icp.setInputTarget(cloud_ren);

    // compute trafo
    PointCloudT cloud_fit;
    icp.align(cloud_fit);
    Eigen::Matrix4f tform = icp.getFinalTransformation();

    return tform.inverse();
}

PYBIND11_MODULE(icp, m) {
    m.def("tricp", [](py::array_t<double> points_obs_array,
                      py::array_t<double> points_ren_array,
                      float trim)
    {
        // input
        auto points_obs_buf = points_obs_array.request();
        if (points_obs_buf.ndim != 2)
            throw std::runtime_error("Expect ren array of shape (num_points, XYZ).");
        if (points_obs_buf.shape[1] != 3)
            throw std::runtime_error("Expect ren array of shape (num_points, XYZ).");
        unsigned long num_points_obs = (unsigned long) points_obs_buf.shape[0];
        auto points_obs = (double*) points_obs_buf.ptr;

        auto points_ren_buf = points_ren_array.request();
        if (points_ren_buf.ndim != 2)
            throw std::runtime_error("Expect obs array of shape (num_points, XYZ).");
        if (points_ren_buf.shape[1] != 3)
            throw std::runtime_error("Expect obs array of shape (num_points, XYZ).");

        unsigned long num_points_ren = (unsigned long) points_ren_buf.shape[0];
        auto points_ren = (double*) points_ren_buf.ptr;

        // compute segmentation
        Eigen::Matrix4f T = performTrICP(points_obs, num_points_obs,
                                         points_ren, num_points_ren,
                                         trim);
        T.transposeInPlace(); // s.t. it is correctly transferred to numpy

        // output
        return py::array_t<float>(
                    py::buffer_info(
                       &T, // pointer
                       sizeof(float), //itemsize
                       py::format_descriptor<float>::format(),
                       2, // ndim
                       std::vector<size_t> { 4, 4 }, // shape
                       std::vector<size_t> { 4 * sizeof(float), sizeof(float)} // strides
                   )
            );
    }, "Register input1 to input2 using at most input1.shape[0]*input3 points.");

    m.def("icp", [](py::array_t<double> points_obs_array,
                    py::array_t<double> points_ren_array,
                    unsigned int max_iterations, float p_correspondence_distance)
    {
        // input
        auto points_obs_buf = points_obs_array.request();
        if (points_obs_buf.ndim != 2)
            throw std::runtime_error("Expect ren array of shape (num_points, XYZ).");
        if (points_obs_buf.shape[1] != 3)
            throw std::runtime_error("Expect ren array of shape (num_points, XYZ).");
        unsigned long num_points_obs = (unsigned long) points_obs_buf.shape[0];
        auto points_obs = (double*) points_obs_buf.ptr;

        auto points_ren_buf = points_ren_array.request();
        if (points_ren_buf.ndim != 2)
            throw std::runtime_error("Expect obs array of shape (num_points, XYZ).");
        if (points_ren_buf.shape[1] != 3)
            throw std::runtime_error("Expect obs array of shape (num_points, XYZ).");

        unsigned long num_points_ren = (unsigned long) points_ren_buf.shape[0];
        auto points_ren = (double*) points_ren_buf.ptr;

        // compute segmentation
        Eigen::Matrix4f T = performICP(points_obs, num_points_obs,
                                       points_ren, num_points_ren,
                                       max_iterations, p_correspondence_distance);
        T.transposeInPlace(); // s.t. it is correctly transferred to numpy

        // output
        return py::array_t<float>(
                    py::buffer_info(
                       &T, // pointer
                       sizeof(float), //itemsize
                       py::format_descriptor<float>::format(),
                       2, // ndim
                       std::vector<size_t> { 4, 4 }, // shape
                       std::vector<size_t> { 4 * sizeof(float), sizeof(float)} // strides
                   )
            );
    }, "Register input1 to input2 using at most input3 iterations.");

}
