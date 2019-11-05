#ifndef MESH_SAMPLING_H
#define MESH_SAMPLING_H

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/common/transforms.h>
#include <vtkVersion.h>
#include <vtkPLYReader.h>
#include <vtkOBJReader.h>
#include <vtkTriangle.h>
#include <vtkTriangleFilter.h>
#include <vtkPolyDataMapper.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/console/print.h>
#include <pcl/console/parse.h>

namespace lass {

void uniform_sampling (vtkSmartPointer<vtkPolyData> polydata, std::size_t n_samples, bool calc_normal, bool calc_color, pcl::PointCloud<pcl::PointXYZRGBNormal> & cloud_out);

}
#endif