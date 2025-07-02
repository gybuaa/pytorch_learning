#include <ATen/TensorUtils.h>
#include <torch/extension.h>
#include <vector>
#include <iostream>
#include <chrono>
#include <random>
#include <fstream>
#include <string>

/*
输入:
1. 原始输入点云 [N_points_num, feature_num] ，比如[30000,4] x y z intensity
2. 参数voxel_size，一个3D体素的大小，单位米， [x_size,y_size,z_size] 
3. 参数coors_range输入点云有效坐标距离范围，单位米， [xmin,ymin,zmin,xmax,ymax,zmax]，也就是超出这个范围都要过滤
4. 参数grid_size，[x_num,y_num,z_num]，体素的总个数，这里面x_num和y_num就是最终2D特征图的长宽，实际上就是参数3和参数2计算出来
5. 参数最大的点数max_point，这个最大点数不是说总共输入点云最大多少，是指一个体素内可以存多少个点，属于输入特征的一个固定参数，所以当一个体素内放进去的点超过这个个数后面的点就不能放了
6. 参数max_voxels，最大的有效体素个数，总体素个数和grid_size中的x*y直接相关，如果x*y小于max_voxels，这个参数明显就没意义，如果x*y > max_voxels，那么注定一些体素格子是空的

输出:
[N,32,4] N个有效pillar内的32个点的坐标+intensity
[N] N个有效pillar各自的有效点个数
[N,3] N个有效pillar内各自的全局网格ID
*/

// 示例：CUDA 张量的加法
// torch::Tensor add_tensors_cuda(torch::Tensor a, torch::Tensor b) {
//     // 检查输入张量是否在 CUDA 上，如果不在则自动转换
//     if (!a.is_cuda()) a = a.to(torch::kCUDA);
//     if (!b.is_cuda()) b = b.to(torch::kCUDA);

//     // 直接调用 PyTorch 的加法（自动在 GPU 上执行）
//     return a + b;
// }

int hard_voxelize_cpu(const at::Tensor& input_points,
                      at::Tensor& output_pillars,
                      at::Tensor& output_num_points_per_voxel,
                      at::Tensor& output_grid_id_per_voxel,
                      int& valid_pillar_nums,
                      const std::vector<float>& input_voxel_size,
                      const std::vector<float>& input_valid_points_range,
                      const int input_max_point_per_voxel = 32,
                      const int max_voxels = 40000,
                      const int NDim = 3) {

    
    AT_ASSERTM(input_points.device().is_cpu(), "raw input_points must be a CPU tensor");
    AT_ASSERTM(input_valid_points_range.size() == 2*NDim && NDim == input_voxel_size.size(), \
               "input_valid_points_range size %u != 2*NDim %u != input_voxel_size size %u !", \
               input_valid_points_range.size(),NDim, input_voxel_size.size());


    //计算x y z方向总体素个数 height*width*depth
    std::vector<int> grid_size(NDim);
    for(int i = 0; i < NDim; ++i) {
      grid_size[i] = (input_valid_points_range[i+NDim] - input_valid_points_range[i])/input_voxel_size[i];
    }
    printf("grid size: [%d,%d,%d]\n", grid_size[0], grid_size[1], grid_size[2]);
    int current_valid_voxel = 0;

    at::Tensor temp_grid_points_num_record = at::zeros({grid_size[0], grid_size[1], 2}, at::kInt);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input_points.scalar_type(), "hard-voxelization cpu forward", [&]{
        auto input_tensor_accessor = input_points.accessor<scalar_t, 2>();
        auto output_pillars_accessor = output_pillars.accessor<scalar_t,3>();
        auto output_num_points_per_voxel_accessor = output_num_points_per_voxel.accessor<int,1>();
        auto output_grid_id_per_voxel_accessor = output_grid_id_per_voxel.accessor<int,2>();
        auto temp_grid_points_num_record_accessor = temp_grid_points_num_record.accessor<int,3>();
        printf("input raw points num :%d\n", input_tensor_accessor.size(0));
        for(int i = 0; i < input_points.size(0); ++i) {
          
          bool is_valid = true;
          //check whether cur point is in valid points range
          for(int j = 0; j < NDim; ++j) {
            if(input_tensor_accessor[i][j] < input_valid_points_range[j] || input_tensor_accessor[i][j] > input_valid_points_range[j+NDim]) {
              printf("filter point1[%f,%f,%f] index %d, not in range [%f, %f]\n", input_tensor_accessor[i][0], input_tensor_accessor[i][1], input_tensor_accessor[i][2], i, input_valid_points_range[j], input_valid_points_range[j+NDim]);
              is_valid = false;
              break;
            }
          }
          if(!is_valid) continue;
          
          //再校验一次height width和depth
          int height = (input_tensor_accessor[i][0] - input_valid_points_range[0])/input_voxel_size[0];
          int width = (input_tensor_accessor[i][1] - input_valid_points_range[1])/input_voxel_size[1];
          int depth = (input_tensor_accessor[i][2] - input_valid_points_range[2])/input_voxel_size[2];
          if(height < 0 || width < 0 || depth < 0 || height >= grid_size[0] || width >= grid_size[1] || depth >= grid_size[2]) { 
            printf("filter point2[%f,%f,%f] index %d grid id[%d,%d,%d], not in grid\n", 
                  input_tensor_accessor[i][0], input_tensor_accessor[i][1], input_tensor_accessor[i][2], i, height, width, depth);
            continue;
          }       
          
          int cur_grid_pt_nums = temp_grid_points_num_record_accessor[height][width][0];
          int voxel_id = temp_grid_points_num_record_accessor[height][width][1];
          if(cur_grid_pt_nums == 0 && current_valid_voxel < max_voxels) { //这个点所在voxel还未加入点,同时小于约定最大体素个数
            for(int k = 0; k < input_tensor_accessor.size(1); k++) {
              output_pillars_accessor[current_valid_voxel][0][k] = input_tensor_accessor[i][k]; //output pillars赋值
            } 
            output_num_points_per_voxel_accessor[current_valid_voxel]++;   //output每个voxel的有效点数赋值

            output_grid_id_per_voxel_accessor[current_valid_voxel][0] = height;  //output每个voxel的grid id赋值
            output_grid_id_per_voxel_accessor[current_valid_voxel][1] = width;
            output_grid_id_per_voxel_accessor[current_valid_voxel][2] = 1;

            temp_grid_points_num_record_accessor[height][width][1] = current_valid_voxel; //更新每个voxel内记录的有效点数和所在voxel的ID
            temp_grid_points_num_record_accessor[height][width][0]+=1;

            current_valid_voxel++;
            // printf("first grid, point[%f,%f,%f] index %d , grid height width depth [%d,%d,%d]\n", 
            //         input_tensor_accessor[i][0],input_tensor_accessor[i][1],input_tensor_accessor[i][2], i, height, width, depth);
          }else if(cur_grid_pt_nums > 0 && cur_grid_pt_nums < input_max_point_per_voxel) {
            for(int k = 0; k < input_tensor_accessor.size(1); k++) {
              output_pillars_accessor[voxel_id][cur_grid_pt_nums][k] = input_tensor_accessor[i][k]; //output_pillars赋值
            }             
            output_num_points_per_voxel_accessor[voxel_id]++;   //output每个voxel的有效点数赋值
            // printf("repeat grid, point[%f,%f,%f] index %d , grid height width depth [%d,%d,%d]\n", 
            //         input_tensor_accessor[i][0],input_tensor_accessor[i][1],input_tensor_accessor[i][2], i, height, width, depth);
            temp_grid_points_num_record_accessor[height][width][0]+=1; //更新每个voxel内记录的有效点数          
          }else {
            printf("enough point, filter point[%f,%f,%f] index %d grid id[%d,%d,%d],cur_grid_pt_nums %d, current_valid_voxel %d\n", 
                  input_tensor_accessor[i][0], input_tensor_accessor[i][1], input_tensor_accessor[i][2], i, height, width, depth, cur_grid_pt_nums, current_valid_voxel);
            continue; //后续的点不再加入当前pillar
          }

        }
    }
    );

    valid_pillar_nums = current_valid_voxel;
    return 0;
}


// 从.bin文件加载点云到Tensor
torch::Tensor load_points_from_bin(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + file_path);
    }

    // 获取文件大小
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    // 检查文件大小是否合理（必须是4的倍数，因为每个点4个float）
    if (file_size % (4 * sizeof(float)) != 0) {
        throw std::runtime_error("Invalid file size. Expected N x 4 float values.");
    }

    size_t num_points = file_size / (4 * sizeof(float));
    auto tensor = torch::empty({(long)num_points, 4}, torch::kFloat32);
    auto accessor = tensor.accessor<float, 2>();

    // 直接读取到Tensor内存
    file.read(reinterpret_cast<char*>(tensor.data_ptr<float>()), file_size);
    file.close();

    return tensor;
}

// 可选：本地测试
int main(int argc, char** argv) {
    // 参数配置
    const std::vector<float> voxel_size = {0.16f, 0.16f, 4.0f};
    const std::vector<float> valid_range = {0.0f, -40.0f, -3.0f, 70.0f, 40.0f, 1.0f};
    const int max_points_per_voxel = 32;
    const int max_voxels = 40000;
    const int NDim = 3;
    int num_tests = 1;

    torch::Tensor points;
    
    // 如果提供了bin文件路径
    if (argc > 1) {
        try {
            std::string bin_path = argv[1];
            std::cout << "Loading points from: " << bin_path << std::endl;
            points = load_points_from_bin(bin_path);
            std::cout << "Loaded " << points.size(0) << " points" << std::endl;
            
            // 只测试一次（实际数据不需要多次测试）
            num_tests = 1;
        } catch (const std::exception& e) {
            std::cerr << "Error loading bin file: " << e.what() << std::endl;
            return -1;
        }
    } else {
        // 随机生成点云（备用）
        const int min_points = 50000;
        const int max_points = 100000;
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> x_dist(valid_range[0], valid_range[3]);
        std::uniform_real_distribution<float> y_dist(valid_range[1], valid_range[4]);
        std::uniform_real_distribution<float> z_dist(valid_range[2], valid_range[5]);
        std::uniform_real_distribution<float> i_dist(0.0f, 1.0f);

        int num_points = min_points + (gen() % (max_points - min_points + 1));
        points = torch::empty({num_points, 4}, torch::kFloat32);
        auto points_a = points.accessor<float, 2>();
        
        for (int i = 0; i < num_points; ++i) {
            points_a[i][0] = x_dist(gen);
            points_a[i][1] = y_dist(gen);
            points_a[i][2] = z_dist(gen);
            points_a[i][3] = i_dist(gen);
        }
    }

    // 准备输出张量
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto pillars = torch::zeros({max_voxels, max_points_per_voxel, 4}, options);
    auto num_points_per_voxel = torch::zeros({max_voxels}, torch::kInt32);
    auto grid_id_per_voxel = torch::zeros({max_voxels, 3}, torch::kInt32);

    // 性能测试
    double total_time = 0.0;
    int valid_pillar_nums = 0;
    for (int test = 0; test < num_tests; ++test) {
        auto start = std::chrono::high_resolution_clock::now();
        
        hard_voxelize_cpu(points, pillars, num_points_per_voxel, grid_id_per_voxel, valid_pillar_nums,
                                    voxel_size, valid_range, max_points_per_voxel, max_voxels, NDim);
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        double milliseconds = elapsed.count() * 1000;
        total_time += milliseconds;
        
        std::cout << "Test " << test + 1 << ": " << points.size(0) 
                  << " points processed in " << milliseconds << " ms" << std::endl;
        std::cout << "valid pillar nums: " << valid_pillar_nums << std::endl;
    }

    std::cout << "\nAverage processing time: " << (total_time / num_tests) << " ms" << std::endl;
    return 0;
}
