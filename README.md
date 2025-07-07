# pytorch_learning  
1.mkdir build  
2.cd build  
3.cmake -DCMAKE_PREFIX_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)") .. && make  
4. ./voxelization_cpu2 ../src/000134.bin  
