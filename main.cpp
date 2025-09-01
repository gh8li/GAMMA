#include <cstring>
#include <iostream>
#include <string>
#include <cuda_runtime.h>

#include "gamma.h"
#include "utils/config.h"
#include "utils/simple_command_parser.h"

namespace {
void PrintMacros(){
    std::cout << "Macros (Items after the third one are for GPU configuration.):" << '\n';
    std::cout << "MAX_QV_COUNT: " << MAX_QV_COUNT << '\n';
    std::cout << "MAX_QE_COUNT: " << MAX_QE_COUNT << '\n';
    std::cout << "MIN_NBR_SIZE: " << MIN_NBR_SIZE << '\n';
    std::cout << "GRID_DIM: " << GRID_DIM << '\n';
    std::cout << "BLOCK_DIM: " << BLOCK_DIM << '\n';
    std::cout << "WARP_SIZE: " << WARP_SIZE << '\n';
    std::cout << "NUM_WARP_PER_BLOCK: " << NUM_WARP_PER_BLOCK << '\n';
    std::cout << "NBR_SPACE: " << NBR_SPACE * sizeof(uint32_t) / 1024 / 1024 << " MiB of uint32_t" << '\n';
    std::cout << "RES_SPACE: " << RES_SPACE * sizeof(uint32_t) / 1024 / 1024 << " MiB of uint32_t" << '\n';
    std::cout << "SIZE_SPACE: " << SIZE_SPACE * sizeof(long) / 1024 / 1024 << " MiB of long" << '\n';
    std::cout << "MIN_NUM_RESULTS_TO_GPU: " << MIN_NUM_RESULTS_TO_GPU << '\n';
}

void PrintGammaInfo(std::string query_path, std::string data_path, std::string update_path,
                    const int32_t device_id, const uint32_t batch_size, const bool print_results) {
    std::cout << "--------------------------------------------------------------------" << std::endl;

    std::cout << "Command Line:" << '\n';
    std::cout << "\tGamma: yes" << '\n';
    std::cout << "\tQuery Graph: " << query_path << '\n';
    std::cout << "\tData Graph: " << data_path << '\n';
    std::cout << "\tData Graph Update: " << update_path << '\n';
    std::cout << "\tDevice ID: " << device_id << '\n';
    std::cout << "\tBatch Size: " << batch_size << '\n';
    std::cout << "\tPrint Results?: " << (print_results?"yes":"no") << '\n';

    std::cout << "--------------------------------------------------------------------" << std::endl;
    PrintMacros();
    std::cout << "--------------------------------------------------------------------" << std::endl;
}
}  // namespace

int main(int argc, char *argv[]) {
    InputParser cmd_parser(argc, argv);

    std::cout << cmd_parser.get_cmd() << std::endl;

    const std::string input_query_path = cmd_parser.get_cmd_option("--query");
    const std::string input_data_path = cmd_parser.get_cmd_option("--data");
    const std::string input_update_path = cmd_parser.get_cmd_option("--update");

    const int32_t input_device_id = cmd_parser.get_int32_cmd_option("--device", /*default_value=*/0);
    const uint32_t input_batch_size = cmd_parser.get_uint32_cmd_option("--batch_size", /*default_value=*/UINT32_MAX);
    const bool print_results = cmd_parser.get_bool_cmd_option("--print_results", /*default_value=*/false);
    const bool print_indexing_time = cmd_parser.get_bool_cmd_option("--print_indexing_time", /*default_value=*/false);

    cudaSetDevice(input_device_id);

    PrintGammaInfo(input_query_path, input_data_path, input_update_path, input_device_id, input_batch_size, print_results);

    gammaProcess(input_query_path, input_data_path, input_update_path, input_batch_size, print_results, print_indexing_time);

    return 0;
}