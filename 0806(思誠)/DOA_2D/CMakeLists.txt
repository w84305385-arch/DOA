# 設定最低 CMake 版本
cmake_minimum_required(VERSION 3.10)

# 定義專案名稱
project(2D_DOA)

# 設定 C++ 標準
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 定義 common 資料夾的路徑
set(COMMON_DIR ${CMAKE_SOURCE_DIR}/common)

# 添加靜態庫 math_func，包含 math_func.cpp
add_library(math_func STATIC ${COMMON_DIR}/math_func.cpp)

# 定義共用文件
set(COMMON_SRC
    ${COMMON_DIR}/complex_matrix_ops.c
    ${COMMON_DIR}/eigen_qr.c
    ${COMMON_DIR}/generate_signal.c
    ${COMMON_DIR}/lu_decomp.c
    ${COMMON_DIR}/spatial_spectrum.c
    ${COMMON_DIR}/multi_beam_weight.c

    ${COMMON_DIR}/complex_matrix_ops_int32.c
    ${COMMON_DIR}/spatial_spectrum_int32.c
)

# 定義執行檔目標及其主文件
set(TARGETS

    doa2d_ml_float           float_DOA_2D/doa2d_ml_float.c
    doa2d_music_bmgsqr_float float_DOA_2D/doa2d_music_bmgsqr_float.c
    doa2d_mvdr_bmgsqr_float  float_DOA_2D/doa2d_mvdr_bmgsqr_float.c
    doa2d_mvdr_lu_float      float_DOA_2D/doa2d_mvdr_lu_float.c

    doa2d_ml_prunesearch_float           float_DOA_2D_prune_and_search/doa2d_ml_prunesearch_float.c
    doa2d_music_bmgsqr_prunesearch_float float_DOA_2D_prune_and_search/doa2d_music_bmgsqr_prunesearch_float.c
    doa2d_mvdr_bmgsqr_prunesearch_float  float_DOA_2D_prune_and_search/doa2d_mvdr_bmgsqr_prunesearch_float.c
    doa2d_mvdr_lu_prunesearch_float      float_DOA_2D_prune_and_search/doa2d_mvdr_lu_prunesearch_float.c


    doa2d_music_bmgsqr_float_int32 float_int32_DOA_2D/doa2d_music_bmgsqr_float_int32.c
    doa2d_mvdr_lu_float_int32      float_int32_DOA_2D/doa2d_mvdr_lu_float_int32.c
    )


# 獲取 TARGETS 列表長度
list(LENGTH TARGETS TARGETS_LENGTH)

# 使用循環生成多個執行檔目標
math(EXPR MAX_INDEX "${TARGETS_LENGTH} - 1")

foreach(i RANGE 0 ${MAX_INDEX} 2)
    # 計算主文件索引
    math(EXPR MAIN_FILE_INDEX "${i} + 1")

    # 提取目標名稱和主文件
    list(GET TARGETS ${i} TARGET_NAME)
    list(GET TARGETS ${MAIN_FILE_INDEX} MAIN_FILE)
    message("Adding target: ${TARGET_NAME} with file: ${MAIN_FILE}")
    
    # 添加執行檔目標
    add_executable(${TARGET_NAME} ${MAIN_FILE} ${COMMON_SRC})
    
    # 添加頭文件目錄
    target_include_directories(${TARGET_NAME} PRIVATE ${COMMON_DIR})
    
    # 將靜態庫鏈接到目標
    target_link_libraries(${TARGET_NAME} PRIVATE math_func)
    
    # 添加編譯器選項
    target_compile_options(${TARGET_NAME} PRIVATE
        -mavx512f
        -mavx512dq
        -mavx512vl
        -mavx512bw
        -g
        -Wall
        -Wextra

    )
endforeach()
