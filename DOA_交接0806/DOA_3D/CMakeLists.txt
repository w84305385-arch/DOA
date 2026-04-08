# 設定最低 CMake 版本
cmake_minimum_required(VERSION 3.10)

# 定義專案名稱
project(3D_DOA)

# 設定 C++ 標準（雖然目前多為 C 檔，但保留 C++ 支援）
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 定義 common 資料夾的路徑
set(COMMON_DIR ${CMAKE_SOURCE_DIR}/common)

# 設定每個 target 自定的 Q_SHIFT（根據你的命名）
set(Q_SHIFT_MAP
    doa3d_music_bmgsqr_float_int32=13
    doa3d_mvdr_bmgsqr_float_int32=10
    doa3d_mvdr_lu_float_int32=10

    doa3d_music_bmgsqr_prunesearch_float_int32=13
    doa3d_mvdr_bmgsqr_prunesearch_float_int32=10
    doa3d_mvdr_lu_prunesearch_float_int32=10
)
set(Q_SHIFT_VALUE 0)

# 添加靜態庫 math_func_3D（來自 common 資料夾）
add_library(math_func_3D STATIC
    ${COMMON_DIR}/math_func_3D.cpp
)


# 定義共用 source 檔案
set(COMMON_SRC
    ${COMMON_DIR}/complex_matrix_ops.c
    ${COMMON_DIR}/eigen_qr.c
    ${COMMON_DIR}/generate_signal_3D.c
    ${COMMON_DIR}/lu_decomp.c
    ${COMMON_DIR}/spatial_spectrum.c
    ${COMMON_DIR}/multi_beam_weight.c

    ${COMMON_DIR}/complex_matrix_ops_int32.c
    ${COMMON_DIR}/spatial_spectrum_int32.c
)

# 定義執行檔目標及其主文件
set(TARGETS
    # original
    doa3d_ml_float float_DOA_3D/doa3d_ml_float.c
    doa3d_music_bmgsqr_float float_DOA_3D/doa3d_music_bmgsqr_float.c
    doa3d_mvdr_bmgsqr_float float_DOA_3D/doa3d_mvdr_bmgsqr_float.c
    doa3d_mvdr_lu_float float_DOA_3D/doa3d_mvdr_lu_float.c


    # prune and search
    doa3d_ml_prunesearch_float float_DOA_3D_prune_and_search/doa3d_ml_prunesearch_float.c
    doa3d_music_bmgsqr_prunesearch_float float_DOA_3D_prune_and_search/doa3d_music_bmgsqr_prunesearch_float.c
    doa3d_mvdr_bmgsqr_prunesearch_float float_DOA_3D_prune_and_search/doa3d_mvdr_bmgsqr_prunesearch_float.c
    doa3d_mvdr_lu_prunesearch_float float_DOA_3D_prune_and_search/doa3d_mvdr_lu_prunesearch_float.c


    # original int32
    doa3d_music_bmgsqr_float_int32 float_int32_DOA_3D/doa3d_music_bmgsqr_float_int32.c
    doa3d_mvdr_bmgsqr_float_int32 float_int32_DOA_3D/doa3d_mvdr_bmgsqr_float_int32.c
    doa3d_mvdr_lu_float_int32 float_int32_DOA_3D/doa3d_mvdr_lu_float_int32.c


    # prune and search int32
    doa3d_music_bmgsqr_prunesearch_float_int32 float_int32_DOA_3D_prune_and_search/doa3d_music_bmgsqr_prunesearch_float_int32.c
    doa3d_mvdr_bmgsqr_prunesearch_float_int32 float_int32_DOA_3D_prune_and_search/doa3d_mvdr_bmgsqr_prunesearch_float_int32.c
    doa3d_mvdr_lu_prunesearch_float_int32 float_int32_DOA_3D_prune_and_search/doa3d_mvdr_lu_prunesearch_float_int32.c
)

# 計算 TARGETS 的長度
list(LENGTH TARGETS TARGETS_LENGTH)
math(EXPR MAX_INDEX "${TARGETS_LENGTH} - 1")

# 使用 foreach 為每個 target 自動產生執行檔
foreach(i RANGE 0 ${MAX_INDEX} 2)
    # 擷取 target name 和對應的 source file
    math(EXPR MAIN_FILE_INDEX "${i} + 1")
    list(GET TARGETS ${i} TARGET_NAME)
    list(GET TARGETS ${MAIN_FILE_INDEX} MAIN_FILE)

    message("Adding target: ${TARGET_NAME} with file: ${MAIN_FILE}")

    # 建立完整 source list（主程式 + 共用 C 檔）
    set(SRC_FILES
        ${MAIN_FILE}
        ${COMMON_SRC}
    )
    
    # 建立可執行檔
    add_executable(${TARGET_NAME} ${SRC_FILES})
    # 預設值
    set(Q_SHIFT_VALUE 9)

    foreach(pair IN LISTS Q_SHIFT_MAP)
        string(REPLACE "=" ";" pair_list ${pair})
        list(GET pair_list 0 name)
        list(GET pair_list 1 value)

        if("${TARGET_NAME}" STREQUAL "${name}")
            set(Q_SHIFT_VALUE ${value})
        endif()
    endforeach()

    message("${TARGET_NAME} Q_SHIFT = ${Q_SHIFT_VALUE}")
    target_compile_definitions(${TARGET_NAME}
        PRIVATE
        Q_SHIFT=${Q_SHIFT_VALUE}
    )
    
    # 包含頭檔路徑（讓 thread_pool.h 等可 include）
    target_include_directories(${TARGET_NAME}
        PRIVATE
        ${COMMON_DIR}
    )
    target_link_libraries(${TARGET_NAME}
        PRIVATE
        math_func_3D
    )

    # 編譯器選項（可依平台調整）
    target_compile_options(${TARGET_NAME}
        PRIVATE
        -g
        -mavx512f
        -mavx512dq
        -mavx512vl
        -mavx512bw
        -Wall
        -Wextra
        #-fsanitize=undefined
        #-fsanitize=address
        #-fsanitize=leak
        #-fsanitize=thread
    )
        target_link_options(${TARGET_NAME}
        PRIVATE
        #-fsanitize=address
        #-fsanitize=undefined 
        #-fsanitize=leak
        #-fno-omit-frame-pointer
    )
endforeach()
