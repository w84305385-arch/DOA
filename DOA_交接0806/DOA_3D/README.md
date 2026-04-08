# DOA演算法開發
C語言DOA 3D演算法獨立模擬版本
使用AVX512指令集加速運算

- 專案啟動 : 
    - NTUST 109吳閔勳學長 : MUSIC與MVDR cpp版本開發、整合OAI
    - NTUST 110徐兆鴻學長 : c版本開發、AVX512矩陣乘法、整合OAI
    - NTUST 111何昀祖學長、陳昱晨學長 : ML DOA開發、int版本開發、快速搜索開發、barker code methed 1 2開發、整合OAI

- 當前階段 : 2024.11.01~2025.06.30
    - 3D Tx model 開發
    - 3D DOA開發
    - 3D DOA Prune_and_Search開發

## System and Environment
本專案在以下環境進行開發與測試：
- CPU : Intel(R) Core(TM) i9-9960X CPU @ 3.10GHz
- Architecture: x86-64
- system : Ubuntu 18.04.6 LTS
- kernal : 5.4.0-150-generic
- gcc :  9.4.0


## How to build
```shell
$cd build`
$cmake ..`
$make`
$make {Executable file name}`
```