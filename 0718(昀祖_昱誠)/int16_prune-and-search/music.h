#ifdef __cplusplus
extern "C" {  
#endif 

int global_music_antenna;
int global_music_QR_iteration;
int global_music_total_round;
int global_music_angle;
int global_music_type;
int global_music_SNR;
int global_RA;

// test cuda function
// void test_CUDA(void);
void music_init(int subframe, uint32_t *rx_data, float *result);


#ifdef __cplusplus  
} // extern "C"  
#endif