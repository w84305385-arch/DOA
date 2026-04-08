#ifndef MULTI_BEAM_WEIGHTS_H
#define MULTI_BEAM_WEIGHTS_H

extern const int rx_m_index[];
extern const int beam_deg_index[];
int find_index(const int *array, int size, int value);

#ifdef __cplusplus
extern "C" {
#endif

// beam weight table: w_multi_beamCont_re[4][2][576]
extern const float w_multi_beamCont_re[4][2][576];

#ifdef __cplusplus
}
#endif

#endif  // COMBINE_BEAM_WEIGHTS_H