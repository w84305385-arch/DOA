#ifndef DOA_PARAMETERS_H
#define DOA_PARAMETERS_H

//----------------------------------------------------------------
// === Physical Parameters ===
typedef struct {
    float fc;         // Carrier frequency [Hz]
    float c;          // Speed of light [m/s]
    float lambda;   // Wavelength [m]
    float kc;       // Wavenumber [rad/m]
    int snr;        // Signal-to-Noise Ratio [dB]
} PhysicalParameters;

// === Tx Parameters ===
typedef struct {
    int Tx_M_x;
    float Tx_beamwidth;
    int Tx_beamwidth_samples;
    float *angle_theta;    // Array of incident angles [degrees]
    float *angle_phi;    // Array of incident angles [degrees]
    int number_angle;    // Number of Tx directions
} TxParameters;

// === Rx Parameters ===
typedef struct {
    int Rx_M_x;
    float d;        // Element spacing [m]
    int nd;         // Number of data samples per channel
    int qr_iter;
    int BMGS_qr_num_blocks;
} RxParameters;
//----------------------------------------------------------------

#endif // DOA_PARAMETERS_H
