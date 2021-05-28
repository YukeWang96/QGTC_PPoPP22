#ifndef CONFIG_H
#define CONFIG_H

#define warpPerBlock 32

#define numThreads 1024                  // for bit-encoding and decoding
#define numThreads_1 warpPerBlock*32     // for MM computation.

#endif