/////////////////////////////////////////////////////
// Convert floating point input into 1-bit input layer
// the height and width will not change for input layer
/////////////////////////////////////////////////////
#ifndef PARAM
#define PARAM

class In128LayerParam
{
    public:
        In128LayerParam(const char* _name, int _input_height, int _input_width, int _bit_width)
            :input_height(_input_height), output_height(_input_height),
            input_width(_input_width), output_width(_input_width),
            input(NULL), input_gpu(NULL), output(NULL), output_gpu(NULL), bitwidth(_bit_width)
        {
            strncpy(name, _name, 8);
        }
        //input utility
        int input_size() { return input_height*input_width; }
        int input_bytes() { return input_size()*sizeof(float);}
        int input_bit_size() { return input_height*input_width; }
        int input_bit_bytes() { return input_bit_size()*sizeof(float);}
        //output utility
        int output_size() { return  output_height*output_width;}
        int output_bytes() { return output_size()*sizeof(float);}
        //binarize on row
        int output_bit_size() { return bitwidth*PAD8(output_height)*STEP128(output_width);}
        // add the bitwidth for computation (BW x compressed_feature_map)
        int output_bit_bytes() { return output_bit_size()*sizeof(uin128);}

        In128LayerParam* initialize(float* input)
        {
            cudaGetDeviceProperties(&deviceProp, dev);
            
            CHECK_NULL_POINTER(input);
            this->input = input;
            SAFE_ALOC_GPU(input_gpu, input_bytes());
            CUDA_SAFE_CALL( cudaMemcpy(input_gpu, input, input_bytes(), cudaMemcpyHostToDevice) );
            // print_image_10x10_float(input);

            // input_gpu (float 32-bit) -->  input_qnt_gpu (uint 32-bit)
            dataQuantization();
            SAFE_ALOC_HOST(input_qnt, input_bytes());
            CUDA_SAFE_CALL( cudaMemcpy(input_qnt, input_qnt_gpu, input_bytes(), cudaMemcpyDeviceToHost) );
            // print_image_10x10_int(input_qnt);

            // input_qnt_gpu (uint 32-bit) -->  output_gpu (packed 1-bit as uint 32-bit)
            bitDecomposition();
            SAFE_ALOC_HOST(output_uin32, bitwidth*output_height*output_width*sizeof(uin32));
            CUDA_SAFE_CALL( cudaMemcpy(output_uin32, output_uin32_gpu, bitwidth*output_height*output_width*sizeof(uin32), cudaMemcpyDeviceToHost));
            
            // print_image_10x10_int_bit_decompose(output_uin32);
            SAFE_ALOC_GPU(output_gpu, output_bit_bytes());
            CUDA_SAFE_CALL( cudaMemset(this->output_gpu, 0, output_bit_bytes()) );
            return this->ready();
        }

        In128LayerParam* ready()
        {
            CHECK_NULL_POINTER(input_gpu);
            CHECK_NULL_POINTER(output_gpu);
            SAFE_ALOC_GPU(gpu, sizeof(In128LayerParam));
            CUDA_SAFE_CALL( cudaMemcpy(this->gpu, this, sizeof(In128LayerParam), cudaMemcpyHostToDevice) );
            return this->gpu;
        }

        void set_output_gpu(uin32* _output_gpu) 
        { 
            this->output_gpu = _output_gpu; 
        }

        uin32* get_input_qnt(){
            return this->input_qnt;
        }
        
        uin32* get_output_gpu()
        { 
            return this->output_gpu; 
        }
        uin32* download_output()
        {
            if (output == NULL) 
                SAFE_ALOC_HOST(output, output_bit_bytes());
            CUDA_SAFE_CALL( cudaMemcpy(output, output_gpu, output_bit_bytes(), cudaMemcpyDeviceToHost) );
            return this->output;
        }
        
        /*Added quantization for quantize the initial value to N-bit representation in INT32*/
        void dataQuantization()
        {
            SAFE_ALOC_GPU(input_qnt_gpu, input_bytes());

            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, Quantize_val,  numThreads, 0);            
            // input_gpu (float 32-bit) -->  input_qnt_gpu (uint 32-bit)
            Quantize_val<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(input_qnt_gpu, input_gpu, output_height*output_width, bitwidth);

            CUDA_SAFE_CALL( cudaPeekAtLastError());
            CUDA_SAFE_CALL( cudaDeviceSynchronize());
        }

        /*Added quantization for decompose the INT32 matrix to N-bit x M x N bit matrix*/
        void bitDecomposition()
        {
            SAFE_ALOC_GPU(output_uin32_gpu, bitwidth*output_height*output_width*sizeof(uin32));
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, Decompose_bit, numThreads, 0);
            Decompose_bit<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(output_uin32_gpu, input_qnt_gpu, output_height*output_width, bitwidth);
        }


        uin32* download_full_output()
        {
            const int size = output_bytes();

            uin32* full_output = NULL;
            SAFE_ALOC_HOST(full_output, size);

            uin32* full_output_gpu = NULL;
            SAFE_ALOC_GPU(full_output_gpu, size);
            CUDA_SAFE_CALL( cudaMemset(full_output_gpu, 0, size) );

            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, UnPackFcOutput128, 
                    numThreads, 0);
            UnPackFcOutput128<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(
                    output_gpu, full_output_gpu, output_height, output_width, bitwidth);

            CUDA_SAFE_CALL( cudaMemcpy(full_output, full_output_gpu, size, cudaMemcpyDeviceToHost) );
            CUDA_SAFE_CALL( cudaFree(full_output_gpu) );
            return full_output;
        }

        // print the image with float-point value.
        void print_image_10x10_float(float* image){
            printf("\n------print_image_10x10_float-----------\n");
            const int show_height = 28;
            const int show_width = show_height;
            for (int i = 0; i < show_height; i++){
                for (int j = 0; j < show_width; j++){
                    printf("%f ", image[i * 28 + j]);
                }
                printf("\n");
            }
            printf("=========================================\n");
        }

        // print the image with their int value.
        void print_image_10x10_int(uin32* image){
            printf("\n------print_image_10x10_int-----------\n");
            const int show_height = 10;
            const int show_width = show_height;
            for (int i = 0; i < show_height; i++){
                for (int j = 0; j < show_width; j++){
                    printf("%d ", image[i * 28 + j]); // * show the first image.
                }
                printf("\n");
            }
            printf("=========================================\n");
        }

        // print the image with their decomposed bit in 32-bit.
        void print_image_10x10_int_bit_decompose(uin32* image){
            const int show_height = 28;
            const int show_width = show_height;
            
            printf("\n------print_image_28x28_int_bit_decompose-----------\n");
            for (int i = 0; i < show_height; i++){
                for (int j = 0; j < show_width; j++){
                    for (int b = bitwidth - 1; b >= 0; b--)
                        printf("%d", image[b * output_width * output_height + i * 28 + j]); // * show the first image.
                    printf(" ");
                }
                printf("\n");
            }
            printf("=========================================\n");
        }

        void release() 
        {
            SAFE_FREE_GPU(input_gpu);
            SAFE_FREE_GPU(output_gpu);
            SAFE_FREE_GPU(gpu);
            SAFE_FREE_HOST(output);
        }

        ~In128LayerParam() { release(); }
    
    public:
        float* input;
        float* input_gpu;           // float input (M x N)

        uin32* input_qnt;
        uin32* input_qnt_gpu;       // quantized uint32 middle representation. (M x N)

        uin32*  output;
        uin32*  output_uin32;
        uin32*  output_uin32_gpu;     // uint32 matrix representation. (bitwidth x M x N)
        uin32*  output_gpu;          //  packed uint32 1-bit matrix representation. (bitwidth x M x N/32)
        
 
        int input_width;
        int input_height;
        int output_width;
        int output_height;
        int bitwidth;

        int numThreads = 1024;
        int numBlocksPerSm;
        cudaDeviceProp deviceProp;

        In128LayerParam* gpu;
        char name[8];
};

class Fc128LayerParam
{
    public:
        Fc128LayerParam(const char* name, int _input_height, int _input_width, 
                int _weight_width, int act_bit, int w_bit) : 
            weight_height(_input_width), weight_width(_weight_width), 
            input_height(_input_height), input_width(_input_width),
            output_height(_input_height), output_width(_weight_width),
            bn_width(_weight_width), weight(NULL), weight_gpu(NULL),
            bn(NULL), bn_gpu(NULL), output(NULL), output_gpu(NULL),
            input(NULL), input_gpu(NULL), gpu(NULL), act_bit(act_bit), w_bit(w_bit)
        {
            strncpy(this->name, name, 8);
        }
        //row major -- input
        int input_size() { return input_height*input_width;}
        int input_bytes() { return input_size()*sizeof(uin32);}
        int input_bit_size() { return PAD8(input_height)*STEP128(input_width);}
        int input_bit_bytes() { return input_bit_size()*sizeof(uin128);}

        //colum major -- weight
        int weight_size() { return weight_height*weight_width;}
        int weight_bytes() { return weight_size()*sizeof(float);}
        int weight_bit_size() { return w_bit*STEP128(weight_height)*PAD128(weight_width);}
        int weight_bit_bytes() { return weight_bit_size()*sizeof(uin128);}
        
        //row-major -- output
        int output_size() { return output_height*output_width;}
        int output_bytes() { return output_size()*sizeof(float);}
        int output_bit_size() { return act_bit*PAD8(output_height)*STEP128(output_width);}
        int output_bit_bytes() { return output_bit_size() * sizeof(uin128);}

        //batch-norm
        int bn_size() { return bn_width;}
        int bn_bytes() { return bn_size()*sizeof(float);}

        Fc128LayerParam* ready()
        {
            CHECK_NULL_POINTER(input_gpu);
            CHECK_NULL_POINTER(output_gpu);
            SAFE_ALOC_GPU(gpu, sizeof(Fc128LayerParam));
            // points to a GPU object
            CUDA_SAFE_CALL( cudaMemcpy(this->gpu, this, sizeof(Fc128LayerParam), cudaMemcpyHostToDevice) );
            return this->gpu;
        }

        void set_input_gpu(uin32* _input_gpu)
        {
            this->input_gpu = _input_gpu;
        }

        Fc128LayerParam* initialize(FILE* config_file, uin32* prev_layer_gpu)
        {
            //Initialize weight[CPU] -- float32
            SAFE_ALOC_HOST(weight, weight_bytes());
            launch_array(config_file, this->weight, weight_size());

            // Arbitarized weight_gpu [GPU]
            SAFE_ALOC_GPU(weight_gpu, weight_bit_bytes());
            CUDA_SAFE_CALL( cudaMemset(this->weight_gpu, 0, weight_bit_bytes()) );

            // Initialize weight_float [GPU] -- float32
            SAFE_ALOC_GPU(weight_float, weight_bytes());
            CUDA_SAFE_CALL( cudaMemcpy(weight_float, weight, weight_bytes(), cudaMemcpyHostToDevice) );

            cudaGetDeviceProperties(&deviceProp, dev);

            // weight_float (float 32-bit) -->  quantized weight_qnt_gpu (uint 32-bit)
            weightQuantization();

            // check weight_uin32[CPU]  <--- weight_qnt_gpu[GPU]
            SAFE_ALOC_HOST(weight_uin32, weight_size()*sizeof(uin32));
            CUDA_SAFE_CALL( cudaMemcpy(weight_uin32, weight_qnt_gpu, weight_size()*sizeof(uin32), cudaMemcpyDeviceToHost));
            
            // printf("weight_height: %d, weight_width: %d\n", weight_height, weight_width);
            // printf("\n\nFC weight (float)");
            // print_image_10x10_float(weight);

            // printf("\n\nFC weight (quantied uint 32)");
            // print_image_10x10_int(weight_uin32);
            // exit(0);

            // weight_qnt_gpu (uint32) -->  weight_uin32_dec_gpu (packed 1-bit as uint 32-bit)
            weightBitDecomposition();

            SAFE_ALOC_HOST(weight_uin32_dec, w_bit*weight_size()*sizeof(uin32));
            CUDA_SAFE_CALL( cudaMemcpy(weight_uin32_dec, weight_uin32_dec_gpu, w_bit*weight_size()*sizeof(uin32), cudaMemcpyDeviceToHost));
            
            // printf("\n\nFC weight (bit decomposed uint 32)");
            // print_image_10x10_int_bit_decompose(weight_uin32_dec);

#ifdef NEWFMT
            // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, PackFcWeight128FMT, 
            //         numThreads, 0);
            // PackFcWeight128FMT<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(
            //         weight_float, weight_gpu, weight_height, weight_width);
#else
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, PackFcWeight128, 
                    numThreads, 0);
            PackFcWeight128<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(
                    weight_uin32_dec_gpu, weight_gpu, weight_height, weight_width, w_bit);
#endif
            // uin32* weight_cpu=NULL;
            // SAFE_ALOC_HOST(weight_cpu, weight_bit_bytes());
            // CUDA_SAFE_CALL( cudaMemcpy(weight_cpu, weight_gpu, weight_bit_bytes(), cudaMemcpyDeviceToHost) );
            // for (int i=0;i<weight_bit_size()*4; i++)
            //     printf("%u ", weight_cpu[i]);
            // exit(0);
            CUDA_CHECK_KERNEL();
            SAFE_FREE_GPU(weight_float);
            
            //Process bn
            SAFE_ALOC_HOST(bn, bn_bytes());
            launch_array(config_file, this->bn, bn_size());
            SAFE_ALOC_GPU(bn_gpu, bn_bytes());
            CUDA_SAFE_CALL( cudaMemcpy(bn_gpu, bn, bn_bytes(), cudaMemcpyHostToDevice) );

            //Allocate output gpu
            SAFE_ALOC_GPU(output_gpu, output_bit_bytes());
            CUDA_SAFE_CALL( cudaMemset(this->output_gpu, 0, output_bit_bytes()) );

            set_input_gpu(prev_layer_gpu);
            return this->ready();
        }

        /* quantization for quantize the initial value to N-bit representation in INT32*/
        void weightQuantization()
        {
            SAFE_ALOC_GPU(weight_qnt_gpu, weight_bytes());

            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, Quantize_val,  numThreads, 0);
            // input_gpu (float 32-bit) -->  input_qnt_gpu (uint 32-bit)
            Quantize_val<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(weight_qnt_gpu, weight_float, weight_size(), w_bit);  
            // printf("after quantize_val\n");
            CUDA_SAFE_CALL( cudaPeekAtLastError());
            CUDA_SAFE_CALL( cudaDeviceSynchronize());
        }

        /*Added quantization for decompose the INT32 matrix to N-bit x M x N bit matrix*/
        void weightBitDecomposition()
        {
            SAFE_ALOC_GPU(weight_uin32_dec_gpu, w_bit * weight_size() * sizeof(uin32));

            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, Decompose_bit, numThreads, 0);

            Decompose_bit<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(weight_uin32_dec_gpu,  weight_qnt_gpu, weight_size(), w_bit);

            CUDA_SAFE_CALL( cudaPeekAtLastError());
            CUDA_SAFE_CALL( cudaDeviceSynchronize());
        }

        // print the image with float-point value.
        void print_image_10x10_float(float* image){
            printf("\n------print_image_10x10_float-----------\n");
            const int show_height = 10;
            const int show_width = show_height;
            for (int i = 0; i < show_height; i++){
                for (int j = 0; j < show_width; j++){
                    printf("%f ", image[i * weight_height + j]);
                }
                printf("\n");
            }
            printf("=========================================\n");
        }

        // print the image with their decomposed bit in 32-bit.
        void print_weight_int_bit_decompose(uin32* image){
            const int show_height = weight_height;
            const int show_width = 1;
            // column store
            printf("\n------print_weight_int_bit_decompose-----------\n");
            for (int i = 0; i < show_width; i++){
                for (int j = 0; j < show_height; j++){
                    for (int b = w_bit - 1; b >= 0; b--)
                        // printf("%d", image[b * weight_size() + i * weight_height + j]);
                        printf("%u", image[b * weight_size() + i * show_height + j]);
                    printf(" ");
                }
                printf("\n");
            }
            printf("=========================================\n");
        }

        // print the image with their int value.
        void print_image_uint32(uin32* image, const int img_width){
            // printf("\n------print_image_%dx%d_int-----------\n", width, width);
            const int print_range = 8;
            for (int i = 0; i < print_range; i++){
                for (int j = 0; j < print_range; j++){
                    printf("%d ", image[i*img_width + j]);
                }
                printf("\n");
            }
            printf("=========================================\n");
        }

        uin32* get_output_gpu()
        {
            return this->output_gpu;
        }

        uin32* get_weight_uin32(){
            return this->weight_uin32;
        }

        uin32* download_output()
        {
            if (output == NULL) SAFE_ALOC_HOST(output, output_bit_bytes());
            CUDA_SAFE_CALL( cudaMemcpy(output, output_gpu, 
                        output_bit_bytes(), cudaMemcpyDeviceToHost) );
            return this->output;
        }

        uin32* download_full_weight()
        {
            const int size = weight_bytes();
            uin32* full_weight = NULL;
            SAFE_ALOC_HOST(full_weight, size);

            uin32* full_weight_gpu = NULL;
            SAFE_ALOC_GPU(full_weight_gpu, size);
            CUDA_SAFE_CALL( cudaMemset(full_weight_gpu, 0, size) ); 

#ifdef NEWFMT
            // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, UnPackFcOutput128FMT, 
            //         numThreads, 0);
            // UnPackFcOutput128FMT<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(
            //         output_gpu, full_output_gpu, output_height, output_width, 3);
#else
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, UnPackFcOutput128, 
                    numThreads, 0);
            // * unpack weight
            UnPackFcWeight128<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(
                    weight_gpu, full_weight_gpu, weight_height, weight_width, w_bit);
#endif
            CUDA_SAFE_CALL( cudaPeekAtLastError());
            CUDA_SAFE_CALL( cudaDeviceSynchronize());

            CUDA_SAFE_CALL( cudaMemcpy(full_weight, full_weight_gpu, size, cudaMemcpyDeviceToHost) );
            CUDA_SAFE_CALL( cudaFree(full_weight_gpu) );

            return full_weight;
        }

        uin32* download_full_output()
        {
            const int size = output_size()*sizeof(uin32);
            
            // output (CPU) uint32
            uin32* full_output = NULL;
            SAFE_ALOC_HOST(full_output, size);

            // output (GPU) uint32
            uin32* full_output_gpu = NULL;
            SAFE_ALOC_GPU(full_output_gpu, size);
            CUDA_SAFE_CALL( cudaMemset(full_output_gpu, 0, size) );
            
#ifdef NEWFMT
            // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, UnPackFcOutput128FMT, 
            //         numThreads, 0);
            // UnPackFcOutput128FMT<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(
            //         output_gpu, full_output_gpu, output_height, output_width, 3);
#else
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, UnPackFcOutput128, 
                    numThreads, 0);
            UnPackFcOutput128<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(
                    output_gpu, full_output_gpu, output_height, output_width, act_bit);
#endif
            CUDA_SAFE_CALL( cudaMemcpy(full_output, full_output_gpu, size, cudaMemcpyDeviceToHost) );
            CUDA_SAFE_CALL( cudaFree(full_output_gpu) );

            return full_output;
        }

        void release()
        {
            SAFE_FREE_HOST(weight);
            SAFE_FREE_HOST(bn);
            SAFE_FREE_HOST(output);
            SAFE_FREE_GPU(weight_gpu);
            SAFE_FREE_GPU(bn_gpu);
            SAFE_FREE_GPU(output_gpu);
            SAFE_FREE_GPU(gpu);
        }
        ~Fc128LayerParam() { release(); }

    public:
        //Input
        uin32* input;
        uin32* input_gpu;
        int input_width;
        int input_height;
        
        //Weight
        float* weight;
        float* weight_float = NULL;
        uin32* weight_gpu;
        int weight_width;
        int weight_height;
        
        //Output
        uin32* output;
        uin32* output_gpu;
        int output_width;
        int output_height;

        // support for arbitary precision.
        int w_bit, act_bit;

        uin32* weight_uin32;
        uin32* weight_qnt_gpu;
        uin32* weight_uin32_dec;
        uin32* weight_uin32_dec_gpu;

        int numThreads = 1024;
        int numBlocksPerSm;
        cudaDeviceProp deviceProp;

        //Batch normalization
        float* bn;
        float* bn_gpu;
        int bn_width;
        //GPU shadow
        Fc128LayerParam* gpu;
        char name[8];
};


class Out128LayerParam
{
    public:
        Out128LayerParam(const char* name, int _input_height, 
                int _input_width, int _weight_width, int act_bit, int w_bit) :
            input_height(_input_height), input_width(_input_width),
            weight_height(_input_width), weight_width(_weight_width),
            output_height(_input_height), output_width(_weight_width),
            input(NULL), input_gpu(NULL), output(NULL), output_gpu(NULL),
            weight(NULL), weight_gpu(NULL), act_bit(act_bit), w_bit(w_bit)
        {
            strncpy(this->name, name, 8);
        }
        // row major
        int input_size() { return input_height*input_width;}
        int input_bytes() { return input_size()*sizeof(uin32);}
        int input_bit_size() { return act_bit*PAD8(input_height)*STEP128(input_width);}
        int input_bit_bytes() { return input_bit_size()*sizeof(uin128);}
        
        // colum major
        int weight_size() { return weight_height*weight_width;}
        int weight_bytes() { return weight_size()*sizeof(float);}
        int weight_bit_size() { return w_bit*STEP128(weight_height)*PAD8(weight_width);}
        int weight_bit_bytes() { return weight_bit_size()*sizeof(uin128);}

        // row major
        int output_size() { return output_height*output_width;}
        int output_bytes() { return output_size()*sizeof(float);}
        int output_bit_size() { return act_bit*output_height*output_width;}
        int output_bit_bytes() { return output_bit_size()*sizeof(float);}

        int bn_size() { return output_width;}
        int bn_bytes() { return output_width*sizeof(float); }
 
        Out128LayerParam* ready()
        {
            CHECK_NULL_POINTER(input_gpu);
            CHECK_NULL_POINTER(output_gpu);
            SAFE_ALOC_GPU(gpu, sizeof(Out128LayerParam));
            CUDA_SAFE_CALL( cudaMemcpy(this->gpu, this, sizeof(Out128LayerParam), cudaMemcpyHostToDevice) );
            return this->gpu;
        }

        void set_input_gpu(uin32* input_gpu)
        {
            this->input_gpu = input_gpu;
        }

        uin32* get_output_gpu()
        {
            return this->output_gpu;
        }

        uin32* get_weight_uin32()
        {
            return this->weight_uin32;
        }

        Out128LayerParam* initialize(FILE* config_file, uin32* prev_layer_gpu)
        {

            // CPU weight -- float32
            SAFE_ALOC_HOST(weight, weight_bytes());
            launch_array(config_file, this->weight, weight_size());
            // GPU weight -- arbitarized bit
            SAFE_ALOC_GPU(weight_gpu, weight_bit_bytes());
            CUDA_SAFE_CALL( cudaMemset(this->weight_gpu, 0, weight_bit_bytes()) );
            // GPU weight -- float32
            SAFE_ALOC_GPU(weight_float, weight_bytes());
            CUDA_SAFE_CALL( cudaMemcpy(weight_float, weight, weight_bytes(), cudaMemcpyHostToDevice) );

            cudaGetDeviceProperties(&deviceProp, dev);

            // weight_float (float 32-bit) -->  quantized weight_qnt_gpu (uint 32-bit)
            weightQuantization();

            SAFE_ALOC_HOST(weight_uin32, weight_size()*sizeof(uin32));
            CUDA_SAFE_CALL( cudaMemcpy(weight_uin32, weight_qnt_gpu, weight_size()*sizeof(uin32), cudaMemcpyDeviceToHost));
            // printf("weight_height: %d, weight_width: %d\n", weight_height, weight_width);
            // print_image_10x10_int(weight_uin32);
            
            // weight_qnt_gpu (uint 32-bit) -->  weight_uin32_dec_gpu (packed 1-bit as uint 32-bit)
            weightBitDecomposition();

            SAFE_ALOC_HOST(weight_uin32_dec, w_bit*weight_size()*sizeof(uin32));
            CUDA_SAFE_CALL( cudaMemcpy(weight_uin32_dec, weight_uin32_dec_gpu, w_bit*weight_size()*sizeof(uin32), cudaMemcpyDeviceToHost));
            // printf("\n\nFC_out weight (bit decomposed uint 32)");
            // print_weight_int_bit_decompose(weight_uin32_dec);
#ifdef NEWFMT
            // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, PackFcWeight128FMT, 
            //         numThreads, 0);
            // PackFcWeight128FMT<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(
            //         weight_float, weight_gpu, weight_height, weight_width);
#else
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, PackFcWeight128, 
                    numThreads, 0);
            PackFcWeight128_OUTPUT<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(
                    weight_uin32_dec_gpu, weight_gpu, weight_height, weight_width, w_bit);
#endif
            CUDA_CHECK_KERNEL();

            // printf("weight_bit size: %d ", weight_bit_size());
            // uin32* weight_cpu=NULL;
            // SAFE_ALOC_HOST(weight_cpu, weight_bit_bytes());
            // CUDA_SAFE_CALL( cudaMemcpy(weight_cpu, weight_gpu, weight_bit_bytes(), cudaMemcpyDeviceToHost) );
            // for (int i=0;i < weight_bit_size()*4; i++)
            //     printf("%u\n", weight_cpu[i]);
            // exit(0);

            //BN
            SAFE_ALOC_HOST(bn_scale, bn_bytes());
            launch_array(config_file, this->bn_scale, bn_size());
            SAFE_ALOC_GPU(bn_scale_gpu, bn_bytes());
            CUDA_SAFE_CALL( cudaMemcpy(bn_scale_gpu, bn_scale, bn_bytes(), cudaMemcpyHostToDevice) );

            SAFE_ALOC_HOST(bn_bias, bn_bytes());
            launch_array(config_file, this->bn_bias, bn_size());
            SAFE_ALOC_GPU(bn_bias_gpu, bn_bytes());
            CUDA_SAFE_CALL( cudaMemcpy(bn_bias_gpu, bn_bias, bn_bytes(), cudaMemcpyHostToDevice) );

            SAFE_ALOC_GPU(output_gpu, output_bytes());
            CUDA_SAFE_CALL( cudaMemset(this->output_gpu, 0, output_bytes()) );
            set_input_gpu(prev_layer_gpu);

            return this->ready();
        }

         /* quantization for quantize the initial value to N-bit representation in INT32*/
        void weightQuantization()
        {
            SAFE_ALOC_GPU(weight_qnt_gpu, weight_bytes());
            
            // printf("out_weight_height: %d, out_weight_width: %d\n", weight_height, weight_width);
            cudaGetDeviceProperties(&deviceProp, dev);
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, Quantize_val,  numThreads, 0);
            
            // weight_gpu (float 32-bit) -->  weigth_qnt_gpu (uint 32-bit)
            Quantize_val<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(weight_qnt_gpu, weight_float, weight_size(), w_bit);

            // printf("after quantize_val\n");
            CUDA_SAFE_CALL( cudaPeekAtLastError());
            CUDA_SAFE_CALL( cudaDeviceSynchronize());
        }

        /*Added quantization for decompose the INT32 matrix to N-bit x M x N bit matrix*/
        void weightBitDecomposition()
        {
            SAFE_ALOC_GPU(weight_uin32_dec_gpu, w_bit*weight_size()*sizeof(uin32));

            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, Decompose_bit, numThreads, 0);
            Decompose_bit<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(weight_uin32_dec_gpu, weight_qnt_gpu, weight_size(), w_bit);

            CUDA_SAFE_CALL( cudaPeekAtLastError());
            CUDA_SAFE_CALL( cudaDeviceSynchronize());
        }

        // print the image with float-point value.
        void print_image_10x10_float(float* image){
            printf("\n------print_image_10x10_float-----------\n");
            const int show_height = 10;
            const int show_width = show_height;
            for (int i = 0; i < show_height; i++){
                for (int j = 0; j < show_width; j++){
                    printf("%f ", image[i * weight_height + j]);
                }
                printf("\n");
            }
            printf("=========================================\n");
        }

       // print the image with float-point value.
        void print_image_10x10_float(float* image, const int width){
            printf("\n------print_image_10x10_float-----------\n");
            const int show_height = width;
            const int show_width = width;
            for (int i = 0; i < show_height; i++){
                for (int j = 0; j < show_width; j++){
                    printf("%f ", image[i * width + j]);
                }
                printf("\n");
            }
            printf("=========================================\n");
        }


        // print the image with their int value.
        void print_image_10x10_int(uin32* image){
            printf("\n------print_image_10x10_int-----------\n");
            const int show_height = 32;
            const int show_width = show_height;
            for (int i = 0; i < show_height; i++){
                for (int j = 0; j < show_width; j++){
                    printf("%d ", image[i * weight_height + j]);
                }
                printf("\n");
            }
            printf("=========================================\n");
        }

        // print the image with their int value.
        void print_output_int32(uin32* image){
            printf("\n------print_output_int32-----------\n");
            const int show_height = output_height;
            const int show_width = output_width;

            printf("show_height, %d, show_width: %d\n", show_height, show_width);
            for (int i = 0; i < show_height; i++){
                printf("[%d] ", i);
                for (int j = 0; j < show_width; j++){
                    printf("%u ", image[i * show_width + j]);
                }
                printf("\n");
            }
            printf("=========================================\n");
        }

        // print the image with their decomposed bit in 32-bit.
        void print_weight_int_bit_decompose(uin32* image){
            const int show_height = weight_height;
            const int show_width = 1;
            
            printf("\n------print_weight_int_bit_decompose-----------\n");
            for (int i = 0; i < show_width; i++){
                for (int j = 0; j < show_height; j++){
                    for (int b = w_bit - 1; b >= 0; b--)
                        printf("%d", image[b * weight_size() + i * weight_height + j]);
                    printf(" ");
                }
                printf("\n");
            }
            printf("=========================================\n");
        }
        
        // * print the input image with their decomposed bit-by-bit
        void print_input_int_bit_decompose(uin32* image){
            const int show_height = input_height;
            const int show_width = input_width;

            printf("show_height: %d, show_width: %d", show_height, show_width);
            
            printf("\n------print_input_int_bit_decompose-----------\n");
            for (int i = 0; i < show_height; i++){
                for (int j = 0; j < show_width; j++){
                    for (int b = act_bit - 1; b >= 0; b--) {
                        // printf("\n%d, %d, %d\n", i, j, b);
                        printf("%d", image[b * input_size() + i * show_width + j]);
                    }
                    printf(" ");
                }
                printf("\n");
            }
            printf("=========================================\n");
        }

        // convert a bit-by-bit matrix to uint32 matrix.
        uin32* bit2uint32(uin32* bit_input){

            uin32* uint32input =  NULL;
            SAFE_ALOC_HOST(uint32input, input_size() * sizeof(uin32));
            
            // row-major store
            printf("\n------Output FC Layer bit2uint32-----------\n");
            for (int i = 0; i < input_height; i++){
                for (int j = 0; j < input_width; j++){
                    uin32 tmp = 0;
                    for (int b = act_bit - 1; b >= 0; b--)
                        tmp += (bit_input[b * input_size() + i * input_width + j] << b);
                    uint32input[i * input_width + j] = tmp;
                }
            }
            return uint32input;
        }


        // print the image with their int value.
        void print_image_uint32(uin32* image, const int img_width){
            // printf("\n------print_image_%dx%d_int-----------\n", width, width);
            const int print_range = 8;
            for (int i = 0; i < print_range; i++){
                for (int j = 0; j < print_range; j++){
                    printf("%d ", image[i*img_width + j]);
                }
                printf("\n");
            }
            printf("=========================================\n");
        }


        uin32* download_full_weight()
        {
            const int size = weight_size() * sizeof(float);
            uin32* full_weight = NULL;
            SAFE_ALOC_HOST(full_weight, size);

            uin32* full_weight_gpu = NULL;
            SAFE_ALOC_GPU(full_weight_gpu, size);
            
            CUDA_SAFE_CALL( cudaMemset(full_weight_gpu, 0, size) );
            cudaGetDeviceProperties(&deviceProp, dev);
            
#ifdef NEWFMT
            // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, UnPackFcOutput128FMT, 
            //         numThreads, 0);
            // UnPackFcOutput128FMT<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(
            //         output_gpu, full_output_gpu, output_height, output_width, 3);
#else
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, UnPackFcOutput128, numThreads, 0);
            UnPackFcWeight128_OUTPUT<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(
                    weight_gpu, full_weight_gpu, weight_height, weight_width, w_bit);
#endif
            CUDA_SAFE_CALL( cudaPeekAtLastError());
            CUDA_SAFE_CALL( cudaDeviceSynchronize());

            CUDA_SAFE_CALL( cudaMemcpy(full_weight, full_weight_gpu, size, cudaMemcpyDeviceToHost) );
            CUDA_SAFE_CALL( cudaFree(full_weight_gpu) );

            return full_weight;
        }

        // for validation input at output layer
        uin32* download_full_input()
        {
            const int size = input_bytes();
            uin32* full_input = NULL;
            SAFE_ALOC_HOST(full_input, size);

            uin32* full_input_gpu = NULL;
            SAFE_ALOC_GPU(full_input_gpu, size);
            CUDA_SAFE_CALL( cudaMemset(full_input_gpu, 0, size) );
            
#ifdef NEWFMT
            // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, UnPackFcOutput128FMT, 
            //         numThreads, 0);
            // UnPackFcOutput128FMT<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(
            //         output_gpu, full_output_gpu, output_height, output_width, 3);
#else
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, UnPackFcOutput128, numThreads, 0);
            UnPackFcOutput128<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(
                    input_gpu, full_input_gpu, input_height, input_width, act_bit);
#endif
            CUDA_SAFE_CALL( cudaPeekAtLastError());
            CUDA_SAFE_CALL( cudaDeviceSynchronize());

            CUDA_SAFE_CALL( cudaMemcpy(full_input, full_input_gpu, size, cudaMemcpyDeviceToHost) );
            CUDA_SAFE_CALL( cudaFree(full_input_gpu) );
            
            return full_input;
        }
        
        //* validate output in int32 format.
        uin32* download_output()
        {
            SAFE_ALOC_HOST(output, output_bytes());
            CUDA_SAFE_CALL( cudaMemcpy(output, output_gpu, output_bytes(), cudaMemcpyDeviceToHost) );
            return this->output;
        }

        void release()
        {
            SAFE_FREE_HOST(weight);
            SAFE_FREE_HOST(output);
            SAFE_FREE_HOST(bn_scale);
            SAFE_FREE_HOST(bn_bias);

            SAFE_FREE_GPU(weight_gpu);
            SAFE_FREE_GPU(output_gpu);
            SAFE_FREE_GPU(gpu);
            SAFE_FREE_GPU(bn_scale_gpu);
            SAFE_FREE_GPU(bn_bias_gpu);
        }
        ~Out128LayerParam() { release(); }
    public:
        //Input
        uin32* input;
        uin32* input_gpu;
        int input_width;
        int input_height;
        
        //Weight
        float* weight;
        float* weight_float=NULL;
        uin32* weight_gpu;
        int weight_width;
        int weight_height;

        //Output
        uin32* output;
        uin32* output_gpu;
        int output_height;
        int output_width;

        // support for arbitary precision.
        int w_bit, act_bit;

        uin32* weight_uin32;
        uin32* weight_qnt_gpu;
        uin32* weight_uin32_dec;
        uin32* weight_uin32_dec_gpu;

        int numThreads = 1024;
        int numBlocksPerSm;
        cudaDeviceProp deviceProp;

        //Batch normalization
        bool has_bn;
        float* bn_scale;
        float* bn_scale_gpu;
        float* bn_bias;
        float* bn_bias_gpu;

        //GPU shadow
        Out128LayerParam* gpu;
        char name[8];
};
#endif
