#include "caffe/layers/dense_block_layer.hpp"
#include "caffe/layers/batch_norm_layer.hpp"
#include "caffe/util/device_alternate.hpp"

namespace caffe {

//template <typename Dtype>
//void caffe_cublas_mul(const int N, const Dtype* a, const Dtype* b, Dtype* y);
//
//template <> 
//void caffe_cublas_mul<float>(const int N, const float* a, const float* b, float* y)
//{
//  float one(1.);
//  float zero(0.);
//  CUBLAS_CHECK(cublasSsbmv(Caffe::cublas_handle(),
//    CUBLAS_FILL_MODE_LOWER, N, 0,
//    &one, a, 1, b, 1,
//    &zero, y, 1));
//}
//template <>
//void caffe_cublas_mul<double>(const int N, const double* a, const double* b, double* y)
//{
//  double one(1.);
//  double zero(0.);
//  CUBLAS_CHECK(cublasDsbmv(Caffe::cublas_handle(),
//    CUBLAS_FILL_MODE_LOWER, N, 0,
//    &one, a, 1, b, 1,
//    &zero, y, 1));
//}

#ifdef USE_CUDNN
template <typename Dtype>
dense_block::StaticVariable<Dtype> dense_block::StaticVariable<Dtype>::instance_;

template <typename Dtype>
dense_block::StaticVariable<Dtype>::StaticVariable() : fast_scale_fwd_op_desc_(NULL)
{
  CUDNN_CHECK(cudnnCreateOpTensorDescriptor(&fast_scale_fwd_op_desc_));
  CUDNN_CHECK(cudnnSetOpTensorDescriptor(fast_scale_fwd_op_desc_, CUDNN_OP_TENSOR_MUL, cudnn::dataType<Dtype>::type, CUDNN_PROPAGATE_NAN));
}

template <typename Dtype>
dense_block::StaticVariable<Dtype>::~StaticVariable()
{
  if (fast_scale_fwd_op_desc_)
    CUDNN_CHECK(cudnnDestroyOpTensorDescriptor(fast_scale_fwd_op_desc_));
}

template <typename Dtype>
static void reduce_nhw(cudnnHandle_t handle, 
                       Dtype alpha_, cudnnTensorDescriptor_t nchw_desc_, const Dtype* nchw_ptr_,
                       Dtype beta_, cudnnTensorDescriptor_t c_desc_, Dtype* c_ptr_)
{
  Dtype a(alpha_);
  Dtype b(beta_);
  // we assume that these 2 tensor descriptors are configured to right data type
  CUDNN_CHECK(cudnnConvolutionBackwardBias(handle,
    &a, nchw_desc_, nchw_ptr_, &b, c_desc_, c_ptr_));
}

template <typename Dtype>
void dense_block::ScaleLayerFastForward(cudnnHandle_t handle,
  cudnnTensorDescriptor_t bottom_desc, Blob<Dtype>* bottom,
  cudnnTensorDescriptor_t top_desc, Blob<Dtype> *top,
  cudnnTensorDescriptor_t scale_bias_desc, ScaleLayer<Dtype> *scale_layer)
{
  CHECK_NE(bottom, top) << "ScaleLayerFastForward dose not support in-place computation";

  Dtype one(1.);
  Dtype zero(0.);

  CUDNN_CHECK(cudnnOpTensor(handle, StaticVariable<Dtype>::get().fast_scale_fwd_op_desc(),
    &one, bottom_desc, bottom->gpu_data(),
    &one, scale_bias_desc, scale_layer->blobs()[0]->gpu_data(),
    &zero, top_desc, top->mutable_gpu_data()));

  CUDNN_CHECK(cudnnAddTensor(handle,
    &one, scale_bias_desc, scale_layer->blobs()[1]->gpu_data(),
    &one, top_desc, top->mutable_gpu_data()));

}

template <typename Dtype>
void dense_block::ScaleLayerFastBackward(cudnnHandle_t handle,
  cudnnTensorDescriptor_t scale_bias_desc, ScaleLayer<Dtype> *scale_layer,
  cudnnTensorDescriptor_t top_desc, Blob<Dtype> *top,
  cudnnTensorDescriptor_t bottom_desc, Blob<Dtype>* bottom)
{
  CHECK_NE(bottom, top) << "ScaleLayerFastForward dose not support in-place computation";
  
  Dtype one(1.);
  Dtype zero(0.);

  // gradient w.r.t bias
  reduce_nhw(handle, one, top_desc,   top->gpu_diff(),     one, scale_bias_desc, scale_layer->blobs()[1]->mutable_gpu_diff());

  // gradient w.r.t scale
  caffe_gpu_mul(bottom->count(), bottom->gpu_data(), top->gpu_diff(), bottom->mutable_gpu_diff());
  reduce_nhw(handle, one, bottom_desc, bottom->gpu_diff(), one, scale_bias_desc, scale_layer->blobs()[0]->mutable_gpu_diff());

  // gradient w.r.t bottom
  CUDNN_CHECK(cudnnOpTensor(handle, StaticVariable<Dtype>::get().fast_scale_fwd_op_desc(),
    &one, top_desc, top->gpu_diff(),
    &one, scale_bias_desc, scale_layer->blobs()[0]->gpu_data(),
    &zero, bottom_desc, bottom->mutable_gpu_diff()));
}

namespace dense_block {
  template void ScaleLayerFastForward(cudnnHandle_t handle,
    cudnnTensorDescriptor_t bottom_desc, Blob<float>* bottom,
    cudnnTensorDescriptor_t top_desc, Blob<float> *top,
    cudnnTensorDescriptor_t scale_bias_desc, ScaleLayer<float> *scale_layer);
  template void ScaleLayerFastForward(cudnnHandle_t handle,
    cudnnTensorDescriptor_t bottom_desc, Blob<double>* bottom,
    cudnnTensorDescriptor_t top_desc, Blob<double> *top,
    cudnnTensorDescriptor_t scale_bias_desc, ScaleLayer<double> *scale_layer);
  template void ScaleLayerFastBackward(cudnnHandle_t handle,
    cudnnTensorDescriptor_t scale_bias_desc, ScaleLayer<float> *scale_layer,
    cudnnTensorDescriptor_t top_desc, Blob<float> *top,
    cudnnTensorDescriptor_t bottom_desc, Blob<float>* bottom);
  template void ScaleLayerFastBackward(cudnnHandle_t handle,
    cudnnTensorDescriptor_t scale_bias_desc, ScaleLayer<double> *scale_layer,
    cudnnTensorDescriptor_t top_desc, Blob<double> *top,
    cudnnTensorDescriptor_t bottom_desc, Blob<double>* bottom);
} // namespace dense_block
#endif // USE_CUNN

template <typename Dtype>
inline static void caffe_gpu_copy_async(const int N, const Dtype* X, Dtype* Y, const cudaStream_t& stream) {
  if (X != Y && Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
      // NOLINT_NEXT_LINE(caffe/alt_fn)
      CUDA_CHECK(cudaMemcpyAsync(Y, X, sizeof(Dtype) * N, cudaMemcpyDeviceToDevice, stream));
#else
      NO_GPU;
#endif
  }
}

template <typename Dtype>
static void assemble_maps_gpu(const int n, const int h, const int w, const int c0, const int c_add,
                              Dtype* dst, const Dtype* new_map, const cudaStream_t& stream)
{
  // c0 = #feature-maps BEFORE assemble
  // c_add = #feature-maps to be added
  const int c1 = c0 + c_add;
  const int c_stride = h * w;
  const int src_stride = c0 * c_stride;
  const int dst_stride = c1 * c_stride;
  const int new_stride = c_add * c_stride;
  
  const Dtype* new_map_ptr = new_map + (n - 1) * new_stride;
  const Dtype *src_ptr = dst + (n - 1) * src_stride;
  Dtype *dst_ptr = dst + (n - 1) * dst_stride;
  Dtype *dst_ptr_for_new = dst_ptr + src_stride;
  
  const int src_count = c0 * c_stride;
  const int new_count = c_add * c_stride;
  
  for (int i = n - 1; i >= 0; i --, 
    new_map_ptr -= new_stride, 
    src_ptr     -= src_stride,
    dst_ptr     -= dst_stride,
    dst_ptr_for_new -= dst_stride)
  {
    if (dst_ptr > src_ptr && dst_ptr - src_ptr < src_count)
      // dst_ptr is pointing within the src region [src_ptr, src_ptr + src_count]
      // directly memcpy will cause data lossing, so we copy channel by channel from back to front
    {
      const int batch = dst_ptr - src_ptr;
      int remains = src_count;
      Dtype* p_dst = dst_ptr + src_count - batch;
      const Dtype* p_src = src_ptr + src_count - batch;
      for (; remains >= batch; remains -= batch, p_dst -= batch, p_src -= batch)
        caffe_gpu_copy_async(batch, p_src, p_dst, stream);
      if (remains)
        caffe_gpu_copy_async(remains, src_ptr, dst_ptr, stream);
    }
    else
      caffe_gpu_copy_async(src_count, src_ptr, dst_ptr, stream);
    
    caffe_gpu_copy_async(new_count, new_map_ptr, dst_ptr_for_new, stream);
  }
  
}

template <typename Dtype>
static void disassemble_maps_gpu(const int n, const int h, const int w, const int c0, const int c_add,
                                 Dtype* src, Dtype* out_map, const cudaStream_t& stream)
{
  // c0 = #feature-maps AFTER disassemble
  // c_add = #feature-maps in out_map
  const int c1 = c0 + c_add;
  const int c_stride = h * w;
  const int src_stride = c1 * c_stride;
  const int dst_stride = c0 * c_stride;
  const int out_stride = c_add * c_stride;
  
  Dtype* out_map_ptr = out_map;
  Dtype *dst_ptr = src;
  const Dtype *src_ptr = src;
  const Dtype *src_ptr_for_out = src_ptr + dst_stride;
  
  const int dst_count = c0 * c_stride;
  const int out_count = c_add * c_stride;
  
  for (int i = 0; i < n; i ++,
    out_map_ptr += out_stride,
    dst_ptr     += dst_stride,
    src_ptr     += src_stride,
    src_ptr_for_out += src_stride)
  {
    caffe_gpu_copy_async(out_count, src_ptr_for_out, out_map_ptr, stream);
    // CAUTION: in memcpy, there shoud not be any overlap between
    //          src memory region and dst memory region
    if (src_ptr > dst_ptr && src_ptr < dst_ptr + dst_count)
    {
      const int batch = src_ptr - dst_ptr;
      int remains = dst_count;
      Dtype* p_dst = dst_ptr;
      const Dtype* p_src = src_ptr;
      for (; remains >= batch; remains -= batch, p_dst += batch, p_src += batch)
        caffe_gpu_copy_async(batch, p_src, p_dst, stream);
      if (remains)
        caffe_gpu_copy_async(remains, p_src, p_dst, stream);
    }
    else
      caffe_gpu_copy_async(dst_count, src_ptr, dst_ptr, stream);
  }
}

template <typename Dtype>
static void assemble_maps_gpu_adding_part(const int n, const int h, const int w, const int c0, const int c_add,
                                          Dtype* dst, const Dtype* new_map, const cudaStream_t& stream)
{
  // c0 = #feature-maps BEFORE assemble
  // c_add = #feature-maps to be added
  const int c1 = c0 + c_add;
  const int c_stride = h * w;
  const int src_stride = c0 * c_stride;
  const int dst_stride = c1 * c_stride;
  const int new_stride = c_add * c_stride;
  
  const Dtype* new_map_ptr = new_map + (n - 1) * new_stride;
  const Dtype *src_ptr = dst + (n - 1) * src_stride;
  Dtype *dst_ptr = dst + (n - 1) * dst_stride;
  Dtype *dst_ptr_for_new = dst_ptr + src_stride;
  
  const int src_count = c0 * c_stride;
  const int new_count = c_add * c_stride;
  
  for (int i = n - 1; i >= 0; i --, 
    new_map_ptr -= new_stride, 
    src_ptr     -= src_stride,
    dst_ptr     -= dst_stride,
    dst_ptr_for_new -= dst_stride)
  {
    //if (dst_ptr > src_ptr && dst_ptr - src_ptr < src_count)
    //  // dst_ptr is pointing within the src region [src_ptr, src_ptr + src_count]
    //  // directly memcpy will cause data lossing, so we copy channel by channel from back to front
    //{
    //  const int batch = dst_ptr - src_ptr;
    //  int remains = src_count;
    //  Dtype* p_dst = dst_ptr + src_count - batch;
    //  const Dtype* p_src = src_ptr + src_count - batch;
    //  for (; remains >= batch; remains -= batch, p_dst -= batch, p_src -= batch)
    //    caffe_gpu_copy_async(batch, p_src, p_dst, stream);
    //  if (remains)
    //    caffe_gpu_copy_async(remains, src_ptr, dst_ptr, stream);
    //}
    //else
    //  caffe_gpu_copy_async(src_count, src_ptr, dst_ptr, stream);
    
    caffe_gpu_copy_async(new_count, new_map_ptr, dst_ptr_for_new, stream);
  }
  
}

template <typename Dtype>
static void disassemble_maps_gpu_adding_part(const int n, const int h, const int w, const int c0, const int c_add,
                                             Dtype* src, Dtype* out_map, const cudaStream_t& stream)
{
  // c0 = #feature-maps AFTER disassemble
  // c_add = #feature-maps in out_map
  const int c1 = c0 + c_add;
  const int c_stride = h * w;
  const int src_stride = c1 * c_stride;
  const int dst_stride = c0 * c_stride;
  const int out_stride = c_add * c_stride;
  
  Dtype* out_map_ptr = out_map;
  Dtype *dst_ptr = src;
  const Dtype *src_ptr = src;
  const Dtype *src_ptr_for_out = src_ptr + dst_stride;
  
  const int dst_count = c0 * c_stride;
  const int out_count = c_add * c_stride;
  
  for (int i = 0; i < n; i ++,
    out_map_ptr += out_stride,
    dst_ptr     += dst_stride,
    src_ptr     += src_stride,
    src_ptr_for_out += src_stride)
  {
    caffe_gpu_copy_async(out_count, src_ptr_for_out, out_map_ptr, stream);
    // CAUTION: in memcpy, there shoud not be any overlap between
    //          src memory region and dst memory region
    //if (src_ptr > dst_ptr && src_ptr < dst_ptr + dst_count)
    //{
    //  const int batch = src_ptr - dst_ptr;
    //  int remains = dst_count;
    //  Dtype* p_dst = dst_ptr;
    //  const Dtype* p_src = src_ptr;
    //  for (; remains >= batch; remains -= batch, p_dst += batch, p_src += batch)
    //    caffe_gpu_copy_async(batch, p_src, p_dst, stream);
    //  if (remains)
    //    caffe_gpu_copy_async(remains, p_src, p_dst, stream);
    //}
    //else
    //  caffe_gpu_copy_async(dst_count, src_ptr, dst_ptr, stream);
  }
}

template <typename Dtype>
static void assemble_maps_gpu_origin_part(const int n, const int h, const int w, const int c0, const int c_add,
                                          Dtype* dst, const Dtype* new_map, const cudaStream_t& stream)
{
  // c0 = #feature-maps BEFORE assemble
  // c_add = #feature-maps to be added
  const int c1 = c0 + c_add;
  const int c_stride = h * w;
  const int src_stride = c0 * c_stride;
  const int dst_stride = c1 * c_stride;
  const int new_stride = c_add * c_stride;
  
  const Dtype* new_map_ptr = new_map + (n - 1) * new_stride;
  const Dtype *src_ptr = dst + (n - 1) * src_stride;
  Dtype *dst_ptr = dst + (n - 1) * dst_stride;
  Dtype *dst_ptr_for_new = dst_ptr + src_stride;
  
  const int src_count = c0 * c_stride;
  const int new_count = c_add * c_stride;
  
  for (int i = n - 1; i >= 0; i --, 
    new_map_ptr -= new_stride, 
    src_ptr     -= src_stride,
    dst_ptr     -= dst_stride,
    dst_ptr_for_new -= dst_stride)
  {
    if (dst_ptr > src_ptr && dst_ptr - src_ptr < src_count)
      // dst_ptr is pointing within the src region [src_ptr, src_ptr + src_count]
      // directly memcpy will cause data lossing, so we copy channel by channel from back to front
    {
      const int batch = dst_ptr - src_ptr;
      int remains = src_count;
      Dtype* p_dst = dst_ptr + src_count - batch;
      const Dtype* p_src = src_ptr + src_count - batch;
      for (; remains >= batch; remains -= batch, p_dst -= batch, p_src -= batch)
        caffe_gpu_copy_async(batch, p_src, p_dst, stream);
      if (remains)
        caffe_gpu_copy_async(remains, src_ptr, dst_ptr, stream);
    }
    else
      caffe_gpu_copy_async(src_count, src_ptr, dst_ptr, stream);
    
    //caffe_gpu_copy_async(new_count, new_map_ptr, dst_ptr_for_new, stream);
  }
  
}

template <typename Dtype>
static void disassemble_maps_gpu_origin_part(const int n, const int h, const int w, const int c0, const int c_add,
                                             Dtype* src, Dtype* out_map, const cudaStream_t& stream)
{
  // c0 = #feature-maps AFTER disassemble
  // c_add = #feature-maps in out_map
  const int c1 = c0 + c_add;
  const int c_stride = h * w;
  const int src_stride = c1 * c_stride;
  const int dst_stride = c0 * c_stride;
  const int out_stride = c_add * c_stride;
  
  Dtype* out_map_ptr = out_map;
  Dtype *dst_ptr = src;
  const Dtype *src_ptr = src;
  const Dtype *src_ptr_for_out = src_ptr + dst_stride;
  
  const int dst_count = c0 * c_stride;
  const int out_count = c_add * c_stride;
  
  for (int i = 0; i < n; i ++,
    out_map_ptr += out_stride,
    dst_ptr     += dst_stride,
    src_ptr     += src_stride,
    src_ptr_for_out += src_stride)
  {
    //caffe_gpu_copy_async(out_count, src_ptr_for_out, out_map_ptr, stream);
    // CAUTION: in memcpy, there shoud not be any overlap between
    //          src memory region and dst memory region
    if (src_ptr > dst_ptr && src_ptr < dst_ptr + dst_count)
    {
      const int batch = src_ptr - dst_ptr;
      int remains = dst_count;
      Dtype* p_dst = dst_ptr;
      const Dtype* p_src = src_ptr;
      for (; remains >= batch; remains -= batch, p_dst += batch, p_src += batch)
        caffe_gpu_copy_async(batch, p_src, p_dst, stream);
      if (remains)
        caffe_gpu_copy_async(remains, p_src, p_dst, stream);
    }
    else
      caffe_gpu_copy_async(dst_count, src_ptr, dst_ptr, stream);
  }
}

template <typename Dtype>
static void updateMovingAverage(BatchNormLayer<Dtype>* layer)
{
  // this is the implementation of update moving average from caffe's BatchNorm
  Dtype moving_average_fraction_ = 
    layer->layer_param().batch_norm_param().moving_average_fraction();

  layer->blobs()[2]->mutable_cpu_data()[0] *= moving_average_fraction_;
  layer->blobs()[2]->mutable_cpu_data()[0] -= 1;
}
#if 0 // debug utils
static cudnnTensorDescriptor_t copyTensor4dDesc(cudnnTensorDescriptor_t tensorDesc)
{
  cudnnTensorDescriptor_t ret;
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&ret));
  
  int n, c, h, w, ns, cs, hs, ws;
  cudnnDataType_t type;
  CUDNN_CHECK(cudnnGetTensor4dDescriptor(tensorDesc, &type, &n, &c, &h, &w, &ns, &cs, &hs, &ws));

  CUDNN_CHECK(cudnnSetTensor4dDescriptor(ret, CUDNN_TENSOR_NCHW, type, n, c, h, w));
  return ret;
}
static void dumpTensorShape(cudnnTensorDescriptor_t tensorDesc)
{
  char tmp[20];
  string ret = "";
  int maxDim = 100, actualDim;
  vector<int> dim(maxDim, 0), stride(maxDim, 0);
  cudnnDataType_t type;
  CUDNN_CHECK(cudnnGetTensorNdDescriptor(tensorDesc, maxDim, &type, &actualDim, dim.data(), stride.data()));
  for (int i = 0; i < actualDim; i ++)
  {
    sprintf(tmp, "%d(%d) ", dim[i], stride[i]);
    ret += tmp;
  }
  LOG(INFO) << ret;
}
template <typename Dtype>
static void fillArray(int count, Dtype *dst, Dtype val)
{
  for (int i = 0; i < count; i ++)
    dst[i] = val;
}
template <typename Dtype>
static void fillArray(int count, Dtype *dst)
{
  for (int i = 0; i < count; i ++)
    dst[i] = (rand() % 2 ? -1 : 1) * Dtype(rand() - RAND_MAX / 2) * 20 / RAND_MAX;
}
#endif // debug utils

#ifdef USE_CUDNN

template <typename Dtype>
inline static double cudnnGetBNEps(Dtype val)
{
  double ret(val);
  return ret < CUDNN_BN_MIN_EPSILON ? CUDNN_BN_MIN_EPSILON : ret;
}

template <typename Dtype>
void DenseBlockLayer<Dtype>::ForwardInference_gpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top)
{
  const vector<int> shape(bottom[0]->shape());
  const int n = shape[0];
  const int k0= shape[1];
  const int h = shape[2];
  const int w = shape[3];

  // copy the input data into working space 
  caffe_copy(bottom[0]->count(), bottom[0]->gpu_data(), maps_diff_.mutable_gpu_data());

  if (pre_bn_layer_->blobs()[2]->cpu_data()[0] > 0)
    convertBatchNormParams();

  Dtype one(1.f), zero(0.f);

  for (int l = 0; l < num_layers_; l++)
  {
    vector<Blob<Dtype>*> the_input_lth(1, input_lth_[l].get());
    vector<Blob<Dtype>*> the_conv3x3_inter_l(1, conv3x3_inter_[l].get());
    vector<Blob<Dtype>*> the_output_lth(1, output_lth_[l].get());

    if (use_bottleneck_)
    {
      vector<Blob<Dtype>*> the_bottleneck_inter_l(1, bottleneck_inter_[l].get());

      {
        //updateMovingAverage( (BatchNormLayer<Dtype>*)(bottle_bn_layers_[l].get()) );
        //CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
        //  cudnn_handle_, CUDNN_BATCHNORM_SPATIAL, &one, &zero,
        //  input_desc_[l], the_input_lth[0]->gpu_data(),
        //  input_desc_[l], the_conv3x3_inter_l[0]->mutable_gpu_data(),
        //  input_scale_bias_desc_[l], bottle_scale_layers_[l]->blobs()[0]->gpu_data(), bottle_scale_layers_[l]->blobs()[1]->gpu_data(),
        //  1 / bottle_bn_layers_[l]->blobs()[2]->cpu_data()[0],
        //  bottle_bn_layers_[l]->blobs()[0]->mutable_gpu_data(),
        //  bottle_bn_layers_[l]->blobs()[1]->mutable_gpu_data(),
        //  cudnnGetBNEps(bottle_bn_layers_[l]->layer_param().batch_norm_param().eps()),
        //  bottleneck_bn_mean_var_[l]->mutable_gpu_data(), bottleneck_bn_mean_var_[l]->mutable_gpu_diff()
        //));
        CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
          cudnn_handle_, CUDNN_BATCHNORM_SPATIAL, &one, &zero,
          input_desc_[l], the_input_lth[0]->gpu_data(),
          input_desc_[l], the_conv3x3_inter_l[0]->mutable_gpu_data(),
          input_scale_bias_desc_[l], 
          bottle_scale_layers_[l]->blobs()[0]->gpu_data(), 
          bottle_scale_layers_[l]->blobs()[1]->gpu_data(),
          bottle_bn_layers_[l]->blobs()[0]->gpu_data(),
          bottle_bn_layers_[l]->blobs()[1]->gpu_data(),
          cudnnGetBNEps(bottle_bn_layers_[l]->layer_param().batch_norm_param().eps())
        ));
      }
      //bottle_bn_layers_[l]->Forward(the_input_lth, the_conv3x3_inter_l);
      // (in gpu) async "assemble" (original part) can start here to prepare for next conv block
      assemble_maps_gpu_origin_part(n, h, w, k0 + l * growth_rate_, growth_rate_,
        maps_diff_.mutable_gpu_data(), (const Dtype*)NULL /*output_lth_[l]->gpu_data()*/, dataCopyStream_);

      //bottle_scale_layers_[l]->Forward(the_conv3x3_inter_l, the_conv3x3_inter_l);
      bottle_relu_layers_[l]->Forward(the_conv3x3_inter_l, the_conv3x3_inter_l);

      conv1x1_layers_[l]->Forward(the_conv3x3_inter_l, the_bottleneck_inter_l);

      //bn_layers_[l]->Forward(the_bottleneck_inter_l, the_bottleneck_inter_l);
      //scale_layers_[l]->Forward(the_bottleneck_inter_l, the_bottleneck_inter_l);
      {
        caffe_copy(the_bottleneck_inter_l[0]->count(),
                   the_bottleneck_inter_l[0]->gpu_data(), 
                   bottleneck_scale_tmp_[l]->mutable_gpu_data());
        //updateMovingAverage( (BatchNormLayer<Dtype>*)(bn_layers_[l].get()) );
        //CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
        //  cudnn_handle_, CUDNN_BATCHNORM_SPATIAL, &one, &zero,
        //  bottleneck_inter_desc_, bottleneck_scale_tmp_[l]->gpu_data(),
        //  bottleneck_inter_desc_, the_bottleneck_inter_l[0]->mutable_gpu_data(),
        //  bottleneck_scale_bias_desc_, scale_layers_[l]->blobs()[0]->gpu_data(), scale_layers_[l]->blobs()[1]->gpu_data(),
        //  1 / bn_layers_[l]->blobs()[2]->cpu_data()[0],
        //  bn_layers_[l]->blobs()[0]->mutable_gpu_data(),
        //  bn_layers_[l]->blobs()[1]->mutable_gpu_data(),
        //  cudnnGetBNEps(bn_layers_[l]->layer_param().batch_norm_param().eps()),
        //  bn_mean_var_[l]->mutable_gpu_data(), bn_mean_var_[l]->mutable_gpu_diff()
        //));
        CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
          cudnn_handle_, CUDNN_BATCHNORM_SPATIAL, &one, &zero,
          bottleneck_inter_desc_, bottleneck_scale_tmp_[l]->gpu_data(),
          bottleneck_inter_desc_, the_bottleneck_inter_l[0]->mutable_gpu_data(),
          bottleneck_scale_bias_desc_,
          scale_layers_[l]->blobs()[0]->gpu_data(),
          scale_layers_[l]->blobs()[1]->gpu_data(),
          bn_layers_[l]->blobs()[0]->gpu_data(),
          bn_layers_[l]->blobs()[1]->gpu_data(),
          cudnnGetBNEps(bn_layers_[l]->layer_param().batch_norm_param().eps())
        ));
      }
      relu_layers_[l]->Forward(the_bottleneck_inter_l, the_bottleneck_inter_l);

      conv3x3_layers_[l]->Forward(the_bottleneck_inter_l, the_output_lth);

    }
    else
    {
      {
        //updateMovingAverage( (BatchNormLayer<Dtype>*)(bn_layers_[l].get()) );
        //CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
        //  cudnn_handle_, CUDNN_BATCHNORM_SPATIAL, &one, &zero,
        //  input_desc_[l], the_input_lth[0]->gpu_data(),
        //  input_desc_[l], the_conv3x3_inter_l[0]->mutable_gpu_data(),
        //  input_scale_bias_desc_[l], scale_layers_[l]->blobs()[0]->gpu_data(), scale_layers_[l]->blobs()[1]->gpu_data(),
        //  1 / bn_layers_[l]->blobs()[2]->cpu_data()[0],
        //  bn_layers_[l]->blobs()[0]->mutable_gpu_data(),
        //  bn_layers_[l]->blobs()[1]->mutable_gpu_data(),
        //  cudnnGetBNEps(bn_layers_[l]->layer_param().batch_norm_param().eps()),
        //  bn_mean_var_[l]->mutable_gpu_data(), bn_mean_var_[l]->mutable_gpu_diff()
        //));
        CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
          cudnn_handle_, CUDNN_BATCHNORM_SPATIAL, &one, &zero,
          input_desc_[l], the_input_lth[0]->gpu_data(),
          input_desc_[l], the_conv3x3_inter_l[0]->mutable_gpu_data(),
          input_scale_bias_desc_[l],
          scale_layers_[l]->blobs()[0]->gpu_data(),
          scale_layers_[l]->blobs()[1]->gpu_data(),
          bn_layers_[l]->blobs()[0]->gpu_data(),
          bn_layers_[l]->blobs()[1]->gpu_data(),
          cudnnGetBNEps(bn_layers_[l]->layer_param().batch_norm_param().eps())
        ));
      }
      //bn_layers_[l]->Forward(the_input_lth, the_conv3x3_inter_l);

      // (in gpu) async "assemble" (original part) can start here to prepare for next conv block
      assemble_maps_gpu_origin_part(n, h, w, k0 + l * growth_rate_, growth_rate_,
        maps_diff_.mutable_gpu_data(), (const Dtype*)NULL /*output_lth_[l]->gpu_data()*/, dataCopyStream_);

      //scale_layers_[l]->Forward(the_conv3x3_inter_l, the_conv3x3_inter_l);
      relu_layers_[l]->Forward(the_conv3x3_inter_l, the_conv3x3_inter_l);

      conv3x3_layers_[l]->Forward(the_conv3x3_inter_l, the_output_lth);

    }

    if (use_dropout_)
    {
      dropout_layers_[l]->Forward(the_output_lth, the_output_lth);
    }


    // (in gpu) start async "assemble" (adding part) for this conv block
    //assemble_maps(n, h, w, k0 + l * growth_rate_, growth_rate_,
    //  maps_diff_.mutable_cpu_data(), output_lth_[l]->cpu_data());
    assemble_maps_gpu_adding_part(n, h, w, k0 + l * growth_rate_, growth_rate_,
      maps_diff_.mutable_gpu_data(), output_lth_[l]->gpu_data(), dataCopyStream_);

    // (in gpu) synchronize "assemble" here so we can start next conv block
    CUDA_CHECK(cudaStreamSynchronize(dataCopyStream_));
  }

  //pre_bn_layer_->Forward(vector<Blob<Dtype>*>(1, &maps_diff_), vector<Blob<Dtype>*>(1, &maps_diff_));
  //post_scale_layer_->Forward(vector<Blob<Dtype>*>(1, &maps_diff_), top);
  {
    //updateMovingAverage( (BatchNormLayer<Dtype>*)(pre_bn_layer_.get()) );
    //CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
    //  cudnn_handle_, CUDNN_BATCHNORM_SPATIAL, &one, &zero,
    //  final_output_desc_, maps_diff_.gpu_data(),
    //  final_output_desc_, top[0]->mutable_gpu_data(),
    //  scale_bias_desc_, post_scale_layer_->blobs()[0]->gpu_data(), post_scale_layer_->blobs()[1]->gpu_data(),
    //  1 / pre_bn_layer_->blobs()[2]->cpu_data()[0],
    //  pre_bn_layer_->blobs()[0]->mutable_gpu_data(),
    //  pre_bn_layer_->blobs()[1]->mutable_gpu_data(),
    //  cudnnGetBNEps(pre_bn_layer_->layer_param().batch_norm_param().eps()),
    //  pre_bn_mean_var_.mutable_gpu_data(), pre_bn_mean_var_.mutable_gpu_diff()
    //));
    CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
      cudnn_handle_, CUDNN_BATCHNORM_SPATIAL, &one, &zero,
      final_output_desc_, maps_diff_.gpu_data(),
      final_output_desc_, top[0]->mutable_gpu_data(),
      scale_bias_desc_, 
      post_scale_layer_->blobs()[0]->gpu_data(), 
      post_scale_layer_->blobs()[1]->gpu_data(),
      pre_bn_layer_->blobs()[0]->gpu_data(),
      pre_bn_layer_->blobs()[1]->gpu_data(),
      cudnnGetBNEps(pre_bn_layer_->layer_param().batch_norm_param().eps())
    ));
  }
  post_relu_layer_->Forward(top, top);
}

template
void DenseBlockLayer<float>::ForwardInference_gpu(const vector<Blob<float>*>& bottom,
  const vector<Blob<float>*>& top);
template
void DenseBlockLayer<double>::ForwardInference_gpu(const vector<Blob<double>*>& bottom,
  const vector<Blob<double>*>& top);

template <typename Dtype>
void DenseBlockLayer<Dtype>::convertBatchNormParams()
{
  CHECK(0 < pre_bn_layer_->blobs()[2]->cpu_data()[0]) << "already in cudnn defintion";
  caffe_gpu_scal(pre_bn_layer_->blobs()[0]->count(), 
                 1 / pre_bn_layer_->blobs()[2]->cpu_data()[0], 
                 pre_bn_layer_->blobs()[0]->mutable_gpu_data());
  caffe_gpu_scal(pre_bn_layer_->blobs()[1]->count(), 
                 1 / pre_bn_layer_->blobs()[2]->cpu_data()[0], 
                 pre_bn_layer_->blobs()[1]->mutable_gpu_data());
  pre_bn_layer_->blobs()[2]->mutable_cpu_data()[0] = -pre_bn_layer_->blobs()[2]->cpu_data()[0];

  for (size_t i = 0; i < bn_layers_.size(); i++)
  {
    CHECK(0 < bn_layers_[i]->blobs()[2]->cpu_data()[0]) << "already in cudnn defintion";
    caffe_gpu_scal(bn_layers_[i]->blobs()[0]->count(),
                   1 / bn_layers_[i]->blobs()[2]->cpu_data()[0],
                   bn_layers_[i]->blobs()[0]->mutable_gpu_data());
    caffe_gpu_scal(bn_layers_[i]->blobs()[1]->count(),
                   1 / bn_layers_[i]->blobs()[2]->cpu_data()[0],
                   bn_layers_[i]->blobs()[1]->mutable_gpu_data());
    bn_layers_[i]->blobs()[2]->mutable_cpu_data()[0] = -bn_layers_[i]->blobs()[2]->cpu_data()[0];
  }

  for (size_t i = 0; i < bottle_bn_layers_.size(); i++)
  {
    CHECK(0 < bottle_bn_layers_[i]->blobs()[2]->cpu_data()[0]) << "already in cudnn defintion";
    caffe_gpu_scal(bottle_bn_layers_[i]->blobs()[0]->count(),
                   1 / bottle_bn_layers_[i]->blobs()[2]->cpu_data()[0],
                   bottle_bn_layers_[i]->blobs()[0]->mutable_gpu_data());
    caffe_gpu_scal(bottle_bn_layers_[i]->blobs()[1]->count(),
                   1 / bottle_bn_layers_[i]->blobs()[2]->cpu_data()[0],
                   bottle_bn_layers_[i]->blobs()[1]->mutable_gpu_data());
    bottle_bn_layers_[i]->blobs()[2]->mutable_cpu_data()[0] = -bottle_bn_layers_[i]->blobs()[2]->cpu_data()[0];
  }

}
#endif // USE_CUDNN

template <typename Dtype>
void DenseBlockLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top)
{
  const vector<int> shape(bottom[0]->shape());
  const int n = shape[0];
  const int k0= shape[1];
  const int h = shape[2];
  const int w = shape[3];

  CHECK_EQ(k0 + num_layers_ * growth_rate_, top[0]->shape()[1])
    << "Invalid top shape according to k0 + num_layers_ * growth_rate_";

#ifdef USE_CUDNN
  if (this->phase_ == TEST)
  {
    this->ForwardInference_gpu(bottom, top);
    return;
  }
#endif // USE_CUDNN

  // copy the input data into working space 
  caffe_copy(bottom[0]->count(), bottom[0]->gpu_data(), maps_diff_.mutable_gpu_data());

#ifdef USE_CUDNN

  // we use BN's blobs[2] to mark whether the params is in caffe
  // definition all cudnn definition 
  // in caffe, blobs[2] > 0, mean = blobs[0] / blobs[2], var = blobs[1] / blobs[1]
  // in cudnn, blobs[2] < 0, mean = blobs[0], var = blobs[1]
  // cudnn=>caffe, blobs[2] = -blobs[2], blobs[0] *= blobs[2], blobs[1] *= blobs[2]
  // caffe=>cudnn, blobs[0] /= blobs[2], blobs[1] /= blobs[2], blobs[2] = -blobs[2]
  if (pre_bn_layer_->blobs()[2]->cpu_data()[0] > 0)
    convertBatchNormParams(); 

  Dtype one(1.f), zero(0.f);

  for (int l = 0; l < num_layers_; l++)
  {
    vector<Blob<Dtype>*> the_input_lth(1, input_lth_[l].get());
    vector<Blob<Dtype>*> the_conv3x3_inter_l(1, conv3x3_inter_[l].get());
    vector<Blob<Dtype>*> the_output_lth(1, output_lth_[l].get());

    if (use_bottleneck_)
    {
      vector<Blob<Dtype>*> the_bottleneck_inter_l(1, bottleneck_inter_[l].get());

      {
        updateMovingAverage( (BatchNormLayer<Dtype>*)(bottle_bn_layers_[l].get()) );
        CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
          cudnn_handle_, CUDNN_BATCHNORM_SPATIAL, &one, &zero,
          input_desc_[l], the_input_lth[0]->gpu_data(),
          input_desc_[l], the_conv3x3_inter_l[0]->mutable_gpu_data(),
          input_scale_bias_desc_[l], bottle_scale_layers_[l]->blobs()[0]->gpu_data(), bottle_scale_layers_[l]->blobs()[1]->gpu_data(),
          1 / bottle_bn_layers_[l]->blobs()[2]->cpu_data()[0],
          bottle_bn_layers_[l]->blobs()[0]->mutable_gpu_data(),
          bottle_bn_layers_[l]->blobs()[1]->mutable_gpu_data(),
          cudnnGetBNEps(bottle_bn_layers_[l]->layer_param().batch_norm_param().eps()),
          bottleneck_bn_mean_var_[l]->mutable_gpu_data(), bottleneck_bn_mean_var_[l]->mutable_gpu_diff()
        ));
      }
      //bottle_bn_layers_[l]->Forward(the_input_lth, the_conv3x3_inter_l);
      // (in gpu) async "assemble" (original part) can start here to prepare for next conv block
      assemble_maps_gpu_origin_part(n, h, w, k0 + l * growth_rate_, growth_rate_,
        maps_diff_.mutable_gpu_data(), (const Dtype*)NULL /*output_lth_[l]->gpu_data()*/, dataCopyStream_);

      //bottle_scale_layers_[l]->Forward(the_conv3x3_inter_l, the_conv3x3_inter_l);
      bottle_relu_layers_[l]->Forward(the_conv3x3_inter_l, the_conv3x3_inter_l);

      conv1x1_layers_[l]->Forward(the_conv3x3_inter_l, the_bottleneck_inter_l);

      //bn_layers_[l]->Forward(the_bottleneck_inter_l, the_bottleneck_inter_l);
      //scale_layers_[l]->Forward(the_bottleneck_inter_l, the_bottleneck_inter_l);
      {
        caffe_copy(the_bottleneck_inter_l[0]->count(),
                   the_bottleneck_inter_l[0]->gpu_data(), 
                   bottleneck_scale_tmp_[l]->mutable_gpu_data());
        updateMovingAverage( (BatchNormLayer<Dtype>*)(bn_layers_[l].get()) );
        CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
          cudnn_handle_, CUDNN_BATCHNORM_SPATIAL, &one, &zero,
          bottleneck_inter_desc_, bottleneck_scale_tmp_[l]->gpu_data(),
          bottleneck_inter_desc_, the_bottleneck_inter_l[0]->mutable_gpu_data(),
          bottleneck_scale_bias_desc_, scale_layers_[l]->blobs()[0]->gpu_data(), scale_layers_[l]->blobs()[1]->gpu_data(),
          1 / bn_layers_[l]->blobs()[2]->cpu_data()[0],
          bn_layers_[l]->blobs()[0]->mutable_gpu_data(),
          bn_layers_[l]->blobs()[1]->mutable_gpu_data(),
          cudnnGetBNEps(bn_layers_[l]->layer_param().batch_norm_param().eps()),
          bn_mean_var_[l]->mutable_gpu_data(), bn_mean_var_[l]->mutable_gpu_diff()
        ));
      }
      relu_layers_[l]->Forward(the_bottleneck_inter_l, the_bottleneck_inter_l);

      conv3x3_layers_[l]->Forward(the_bottleneck_inter_l, the_output_lth);

    }
    else
    {
      {
        updateMovingAverage( (BatchNormLayer<Dtype>*)(bn_layers_[l].get()) );
        CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
          cudnn_handle_, CUDNN_BATCHNORM_SPATIAL, &one, &zero,
          input_desc_[l], the_input_lth[0]->gpu_data(),
          input_desc_[l], the_conv3x3_inter_l[0]->mutable_gpu_data(),
          input_scale_bias_desc_[l], scale_layers_[l]->blobs()[0]->gpu_data(), scale_layers_[l]->blobs()[1]->gpu_data(),
          1 / bn_layers_[l]->blobs()[2]->cpu_data()[0],
          bn_layers_[l]->blobs()[0]->mutable_gpu_data(),
          bn_layers_[l]->blobs()[1]->mutable_gpu_data(),
          cudnnGetBNEps(bn_layers_[l]->layer_param().batch_norm_param().eps()),
          bn_mean_var_[l]->mutable_gpu_data(), bn_mean_var_[l]->mutable_gpu_diff()
        ));
      }
      //bn_layers_[l]->Forward(the_input_lth, the_conv3x3_inter_l);

      // (in gpu) async "assemble" (original part) can start here to prepare for next conv block
      assemble_maps_gpu_origin_part(n, h, w, k0 + l * growth_rate_, growth_rate_,
        maps_diff_.mutable_gpu_data(), (const Dtype*)NULL /*output_lth_[l]->gpu_data()*/, dataCopyStream_);

      //scale_layers_[l]->Forward(the_conv3x3_inter_l, the_conv3x3_inter_l);
      relu_layers_[l]->Forward(the_conv3x3_inter_l, the_conv3x3_inter_l);

      conv3x3_layers_[l]->Forward(the_conv3x3_inter_l, the_output_lth);

    }

    if (use_dropout_)
    {
      dropout_layers_[l]->Forward(the_output_lth, the_output_lth);
    }


    // (in gpu) start async "assemble" (adding part) for this conv block
    //assemble_maps(n, h, w, k0 + l * growth_rate_, growth_rate_,
    //  maps_diff_.mutable_cpu_data(), output_lth_[l]->cpu_data());
    assemble_maps_gpu_adding_part(n, h, w, k0 + l * growth_rate_, growth_rate_,
      maps_diff_.mutable_gpu_data(), output_lth_[l]->gpu_data(), dataCopyStream_);

    // (in gpu) synchronize "assemble" here so we can start next conv block
    CUDA_CHECK(cudaStreamSynchronize(dataCopyStream_));
  }

  // maps_diff_.data() store the output data (before pre_bn_layer_), 
  // which will be used in backward of the input scale of each conv block
  // Caffe's BatchNormLayer does not touch its top data in Backward, and we will use it to 
  // infer the input of 1st conv in every conv block

  //pre_bn_layer_->Forward(vector<Blob<Dtype>*>(1, &maps_diff_), vector<Blob<Dtype>*>(1, &maps_diff_));
  //post_scale_layer_->Forward(vector<Blob<Dtype>*>(1, &maps_diff_), top);
  {
    updateMovingAverage( (BatchNormLayer<Dtype>*)(pre_bn_layer_.get()) );
    CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
      cudnn_handle_, CUDNN_BATCHNORM_SPATIAL, &one, &zero,
      final_output_desc_, maps_diff_.gpu_data(),
      final_output_desc_, top[0]->mutable_gpu_data(),
      scale_bias_desc_, post_scale_layer_->blobs()[0]->gpu_data(), post_scale_layer_->blobs()[1]->gpu_data(),
      1 / pre_bn_layer_->blobs()[2]->cpu_data()[0],
      pre_bn_layer_->blobs()[0]->mutable_gpu_data(),
      pre_bn_layer_->blobs()[1]->mutable_gpu_data(),
      cudnnGetBNEps(pre_bn_layer_->layer_param().batch_norm_param().eps()),
      pre_bn_mean_var_.mutable_gpu_data(), pre_bn_mean_var_.mutable_gpu_diff()
    ));
  }
  post_relu_layer_->Forward(top, top);

#else // USE_CUDNN

  for (int l = 0; l < num_layers_; l++)
  {
    vector<Blob<Dtype>*> the_input_lth(1, input_lth_[l].get());
    vector<Blob<Dtype>*> the_conv3x3_inter_l(1, conv3x3_inter_[l].get());
    vector<Blob<Dtype>*> the_output_lth(1, output_lth_[l].get());

    if (use_bottleneck_)
    {
      vector<Blob<Dtype>*> the_bottleneck_inter_l(1, bottleneck_inter_[l].get());

	  bottle_bn_layers_[l]->Forward(the_input_lth, the_conv3x3_inter_l);
	  // (in gpu) async "assemble" (original part) can start here to prepare for next conv block
      assemble_maps_gpu_origin_part(n, h, w, k0 + l * growth_rate_, growth_rate_,
        maps_diff_.mutable_gpu_data(), (const Dtype*)NULL /*output_lth_[l]->gpu_data()*/, dataCopyStream_);

	  bottle_scale_layers_[l]->Forward(the_conv3x3_inter_l, the_conv3x3_inter_l);
	  bottle_relu_layers_[l]->Forward(the_conv3x3_inter_l, the_conv3x3_inter_l);

      conv1x1_layers_[l]->Forward(the_conv3x3_inter_l, the_bottleneck_inter_l);

	  bn_layers_[l]->Forward(the_bottleneck_inter_l, the_bottleneck_inter_l);
	  scale_layers_[l]->Forward(the_bottleneck_inter_l, the_bottleneck_inter_l);
      relu_layers_[l]->Forward(the_bottleneck_inter_l, the_bottleneck_inter_l);

      conv3x3_layers_[l]->Forward(the_bottleneck_inter_l, the_output_lth);

    }
    else
    {
      bn_layers_[l]->Forward(the_input_lth, the_conv3x3_inter_l);

      // (in gpu) async "assemble" (original part) can start here to prepare for next conv block
      assemble_maps_gpu_origin_part(n, h, w, k0 + l * growth_rate_, growth_rate_,
        maps_diff_.mutable_gpu_data(), (const Dtype*)NULL /*output_lth_[l]->gpu_data()*/, dataCopyStream_);

	  scale_layers_[l]->Forward(the_conv3x3_inter_l, the_conv3x3_inter_l);
	  relu_layers_[l]->Forward(the_conv3x3_inter_l, the_conv3x3_inter_l);

      conv3x3_layers_[l]->Forward(the_conv3x3_inter_l, the_output_lth);

    }

    if (use_dropout_)
    {
      dropout_layers_[l]->Forward(the_output_lth, the_output_lth);
    }


    // (in gpu) start async "assemble" (adding part) for this conv block
    //assemble_maps(n, h, w, k0 + l * growth_rate_, growth_rate_,
    //  maps_diff_.mutable_cpu_data(), output_lth_[l]->cpu_data());
    assemble_maps_gpu_adding_part(n, h, w, k0 + l * growth_rate_, growth_rate_,
      maps_diff_.mutable_gpu_data(), output_lth_[l]->gpu_data(), dataCopyStream_);

    // (in gpu) synchronize "assemble" here so we can start next conv block
    CUDA_CHECK(cudaStreamSynchronize(dataCopyStream_));
  }

  // maps_diff_.data() store the output data (before post_scale_layer_, after pre_bn_layer_), 
  // which will be used in backward of the input scale of each conv block
  // Caffe's BatchNormLayer does not touch its top data in Backward, and we will use it to 
  // infer the input of 1st conv in every conv block

  pre_bn_layer_->Forward(vector<Blob<Dtype>*>(1, &maps_diff_), vector<Blob<Dtype>*>(1, &maps_diff_));
  post_scale_layer_->Forward(vector<Blob<Dtype>*>(1, &maps_diff_), top);
  post_relu_layer_->Forward(top, top);

#endif // USE_CUDNN

}

template <typename Dtype>
void DenseBlockLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
  const vector<int> shape(top[0]->shape());
  const int n = shape[0];
  const int k0= shape[1] - num_layers_ * growth_rate_;
  const int h = shape[2];
  const int w = shape[3];
  
  CHECK_EQ(k0, bottom[0]->shape()[1])
    << "Invalid top shape according to k0 + num_layers_ * growth_rate_";

#ifdef USE_CUDNN

  Dtype one(1.f), zero(0.f);

  post_relu_layer_->Backward(top, need_propagate_down_, top);
  //post_scale_layer_->Backward(top, need_propagate_down_, vector<Blob<Dtype>*>(1, &maps_diff_));
  //pre_bn_layer_->Backward(vector<Blob<Dtype>*>(1, &maps_diff_), need_propagate_down_, vector<Blob<Dtype>*>(1, &maps_diff_));
  // maps_diff still hold the data of feature maps (before BN)
  {
    CUDNN_CHECK(cudnnBatchNormalizationBackward(
      cudnn_handle_, CUDNN_BATCHNORM_SPATIAL, &one, &zero, &one, &one, 
      final_output_desc_, maps_diff_.gpu_data(), 
      final_output_desc_, top[0]->gpu_diff(), 
      final_output_desc_, maps_diff_.mutable_gpu_diff(), 
      scale_bias_desc_, post_scale_layer_->blobs()[0]->gpu_data(),
      post_scale_layer_->blobs()[0]->mutable_gpu_diff(),
      post_scale_layer_->blobs()[1]->mutable_gpu_diff(),
      cudnnGetBNEps(pre_bn_layer_->layer_param().batch_norm_param().eps()),
      pre_bn_mean_var_.gpu_data(), pre_bn_mean_var_.gpu_diff()
    ));
  }

  for (int l = num_layers_ - 1; l >= 0; l--)
  {
    vector<Blob<Dtype>*> the_input_lth(1, input_lth_[l].get());
    vector<Blob<Dtype>*> the_conv3x3_inter_l(1, conv3x3_inter_[l].get());
    vector<Blob<Dtype>*> the_output_lth(1, output_lth_[l].get());

    // diff and data, each use a individual stream
    // (in gpu) start async "disassemble" (adding part) for this conv block
    disassemble_maps_gpu_adding_part(n, h, w, k0 + l * growth_rate_, growth_rate_,
      maps_diff_.mutable_gpu_data(), output_lth_[l]->mutable_gpu_data(), dataCopyStream_);
    disassemble_maps_gpu_adding_part(n, h, w, k0 + l * growth_rate_, growth_rate_,
      maps_diff_.mutable_gpu_diff(), output_lth_[l]->mutable_gpu_diff(), diffCopyStream_);

    // (in gpu) synchronize "disassemble" (adding part) here so we can start the 
    // Backward for the conv block
    CUDA_CHECK(cudaStreamSynchronize(dataCopyStream_));
    CUDA_CHECK(cudaStreamSynchronize(diffCopyStream_));

    // (in gpu) start async "disassemble" (original part) to prepare for Backward of 
    // earlier conv in the conv block
    disassemble_maps_gpu_origin_part(n, h, w, k0 + l * growth_rate_, growth_rate_,
      maps_diff_.mutable_gpu_data(), (Dtype*)NULL /*output_lth_[l]->mutable_gpu_data()*/, dataCopyStream_);
    disassemble_maps_gpu_origin_part(n, h, w, k0 + l * growth_rate_, growth_rate_,
      maps_diff_.mutable_gpu_diff(), (Dtype*)NULL /*output_lth_[l]->mutable_gpu_diff()*/, diffCopyStream_);


    if (use_dropout_)
    {
      dropout_layers_[l]->Backward(the_output_lth, need_propagate_down_, the_output_lth);
    }

    if (use_bottleneck_)
    {
      vector<Blob<Dtype>*> the_bottleneck_inter_l(1, bottleneck_inter_[l].get());

      conv3x3_layers_[l]->Backward(the_output_lth, need_propagate_down_, the_bottleneck_inter_l);

      relu_layers_[l]->Backward(the_bottleneck_inter_l, need_propagate_down_, the_bottleneck_inter_l);
      //scale_layers_[l]->Backward(the_bottleneck_inter_l, need_propagate_down_, the_bottleneck_inter_l);
      //bn_layers_[l]->Backward(the_bottleneck_inter_l, need_propagate_down_, the_bottleneck_inter_l);
      {
        CUDNN_CHECK(cudnnBatchNormalizationBackward(
          cudnn_handle_, CUDNN_BATCHNORM_SPATIAL, &one, &zero, &one, &one, 
          bottleneck_inter_desc_, bottleneck_scale_tmp_[l]->gpu_data(), 
          bottleneck_inter_desc_, the_bottleneck_inter_l[0]->gpu_diff(), 
          bottleneck_inter_desc_, bottleneck_scale_tmp_[l]->mutable_gpu_diff(), 
          bottleneck_scale_bias_desc_, scale_layers_[l]->blobs()[0]->gpu_data(),
          scale_layers_[l]->blobs()[0]->mutable_gpu_diff(),
          scale_layers_[l]->blobs()[1]->mutable_gpu_diff(),
          cudnnGetBNEps(bn_layers_[l]->layer_param().batch_norm_param().eps()),
          bn_mean_var_[l]->gpu_data(), bn_mean_var_[l]->gpu_diff()
        ));
        caffe_copy(the_bottleneck_inter_l[0]->count(),
                   bottleneck_scale_tmp_[l]->gpu_diff(),
                   the_bottleneck_inter_l[0]->mutable_gpu_diff());
      }

      // (in gpu) synchronize "disassemble" (original part) so we can continue the preparation 
      // for Backward of conv3x3 in the conv block
      CUDA_CHECK(cudaStreamSynchronize(dataCopyStream_));
      CUDA_CHECK(cudaStreamSynchronize(diffCopyStream_));

      // re-calculate the bottom_data of conv1x1 from bottle_scale_layers_[l] and bottle_relu_layers_[l]
      // input_lth_[l] (maps_diff_) hold the data after BN, only scale is needed
      //bottle_scale_layers_[l]->Forward(the_input_lth, the_conv3x3_inter_l);
      {
        Dtype *tmp_space = pre_bn_mean_var_.mutable_gpu_diff();
        caffe_gpu_powx(
          bottleneck_bn_mean_var_[l]->count(), 
          bottleneck_bn_mean_var_[l]->gpu_diff(), 
          Dtype(-2.), tmp_space
        ); // Backward of pre_bn_layer_ is complete, space of pre_bn_mean_var_ can be reuse
        CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
          cudnn_handle_, CUDNN_BATCHNORM_SPATIAL, &one, &zero,
          input_desc_[l], the_input_lth[0]->gpu_data(),
          input_desc_[l], the_conv3x3_inter_l[0]->mutable_gpu_data(),
          input_scale_bias_desc_[l], bottle_scale_layers_[l]->blobs()[0]->gpu_data(), bottle_scale_layers_[l]->blobs()[1]->gpu_data(),
          bottleneck_bn_mean_var_[l]->gpu_data(), tmp_space,
          cudnnGetBNEps(bottle_bn_layers_[l]->layer_param().batch_norm_param().eps())
        ));
      }
      bottle_relu_layers_[l]->Forward(the_conv3x3_inter_l, the_conv3x3_inter_l);

      conv1x1_layers_[l]->Backward(the_bottleneck_inter_l, need_propagate_down_, the_conv3x3_inter_l);

      bottle_relu_layers_[l]->Backward(the_conv3x3_inter_l, need_propagate_down_, the_conv3x3_inter_l);
      //bottle_scale_layers_[l]->Backward(the_conv3x3_inter_l, need_propagate_down_, the_conv3x3_inter_l);
      //bottle_bn_layers_[l]->Backward(the_conv3x3_inter_l, need_propagate_down_, the_input_lth);
      {
        CUDNN_CHECK(cudnnBatchNormalizationBackward(
          cudnn_handle_, CUDNN_BATCHNORM_SPATIAL, &one, &zero, &one, &one, 
          input_desc_[l], the_input_lth[0]->gpu_data(), 
          input_desc_[l], the_conv3x3_inter_l[0]->gpu_diff(), 
          input_desc_[l], the_input_lth[0]->mutable_gpu_diff(), 
          input_scale_bias_desc_[l], bottle_scale_layers_[l]->blobs()[0]->gpu_data(),
          bottle_scale_layers_[l]->blobs()[0]->mutable_gpu_diff(),
          bottle_scale_layers_[l]->blobs()[1]->mutable_gpu_diff(),
          cudnnGetBNEps(bottle_bn_layers_[l]->layer_param().batch_norm_param().eps()),
          bottleneck_bn_mean_var_[l]->gpu_data(), bottleneck_bn_mean_var_[l]->gpu_diff()
        ));
      }
    }
    else
    {
      // (in gpu) synchronize "disassemble" (original part) so we can continue the preparation 
      // for Backward of conv3x3 in the conv block
      CUDA_CHECK(cudaStreamSynchronize(dataCopyStream_));
      CUDA_CHECK(cudaStreamSynchronize(diffCopyStream_));

      // re-calculate the bottom_data of conv3x3 from scale_layers_[l] and relu_layers_[l]

      //scale_layers_[l]->Forward(the_input_lth, the_conv3x3_inter_l);
      {
        Dtype *tmp_space = pre_bn_mean_var_.mutable_gpu_diff();
        caffe_gpu_powx(
          bn_mean_var_[l]->count(), 
          bn_mean_var_[l]->gpu_diff(), 
          Dtype(-2.), tmp_space
        ); // Backward of pre_bn_layer_ is complete, space of pre_bn_mean_var_ can be reuse
        CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
          cudnn_handle_, CUDNN_BATCHNORM_SPATIAL, &one, &zero,
          input_desc_[l], the_input_lth[0]->gpu_data(),
          input_desc_[l], the_conv3x3_inter_l[0]->mutable_gpu_data(),
          input_scale_bias_desc_[l], scale_layers_[l]->blobs()[0]->gpu_data(), scale_layers_[l]->blobs()[1]->gpu_data(),
          bn_mean_var_[l]->gpu_data(), tmp_space,
          cudnnGetBNEps(bn_layers_[l]->layer_param().batch_norm_param().eps())
        ));
      }
      relu_layers_[l]->Forward(the_conv3x3_inter_l, the_conv3x3_inter_l);

      conv3x3_layers_[l]->Backward(the_output_lth, need_propagate_down_, the_conv3x3_inter_l);

      relu_layers_[l]->Backward(the_conv3x3_inter_l, need_propagate_down_, the_conv3x3_inter_l);
      //scale_layers_[l]->Backward(the_conv3x3_inter_l, need_propagate_down_, the_conv3x3_inter_l);
      //bn_layers_[l]->Backward(the_conv3x3_inter_l, need_propagate_down_, the_input_lth);
      {
        CUDNN_CHECK(cudnnBatchNormalizationBackward(
          cudnn_handle_, CUDNN_BATCHNORM_SPATIAL, &one, &zero, &one, &one, 
          input_desc_[l], the_input_lth[0]->gpu_data(), 
          input_desc_[l], the_conv3x3_inter_l[0]->gpu_diff(), 
          input_desc_[l], the_input_lth[0]->mutable_gpu_diff(), 
          input_scale_bias_desc_[l], scale_layers_[l]->blobs()[0]->gpu_data(),
          scale_layers_[l]->blobs()[0]->mutable_gpu_diff(),
          scale_layers_[l]->blobs()[1]->mutable_gpu_diff(),
          cudnnGetBNEps(bn_layers_[l]->layer_param().batch_norm_param().eps()),
          bn_mean_var_[l]->gpu_data(), bn_mean_var_[l]->gpu_diff()
        ));
      }
    }

    { // add the diff together before continue
      const int count = input_lth_[l]->count();
      Dtype* target_ptr;
      const Dtype* adding_in_ptr;
      //if (l > 0) // in the original structure we will always copy the diff to bottom.diff
      {
        target_ptr = maps_diff_.mutable_gpu_diff();
        adding_in_ptr = tmp_diff_.gpu_diff(); // diff of input_lth_[l]
      }
      //else
      //{
      //  // for the first conv block, store the sum of diff in tmp_diff_.diff (input_lth_[0].diff)
      //  // because pre_bn_layer_ treat input_lth_[0] as the top blob.
      //  target_ptr = tmp_diff_.mutable_gpu_diff(); // diff of input_lth_[l]
      //  adding_in_ptr = maps_diff_.gpu_diff();
      //}

      // in gpu caffe_gpu_axpy is used
      //caffe_axpy(count, Dtype(1.), adding_in_ptr, target_ptr);
      caffe_gpu_axpy(count, Dtype(1.), adding_in_ptr, target_ptr);
    }
  }

#else // USE_CUDNN

  post_relu_layer_->Backward(top, need_propagate_down_, top);
  post_scale_layer_->Backward(top, need_propagate_down_, vector<Blob<Dtype>*>(1, &maps_diff_));
  pre_bn_layer_->Backward(vector<Blob<Dtype>*>(1, &maps_diff_), need_propagate_down_, vector<Blob<Dtype>*>(1, &maps_diff_));
  // maps_diff still hold the data of feature maps (after BN, before Scale)

  for (int l = num_layers_ - 1; l >= 0 ; l --)
  {
    vector<Blob<Dtype>*> the_input_lth(1, input_lth_[l].get());
    vector<Blob<Dtype>*> the_conv3x3_inter_l(1, conv3x3_inter_[l].get());
    vector<Blob<Dtype>*> the_output_lth(1, output_lth_[l].get());
    
    // diff and data, each use a individual stream
    // (in gpu) start async "disassemble" (adding part) for this conv block
    disassemble_maps_gpu_adding_part(n, h, w, k0 + l * growth_rate_, growth_rate_, 
                     maps_diff_.mutable_gpu_data(), output_lth_[l]->mutable_gpu_data(), dataCopyStream_);
    disassemble_maps_gpu_adding_part(n, h, w, k0 + l * growth_rate_, growth_rate_, 
                     maps_diff_.mutable_gpu_diff(), output_lth_[l]->mutable_gpu_diff(), diffCopyStream_);
    
    // (in gpu) synchronize "disassemble" (adding part) here so we can start the 
    // Backward for the conv block
    CUDA_CHECK(cudaStreamSynchronize(dataCopyStream_));
    CUDA_CHECK(cudaStreamSynchronize(diffCopyStream_));

    // (in gpu) start async "disassemble" (original part) to prepare for Backward of 
    // earlier conv in the conv block
    disassemble_maps_gpu_origin_part(n, h, w, k0 + l * growth_rate_, growth_rate_,
      maps_diff_.mutable_gpu_data(), (Dtype*)NULL /*output_lth_[l]->mutable_gpu_data()*/, dataCopyStream_);
    disassemble_maps_gpu_origin_part(n, h, w, k0 + l * growth_rate_, growth_rate_,
      maps_diff_.mutable_gpu_diff(), (Dtype*)NULL /*output_lth_[l]->mutable_gpu_diff()*/, diffCopyStream_);
    
    
    if (use_dropout_)
    {
      dropout_layers_[l]->Backward(the_output_lth, need_propagate_down_, the_output_lth);
    }
    
    if (use_bottleneck_)
    {
      vector<Blob<Dtype>*> the_bottleneck_inter_l(1, bottleneck_inter_[l].get());
      
      conv3x3_layers_[l]->Backward(the_output_lth, need_propagate_down_, the_bottleneck_inter_l);
      
      relu_layers_[l]->Backward(the_bottleneck_inter_l, need_propagate_down_, the_bottleneck_inter_l);
      scale_layers_[l]->Backward(the_bottleneck_inter_l, need_propagate_down_, the_bottleneck_inter_l);
      bn_layers_[l]->Backward(the_bottleneck_inter_l, need_propagate_down_, the_bottleneck_inter_l);

      // (in gpu) synchronize "disassemble" (original part) so we can continue the preparation 
      // for Backward of conv3x3 in the conv block
      CUDA_CHECK(cudaStreamSynchronize(dataCopyStream_));
      CUDA_CHECK(cudaStreamSynchronize(diffCopyStream_));
      
      // re-calculate the bottom_data of conv1x1 from bottle_scale_layers_[l] and bottle_relu_layers_[l]
      // input_lth_[l] (maps_diff_) hold the data after BN, only scale is needed
      bottle_scale_layers_[l]->Forward(the_input_lth, the_conv3x3_inter_l);
      bottle_relu_layers_[l]->Forward(the_conv3x3_inter_l, the_conv3x3_inter_l);
      
      conv1x1_layers_[l]->Backward(the_bottleneck_inter_l, need_propagate_down_, the_conv3x3_inter_l);
      
      bottle_relu_layers_[l]->Backward(the_conv3x3_inter_l, need_propagate_down_, the_conv3x3_inter_l);
      bottle_scale_layers_[l]->Backward(the_conv3x3_inter_l, need_propagate_down_, the_conv3x3_inter_l);
      bottle_bn_layers_[l]->Backward(the_conv3x3_inter_l, need_propagate_down_, the_input_lth);
    }
    else
    {
      // (in gpu) synchronize "disassemble" (original part) so we can continue the preparation 
      // for Backward of conv3x3 in the conv block
      CUDA_CHECK(cudaStreamSynchronize(dataCopyStream_));
      CUDA_CHECK(cudaStreamSynchronize(diffCopyStream_));
      
      // re-calculate the bottom_data of conv3x3 from scale_layers_[l] and relu_layers_[l]

      scale_layers_[l]->Forward(the_input_lth, the_conv3x3_inter_l);
      relu_layers_[l]->Forward(the_conv3x3_inter_l, the_conv3x3_inter_l);
      
      conv3x3_layers_[l]->Backward(the_output_lth, need_propagate_down_, the_conv3x3_inter_l);
      
      relu_layers_[l]->Backward(the_conv3x3_inter_l, need_propagate_down_, the_conv3x3_inter_l);
      scale_layers_[l]->Backward(the_conv3x3_inter_l, need_propagate_down_, the_conv3x3_inter_l);
      bn_layers_[l]->Backward(the_conv3x3_inter_l, need_propagate_down_, the_input_lth);
    }
    
    { // add the diff together before continue
      const int count = input_lth_[l]->count();
      Dtype* target_ptr;
      const Dtype* adding_in_ptr;
      //if (l > 0) // in the original structure we will always copy the diff to bottom.diff
      {
        target_ptr = maps_diff_.mutable_gpu_diff(); 
        adding_in_ptr = tmp_diff_.gpu_diff(); // diff of input_lth_[l]
      }
      //else
      //{
      //  // for the first conv block, store the sum of diff in tmp_diff_.diff (input_lth_[0].diff)
      //  // because pre_bn_layer_ treat input_lth_[0] as the top blob.
      //  target_ptr = tmp_diff_.mutable_gpu_diff(); // diff of input_lth_[l]
      //  adding_in_ptr = maps_diff_.gpu_diff();
      //}
      
      // in gpu caffe_gpu_axpy is used
      //caffe_axpy(count, Dtype(1.), adding_in_ptr, target_ptr);
      caffe_gpu_axpy(count, Dtype(1.), adding_in_ptr, target_ptr);
    }
  }

#endif // USE_CUDNN
  
  caffe_copy(bottom[0]->count(), maps_diff_.gpu_diff(), bottom[0]->mutable_gpu_diff());
}

INSTANTIATE_LAYER_GPU_FUNCS(DenseBlockLayer);

} // namespace caffe
