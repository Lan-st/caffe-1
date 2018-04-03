#ifndef CAFFE_NOISE_LAYER_HPP_
#define CAFFE_NOISE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {
/**
 * Add noise to the input blob element-wisely, gaussian and uniform noise supported for now
 */
template <typename Dtype>
class NoiseLayer : public NeuronLayer<Dtype>
{
 public:
  explicit NoiseLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}
      
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "Noise"; }
  
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
      
  NoiseParameter::NoiseType type_;
  Dtype ratio_upper_, ratio_lower_, width_lower_, width_upper_;
  FillerParameter filler_param_;
  Blob<Dtype> noise_;
  shared_ptr<Filler<Dtype> > filler_;
};

} // namespace caffe

#endif // CAFFE_NOISE_LAYER_HPP_