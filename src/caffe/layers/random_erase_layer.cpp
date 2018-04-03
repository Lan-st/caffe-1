#include "caffe/layers/noise_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void NoiseLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  filler_param_ = this-> layer_param_.random_erase_param().filler();
  ratio_upper_ = this-> layer_param_.random_erase_param().ratio_upper();
  ratio_lower_ = this-> layer_param_.random_erase_param().ratio_lower();
  width_lower_ = this-> layer_param_.random_erase_param().width_lower();
  width_upper_ = this-> layer_param_.random_erase_param().width_upper();
  filler_.reset(GetFiller<Dtype>(filler_param_));

}

template <typename Dtype>
void NoiseLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
  noise_.Reshape(bottom[0]->shape());
}

template <typename Dtype>
void NoiseLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top)
{
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  if (this->phase_ == TRAIN)
  {
    filler_-> Fill(noise_);
    Dtype* noise_data = noise_.mutable_cpu_data();
    caffe_rng_gaussian<Dtype>(count, mean_, stdvar_, noise_data);
    caffe_add<Dtype>(count, bottom_data, noise_data, top_data);
  }
  else
  {
    caffe_copy(count, bottom_data, top_data);
  }
}

template <typename Dtype>
void NoiseLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom)
{
  if (propagate_down[0])
  {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_copy(top[0]->count(), top_diff, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(NoiseLayer);
#endif

INSTANTIATE_CLASS(NoiseLayer);
REGISTER_LAYER_CLASS(Noise);

} // namespace caffe
