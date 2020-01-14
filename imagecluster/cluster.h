/************************************************************************************
***
*** File Author: Dell, , 2018-11-12 20:14:49
***
************************************************************************************/

#include <vector>
#include <torch/extension.h>

void wkmeans_cluster(const at::Tensor &sample, int K, int maxloops, at::Tensor &center);
at::Tensor wkmeans_class(const at::Tensor &input, const at::Tensor &center);
// at::Tensor wkmeans_render(const at::Tensor &input, const at::Tensor &center);

void image_cluster(const at::Tensor &hist, int K, int maxloops, at::Tensor &index, at::Tensor &center);
void image_cluster_forward(const at::Tensor& input, const at::Tensor& index, const at::Tensor& center, at::Tensor& output, at::Tensor& label);
int image_cluster_backward(at::Tensor grad_output, at::Tensor input, at::Tensor grad_input);
void image_label_segment(const at::Tensor& label, int radius, at::Tensor& mask);
void image_color_mask(const at::Tensor& mask, const at::Tensor& colors, at::Tensor& output);
void image_mask_adjmatrix(const at::Tensor& input_mask, int radius, at::Tensor& output_matrix);
