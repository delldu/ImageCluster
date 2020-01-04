/************************************************************************************
***
*** File Author: Dell, 2018-11-12 20:14:49
***
************************************************************************************/
// https://pytorch.org/cppdocs/notes/tensor_basics.html

#include "cluster.h"


/************************************************************************************
*  Weight K-Means data format:
*
*  Sample: [N, (f1, f2, ..., fn, w)]
*  Center: [K, (f1, f2, ..., fn, w)] 
*
*  Input: [N, (f1, f2, ..., fn)]
*  Output: [N, (f1, f2, ..., fn)]
* 
************************************************************************************/

void update_center(const at::Tensor &sample, const at::Tensor &cindex, at::Tensor &center)
{
    const int N = sample.size(0);
    const int K = center.size(0);
    const int n_feats = center.size(1) - 1;

    auto cindex_a = cindex.accessor<int32_t, 1>();
    auto sample_a = sample.accessor<float, 2>();
    auto center_a = center.accessor<float, 2>();

    center.zero_();

    for (int i = 0; i < N; i++) {
        int c = cindex_a[i];
        float w = sample_a[i][n_feats];

        for (int j = 0; j < n_feats; j++) {
            center_a[c][j] += w*sample_a[i][j];
        }   
        center_a[c][n_feats] += w;      // sum w
    }

    for (int i = 0; i < K; i++) {
        float w = center_a[i][n_feats];

        if (w > 0) {
            for (int j = 0; j < n_feats; j++)
                center_a[i][j] /= w;
       }
    }
}

void update_distance(const at::Tensor &input, const at::Tensor &center, at::Tensor &dmatrix)
{
    const int N = input.size(0);
    const int n_feats = center.size(1) - 1;
    const int K = center.size(0);

    auto input_a = input.accessor<float, 2>();
    auto center_a = center.accessor<float, 2>();
    auto dmatrix_a = dmatrix.accessor<float, 2>();

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            float sum = 0;
            for (int k = 0; k < n_feats; k++) {
                float d = input_a[i][k] - center_a[j][k];
                sum += d * d;
            }
            dmatrix_a[i][j] = sum;
        }
    }
}

int update_class(const at::Tensor &dmatrix, at::Tensor &cindex)
{
    int flag = 0;
    const int N = dmatrix.size(0);
    const int K = dmatrix.size(1);

    auto dmatrix_a = dmatrix.accessor<float, 2>();
    auto cindex_a = cindex.accessor<int32_t, 1>();

    for (int i = 0; i < N; i++) {
        int index = 0;
        float d = dmatrix_a[i][0];
        for (int k = 1; k < K; k++) {
            if (dmatrix_a[i][k] < d) {
                index = k;
                d = dmatrix_a[i][k];
            }
        }
        if (cindex_a[i] != index) {
            flag = 1;
            cindex_a[i] = index;
        }
    }

    return flag;
}

void sort_center(at::Tensor &center)
{
    const int n_feats = center.size(1) - 1;

    // narrow(input, dimension, start, length) -> Tensor
    auto weight = center.narrow(/*dim=*/1, n_feats, 1);

    // sort(input, dim=None, descending=False, out=None) -> (Tensor, LongTensor)
    auto index = std::get<1>(weight.sort(0, 1)); // sort by row, descending
    index =  index.view(-1);

    auto s = at::index_select(center, 0, index);
    s.contiguous();

    // Copy s to center
    auto s_a = s.accessor<float, 2>();
    auto center_a = center.accessor<float, 2>();
    for (int i = 0; i < center.size(0); i++) {
        for (int j = 0; j < center.size(1); j++) {
            center_a[i][j] = s_a[i][j];
        }
    }
}

/************************************************************************************
*
* wkmeans_cluster(sample, K, 128, center)
*
************************************************************************************/
void wkmeans_cluster(const at::Tensor &sample, int K, int maxloops, at::Tensor &center)
{
    int flag = 0;
    int epoch = 0;
   
    const int N = sample.size(0);

    // distance matrix
    auto dmatrix = at::zeros({N, K}, at::kFloat);

    // cluster index for class
    auto cindex = at::zeros({N}, at::kInt);

    sample.contiguous();
    dmatrix.contiguous();

    // Init cindex
    int stride = (N + K - 1)/K;
    auto cindex_a = cindex.accessor<int32_t, 1>();
    for (int i = 0; i < N; i++) {
        cindex_a[i] = i/stride;
    }

    do {
        update_center(sample, cindex, center);
       
        update_distance(sample, center, dmatrix);

        flag = update_class(dmatrix, cindex);

        ++epoch;

    } while(epoch < maxloops && flag);


    // reverse sort center
    sort_center(center); 
    // std::cout << "Real Cluster Loop " << epoch << " Times !!!" << std::endl;
}


/************************************************************************************
*
* index = wkmeans_class(input, center)
*
************************************************************************************/
at::Tensor wkmeans_class(const at::Tensor &input, const at::Tensor &center)
{
    const int N = input.size(0);
    const int K = center.size(0);

    // distance matrix
    auto dmatrix = at::zeros({N, K}, at::kFloat);

    // cluster index for class
    auto cindex = at::zeros({N}, at::kInt);
    dmatrix.contiguous();
    cindex.contiguous();

    update_distance(input, center, dmatrix);
    update_class(dmatrix, cindex);

    return cindex;
}

/************************************************************************************
*
* output = wkmeans_render(input, center)
*
************************************************************************************/
at::Tensor wkmeans_render(const at::Tensor &input, const at::Tensor &center)
{
    const int N = input.size(0);
    const int n_feats = center.size(1) - 1;

    auto output = at::zeros({N, n_feats}, at::kFloat);
    output.contiguous();

    // cluster index for class
    auto cindex = wkmeans_class(input, center);
    cindex.contiguous();

    auto output_a = output.accessor<float, 2>();
    auto cindex_a = cindex.accessor<int32_t, 1>();
    auto center_a = center.accessor<float, 2>();

    for (int i = 0; i < N; i++) {
        int index = cindex_a[i];

        for (int j = 0; j < n_feats; j++) {
            output_a[i][j] = center_a[index][j];
        }
    }

    return output;
}
