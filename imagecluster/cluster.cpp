/************************************************************************************
***
*** File Author: Dell,  2018-11-12 01:50:34
***
************************************************************************************/
// https://pytorch.org/cppdocs/notes/tensor_basics.html

#include "cluster.h"


#define MAX_HISTOGRAM_BINS 65536
#define RGB565_R(x16) ((((x16) >> 11) & 0x1f) << 3)
#define RGB565_G(x16) ((((x16) >> 5) & 0x3f) << 2)
#define RGB565_B(x16) (((x16) & 0x1f) << 3)
#define RGB565_NO(r, g, b) (((r)&0xf8) << 8 | ((g) & 0xfc) << 3 | ((b) & 0xf8) >> 3)

// r, g, b, w
#define R_COL 0
#define G_COL 1
#define B_COL 2
#define W_COL 3


void image_cluster(const at::Tensor &hist, int K, int maxloops, at::Tensor &index, at::Tensor &center)
{
    const int N = hist.size(0);

    if (N != MAX_HISTOGRAM_BINS) {
        std::cout << "Error: histogram is NOT for image cluster." << std::endl;
        exit(-1);
    }

    hist.contiguous();
    index.contiguous();
    center.contiguous();


    at::Tensor sample = at::zeros({N, 4}, at::kFloat);
    sample.contiguous();

    auto hist_a = hist.accessor<float, 1>();
    auto sample_a = sample.accessor<float, 2>();

    // Create samples
    for (int i = 0; i < N; i++) {
        sample_a[i][R_COL] = RGB565_R(i);
        sample_a[i][G_COL] = RGB565_G(i);
        sample_a[i][B_COL] = RGB565_B(i);

        sample_a[i][W_COL] = hist_a[i];
    }

    wkmeans_cluster(sample, K, maxloops, center);

    // Class RGB565 samples (Total max is 65536)
    auto cindex = wkmeans_class(sample, center);
    cindex.contiguous();
    auto cindex_a = cindex.accessor<int32_t, 1>();
    auto index_a = index.accessor<int32_t, 1>();

    for (int i = 0; i < N; i++) {
        index_a[i] = cindex_a[i];
    }
}

void image_cluster_forward(const at::Tensor& input, const at::Tensor &index, const at::Tensor& center, at::Tensor& output, at::Tensor& label)
{
    int c;
    float r, g, b;

    const int B = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);

    if (C != 3) {
        std::cout << "Error: Input is not RGB image tensor." << std::endl;
        exit(-1);
    }

    input.contiguous();
    output.contiguous();
    label.contiguous();

    auto input_a = input.accessor<float, 4>();
    auto index_a = index.accessor<int32_t, 1>();
    auto center_a = center.accessor<float, 2>();
    auto label_a = label.accessor<int32_t, 4>();
    auto output_a = output.accessor<float, 4>();

    for (int bat = 0; bat < B; bat++) {
        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                r = input_a[bat][0][i][j] * 255.0;
                g = input_a[bat][1][i][j] * 255.0;
                b = input_a[bat][2][i][j] * 255.0;

                c = RGB565_NO(int(r), int(g), int(b));
                c = index_a[c];

                label_a[bat][0][i][j] = c;

                output_a[bat][0][i][j] = center_a[c][R_COL];
                output_a[bat][1][i][j] = center_a[c][G_COL];
                output_a[bat][2][i][j] = center_a[c][B_COL];
            }
        }
    }

    output.div_(255.0);
}

int image_cluster_backward(at::Tensor grad_output, at::Tensor input, at::Tensor grad_input)
{
    grad_input.resize_as_(grad_output);
    grad_input.fill_(1);

    return 1;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cluster", &image_cluster, "Image cluster");
    m.def("forward", &image_cluster_forward, "Image cluster forward");
    m.def("backward", &image_cluster_backward, "Image cluster backward");
}

