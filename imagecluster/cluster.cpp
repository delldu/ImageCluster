/************************************************************************************
***
*** File Author: Dell,  2018-11-12 01:50:34
***
************************************************************************************/
// https://pytorch.org/cppdocs/notes/tensor_basics.html

#include "cluster.h"

#define CheckPoint(fmt, arg...) printf("# CheckPoint: %d(%s): " fmt "\n", (int)__LINE__, __FILE__, ##arg)

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

// 
#define MIN(a, b)  ((a) > (b)? (b) : (a))
#define MAX(a, b)  ((a) > (b)? (a) : (b))
#define MAX_STACK_ELEMENTS (8 * 1024 * 1024)

struct {
    int row[MAX_STACK_ELEMENTS], col[MAX_STACK_ELEMENTS], count;
} temp_stack;

void statck_reset()
{
    temp_stack.count = 0;
}

void stack_push(int row, int col)
{
    if (temp_stack.count < MAX_STACK_ELEMENTS) {
        temp_stack.row[temp_stack.count] = row;
        temp_stack.col[temp_stack.count] = col;
        temp_stack.count++;
    }
}

void stack_pop(int *row, int *col)
{
    *row = temp_stack.row[temp_stack.count - 1];
    *col = temp_stack.col[temp_stack.count - 1];
    temp_stack.count--;
}


int stack_empty()
{
    return temp_stack.count < 1;
}


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


void image_cluster_segment(const at::Tensor& label, int radius, at::Tensor& output_mask)
{
    int i, j, i2, j2, c, row, col, instance;

    const int B = label.size(0);
    const int C = label.size(1);
    const int H = label.size(2);
    const int W = label.size(3);

    if (C != 1) {
        std::cout << "Error: label is not Bx1xHxW tensor." << std::endl;
        exit(-1);
    }

    label.contiguous();
    output_mask.contiguous();

    auto label_a = label.accessor<int32_t, 4>();
    auto output_mask_a = output_mask.accessor<int32_t, 4>();

    for (int bat = 0; bat < B; bat++) {
        instance = 0;
        for (i = 0; i < H; i++) {
            for (j = 0; j < W; j++) {
                c =  label_a[bat][0][i][j];
                if (c < 0)
                    continue;

                // New tracking from here (row == i, col == j, label == c)
                instance++;
                statck_reset();

                stack_push(i, j);
                while(! stack_empty()) {
                    stack_pop(&row, &col);
                    // Save row, col to output_mask
                    output_mask_a[bat][0][row][col] = instance;
                    label_a[bat][0][row][col] = -1;
                    // Search neighbours around(row, col) ...
                    for (i2 = MAX(0, row - radius); i2 < MIN(H, row + radius); i2++) {
                        for (j2 = MAX(0, col - radius); j2 < MIN(W, col + radius); j2++) {
                            if (label_a[bat][0][i2][j2] == c)
                                stack_push(i2, j2);
                        }
                    }
                }
                // Finish tracking ! 
            }
        }
    }
}

void image_cluster_adjmatrix(const at::Tensor& input_mask, at::Tensor& output_matrix)
{
    const int B = input_mask.size(0);
    const int C = input_mask.size(1);
    const int H = input_mask.size(2);
    const int W = input_mask.size(3);

    if (C != 1) {
        std::cout << "Error: input mask is not Bx1xHxW tensor." << std::endl;
        exit(-1);
    }


    input_mask.contiguous();
    output_matrix.contiguous();

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cluster", &image_cluster, "Image cluster");
    m.def("forward", &image_cluster_forward, "Image cluster forward");
    m.def("backward", &image_cluster_backward, "Image cluster backward");
    m.def("segment", &image_cluster_segment, "Image segment");
    m.def("adjmatrix", &image_cluster_adjmatrix, "Image adjmatrix");
}

