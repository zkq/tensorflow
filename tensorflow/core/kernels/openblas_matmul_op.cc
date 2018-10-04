
#if defined(OPENBLAS)


#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/fill_functor.h"

#include "cblas.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename T, bool USE_CUBLAS>
class OpenBlasMatMulOp : public OpKernel {
public:
    explicit OpenBlasMatMulOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_a", &transpose_a_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_b", &transpose_b_));
    }

    void Compute(OpKernelContext *ctx) override {
        const Tensor &a = ctx->input(0);
        const Tensor &b = ctx->input(1);

        // Check that the dimensions of the two matrices are valid.
        OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(a.shape()),
                    errors::InvalidArgument("In[0] is not a matrix"));
        OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(b.shape()),
                    errors::InvalidArgument("In[1] is not a matrix"));
        Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
        dim_pair[0].first = transpose_a_ ? 0 : 1;
        dim_pair[0].second = transpose_b_ ? 1 : 0;

        OP_REQUIRES(
            ctx, a.dim_size(dim_pair[0].first) == b.dim_size(dim_pair[0].second),
            errors::InvalidArgument(
                "Matrix size-incompatible: In[0]: ", a.shape().DebugString(),
                ", In[1]: ", b.shape().DebugString()));
        int a_dim_remaining = 1 - dim_pair[0].first;
        int b_dim_remaining = 1 - dim_pair[0].second;
        TensorShape out_shape(
            {a.dim_size(a_dim_remaining), b.dim_size(b_dim_remaining)});
        Tensor* out = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));

        if (out->NumElements() == 0) {
            // If a has shape [0, x] or b has shape [x, 0], the output shape
            // is a 0-element matrix, so there is nothing to do.
            return;
        }

        if (a.NumElements() == 0 || b.NumElements() == 0) {
            // If a has shape [x, 0] and b has shape [0, y], the
            // output shape is [x, y] where x and y are non-zero, so we fill
            // the output with zeros.
            functor::SetZeroFunctor<Device, T> f;
            f(ctx->eigen_device<Device>(), out->flat<T>());
            return;
        }

        const int m = a.dim_size(1 - dim_pair[0].first);
        const int k = a.dim_size(dim_pair[0].first);
        const int n = b.dim_size(1 - dim_pair[0].second);
        bool transpose_a = dim_pair[0].first == 0;
        bool transpose_b = dim_pair[0].second == 1;

        auto a_ptr = (a.template flat<T>().data());
        auto b_ptr = (b.template flat<T>().data());
        auto c_ptr = (out->template flat<T>().data());

        OpenBlasGemm(transpose_a, transpose_b, m, n, k, a_ptr, transpose_a ? m : k,
                    b_ptr, transpose_b ? k : n, c_ptr, n);
        
    }

private:
    bool transpose_a_;
    bool transpose_b_;

    // FP32 GEMM
    void OpenBlasGemm(bool transa, bool transb, const int m, const int n,
                    const int k, const float* a, const int lda, const float* b,
                    const int ldb, float* c, const int ldc) {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        printf("using openblas sgemm\n");
        cblas_sgemm(CblasRowMajor, transa ? CblasTrans : CblasNoTrans,
                    transb ? CblasTrans : CblasNoTrans, m, n, k, alpha, const_cast<float *>(a), lda, const_cast<float *>(b),
                    ldb, beta, c, ldc);
    }
    
    // FP64 GEMM
    void OpenBlasGemm(bool transa, bool transb, const int m, const int n,
                    const int k, const double* a, const int lda, const double* b,
                    const int ldb, double* c, const int ldc) {
        const double alpha = 1.0;
        const double beta = 0.0;
        printf("using openblas dgemm\n");
        cblas_dgemm(CblasRowMajor, transa ? CblasTrans : CblasNoTrans,
                    transb ? CblasTrans : CblasNoTrans, m, n, k, alpha, const_cast<double*>(a), lda, const_cast<double*>(b),
                    ldb, beta, c, ldc);
    }
};

#define REGISTER_CPU(T)                                         \
    REGISTER_KERNEL_BUILDER(                                      \
        Name("MatMul").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
        OpenBlasMatMulOp<CPUDevice, T, false /* cublas, ignored for CPU */>);

#ifdef ENABLE_OPENBLAS
TF_CALL_float(REGISTER_CPU);
TF_CALL_double(REGISTER_CPU);
#endif  // ENABLE_OPENBLAS

}  // namespace tensorflow

#endif  // OPENBLAS
