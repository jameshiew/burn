use burn_tensor::ops::ConvOptions;
use cubecl::linalg::matmul::{
    components::{
        global::{AccumulatorLoader, Unloader},
        stage, MatmulProblem, MatrixLayout,
    },
    kernels::{matmul::AdvancedConfig, MatmulAvailabilityError},
};
use cubecl::prelude::*;

use super::Config;

#[cube]
pub trait Convolution<EG: Numeric, ES: Numeric, Acc: Numeric, SMM: stage::Matmul<ES, EG, Acc>>:
    'static + Send + Sync + ConvolutionKernel<EG, EG, Config: Config>
{
    type LhsLoader: CubeType;
    type RhsLoader: CubeType;
    type AccumulatorLoader: AccumulatorLoader<EG, Acc, SMM::Config>;

    type Out: Unloader<EG>;
    type Accumulator: CubeType;

    /// Performs the convolution over data loaded by the
    /// LHS and RHS loaders, over the range given for K, and stores with
    /// using the output unloader.
    ///
    /// To compute the whole range of k values, use k_range=(0, K) where
    /// K is the K dimension of LHS and RHS.
    fn execute(
        lhs_loader: Self::LhsLoader,
        rhs_loader: Self::RhsLoader,
        acc_loader: Self::AccumulatorLoader,
        unloader: Self::Out,
        acc: &mut Self::Accumulator,
        k_range: (u32, u32),
        #[comptime] config: Self::Config,
    );

    fn init_lhs_loader(
        lhs: &Tensor<Line<EG>>,
        x_offset: u32,
        y_offset: u32,
        #[comptime] config: Self::Config,
    ) -> Self::LhsLoader;

    fn init_rhs_loader(
        rhs: &Tensor<Line<EG>>,
        x_offset: u32,
        y_offset: u32,
        #[comptime] config: Self::Config,
    ) -> Self::RhsLoader;

    fn init_bias_loader(
        rhs: &Tensor<Line<EG>>,
        n_offset: u32,
        #[comptime] config: Self::Config,
        #[comptime] has_bias: bool,
    ) -> Self::AccumulatorLoader;

    fn init_unloader(out: &mut Tensor<Line<EG>>, x_offset: u32, y_offset: u32) -> Self::Out;

    fn init_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator;
}

/// Provides configuration for a matmul kernel at any level
pub trait ConvolutionKernel<I: Numeric, O: Numeric> {
    /// Configuration tailored to the matmul implementation
    type Config: Config;

    /// Asserts that the configuration for this matmul will lead to a valid computation
    fn check_config(config: Self::Config);

    /// Checks if the client can handle the features used in this computation
    fn check_availability<R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
    ) -> Result<(), MatmulAvailabilityError>;

    fn make_config(
        problem: &ConvolutionProblem,
        cube_dim: &CubeDim,
        cube_count: &CubeCount,
        advanced_config: &AdvancedConfig,
    ) -> Self::Config;
}

/// Provides launch entry point to solve a matmul
pub trait ConvolutionLaunch<I: Numeric, O: Numeric>: ConvolutionKernel<I, O> {
    /// Entry point
    ///
    /// # Safety
    ///
    /// Out-of-bounds can happen
    #[allow(clippy::too_many_arguments)]
    unsafe fn launch_unchecked<R: Runtime>(
        client: &ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel>,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        input: TensorArg<'_, R>,
        weight: TensorArg<'_, R>,
        bias: TensorArg<'_, R>,
        out: TensorArg<'_, R>,
        config: <Self as ConvolutionKernel<I, O>>::Config,
    );
}

#[derive(Clone)]
/// Description of a matmul problem to solve, regardless of actual data
pub struct ConvolutionProblem {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub lhs_layout: MatrixLayout,
    pub rhs_layout: MatrixLayout,
    pub lhs_line_size: u8,
    pub rhs_line_size: u8,
    pub out_line_size: u8,

    pub kernel_size: (u32, u32),
    pub options: ConvOptions<2>,
    pub out_shape_y: usize,
    pub out_shape_x: usize,
    pub has_bias: bool,
}

impl ConvolutionProblem {
    pub fn as_matmul_problem(&self) -> MatmulProblem {
        MatmulProblem {
            m: self.m,
            n: self.n,
            k: self.k,
            batches: (vec![], vec![]),
            lhs_layout: self.lhs_layout,
            rhs_layout: self.rhs_layout,
            lhs_line_size: self.lhs_line_size,
            rhs_line_size: self.rhs_line_size,
            out_line_size: self.out_line_size,
        }
    }
}