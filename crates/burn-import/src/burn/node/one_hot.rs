use super::{Node, NodeCodegen};
use crate::burn::{ScalarType, TensorType, Type};

use burn::record::PrecisionSettings;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct OneHotNode {
    pub input: TensorType,
    pub num_classes: ScalarType,
    pub output: TensorType,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for OneHotNode {
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn input_types(&self) -> Vec<crate::burn::Type> {
        vec![
            Type::Tensor(self.input.clone()),
            Type::Scalar(self.num_classes.clone()),
        ]
    }

    fn forward(
        &self,
        scope: &mut crate::burn::Scope,
        node_position: usize,
    ) -> proc_macro2::TokenStream {
        let num_classes = &self.num_classes.name;

        let input = scope.tensor_use_owned(&self.input, node_position);
        let output = &self.output.name;

        quote! {
            let #output = #input.one_hot(#num_classes as usize);
        }
    }

    fn into_node(self) -> super::Node<PS> {
        Node::OneHot(self)
    }
}

#[cfg(test)]
mod tests {

    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{
        graph::BurnGraph, node::test::assert_tokens, ScalarKind, ScalarType, TensorType,
    };

    #[test]
    fn test_codegen_one_hot() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(OneHotNode::new(
            TensorType::new_int("input", 1),
            ScalarType::new("num_classes", ScalarKind::Int64),
            TensorType::new_int("output", 2),
        ));

        graph.register_input_output(
            vec!["input".to_string(), "num_classes".to_string()],
            vec!["output".to_string()],
        );

        let expected = quote! {
            use burn::tensor::Int;
            use burn::{
                module::Module,
                tensor::{backend::Backend, Tensor},
            };

            #[derive(Module, Debug)]
            pub struct Model<B: Backend> {
                phantom: core::marker::PhantomData<B>,
                device: burn::module::Ignored<B::Device>,
            }

            impl<B: Backend> Model <B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    Self {
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }

                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(
                    &self,
                    input: Tensor<B, 1, Int>,
                    num_classes: i64
                ) -> Tensor<B, 2, Int> {
                    let output = input.one_hot(num_classes as usize);
                    output
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
