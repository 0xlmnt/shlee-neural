use rand::{thread_rng, Rng};

pub struct LayerSpecification {
    pub neurons: usize,
}

pub struct Network {
    layers: Vec<Layer>,
}

struct Layer {
    neurons: Vec<Neuron>,
}

struct Neuron {
    weights: Vec<f32>,
    bias: f32,
}


impl Network {
    fn propagate(&self, inputs: Vec<f32>) -> Vec<f32> {
        self.layers
            .iter()
            .fold(inputs, |inputs, layer| {
                layer.propagate(inputs)
            })
    }

    fn random(layers: &[LayerSpecification]) -> Self {
        assert!(layers.len() > 1);


        let layers = layers.windows(2)
            .map(|layers| {
                Layer::random(layers[0].neurons, layers[1].neurons)
            })
            .collect();

        Self {
            layers
        }
    }
}

impl Layer {
    fn propagate(&self, inputs: Vec<f32>) -> Vec<f32> {
        self.neurons
            .iter()
            .map(|neuron| {
                neuron.propagate(&inputs)
            })
            .collect()
    }

    fn random(n_input: usize, n_output: usize) -> Self {
        Self {
            neurons: (0..n_output).map(|_| {
                Neuron::random(n_input)
            }).collect()
        }
    }
}

impl Neuron {
    fn propagate(&self, inputs: &[f32]) -> f32 {
        assert_eq!(self.weights.len(), inputs.len());

        let output = inputs.iter()
            .zip(&self.weights)
            .map(|(input, weight)| {
                input * weight
            })
            .sum::<f32>();

        (output + self.bias).max(0.0)
    }

    fn random(out_size: usize) -> Self {
        let mut rng = thread_rng();
        let gen = || -> f32 {
            rng.gen_range(-1.0..=1.0)
        };

        let bias = gen();

        let weights = (0..out_size)
            .map(|_| {
                gen()
            })
            .collect();

        Self {
            bias,
            weights
        }
    }
}
