use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Agent {
    pub fitness: f64,
    pub nn: NeuralNetwork,
    pub rank: isize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NeuralNetwork {
    pub input_nodes: usize,
    pub layer_sizes: Vec<usize>,
    pub output_nodes: usize,
    pub edge_count: usize,
    pub nodes: Vec<Vec<Node>>,
}
/*
struct WeightMatrix {
    pub data: Vec<Vec<f64>>,
}

impl WeightMatrix {
    pub fn from_node_lists(nodes: Vec<Node>) -> Self {

    }
}
*/
struct MutationType {
    pub mutation: fn(&mut NeuralNetwork) -> bool,
    pub weight: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Node {
    pub bias: f64,
    //edges stored in an adjacency list
    pub incoming_edges: Vec<Edge>,
    pub outgoing_edges: Vec<Edge>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Edge {
    input: [usize; 2],
    out: [usize; 2],
    weight: f64,
}

impl NeuralNetwork {
    pub fn rand_wb() -> f64 {
        rand::random::<f64>() * 2. - 1.
    }
    pub fn new(input_nodes: usize, output_nodes: usize) -> Self {
        //create the simplest topology network
        //create input and output nodes
        let mut nodes = vec![Vec::new(), Vec::new()];
        for _ in 0..input_nodes {
            nodes[0].push(Node {
                bias: 0.,
                incoming_edges: Vec::new(),
                outgoing_edges: Vec::new(),
            });
        }
        for _ in 0..output_nodes {
            nodes[1].push(Node {
                bias: Self::rand_wb(),
                incoming_edges: Vec::new(),
                outgoing_edges: Vec::new(),
            });
        }
        //create edges between input and output nodes
        for i in 0..input_nodes {
            for j in 0..output_nodes {
                nodes[0][i].outgoing_edges.push(Edge {
                    input: [0usize, i],
                    out: [1usize, j],
                    weight: Self::rand_wb(),
                });
                let edge = nodes[0][i].outgoing_edges.last().unwrap().clone();
                nodes[1][j].incoming_edges.push(edge);
            }
        }
        Self {
            input_nodes,
            layer_sizes: vec![input_nodes, output_nodes],
            output_nodes,
            edge_count: input_nodes * output_nodes,
            nodes,
        }
    }

    pub fn rand_node(&mut self) -> [usize; 2] {
        //return a random node
        let mut node = [0usize, rand::random::<i32>() as usize % (self.layer_sizes.iter().sum::<usize>() - self.output_nodes)];
        while node[1] >= self.layer_sizes[node[0]] {
            node[1] -= self.layer_sizes[node[0]];
            node[0] += 1;
        }
        node
    }

    pub fn rand_edge(&mut self) -> [usize; 4] {
        //return a random edge
        let mut edge = [0usize, 0usize, 0usize, 0usize];
        let mut edge_index = rand::random::<u32>() as usize % self.edge_count;
        for i in 0..self.layer_sizes.len() {
            for j in 0..self.layer_sizes[i] {
                for k in 0..self.nodes[i][j].outgoing_edges.len() {
                    if edge_index == 0 {
                        edge = [i, j, self.nodes[i][j].outgoing_edges[k].out[0], self.nodes[i][j].outgoing_edges[k].out[1]];
                        break;
                    }
                    edge_index -= 1;
                }
            }
        }
        edge
    }

    pub fn activation_function(x: f64) -> f64 {
        Self::elu(x)
    }
    
    pub fn relu(x: f64) -> f64 {
        x.max(0.0)
    }
    
    pub fn elu(x: f64) -> f64 {
        if x > 0.0 {
            x
        } else {
            0.1 * (x.exp() - 1.0)
        }
    }
    
    pub fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
    

    /// Forward propagation
    pub fn predict(&self, input: Vec<f64>) -> Vec<f64> {
        //resize values vector
        let mut values = Vec::new();
        for i in 0..self.layer_sizes.len() {
            values.push(vec![0.0; self.layer_sizes[i]]);
        }
        //set input values
        for i in 0..self.input_nodes {
            values[0][i] = input[i];
        }
        //calculate values for all nodes
        for i in 1..self.layer_sizes.len() {
            for j in 0..self.layer_sizes[i] {
                let mut sum = self.nodes[i][j].bias;
                for k in 0..self.nodes[i][j].incoming_edges.len() {
                    let edge = &self.nodes[i][j].incoming_edges[k];
                    sum += edge.weight * values[edge.input[0]][edge.input[1]];
                }
                values[i][j] = NeuralNetwork::activation_function(sum);
            }
        }
        //return output values
        let mut output = Vec::new();
        for i in 0..self.output_nodes {
            output.push(values[self.layer_sizes.len() - 1][i]);
        }
        output
    }

    pub fn add_connection_rand(&mut self) -> bool {
        //find two random nodes and add a connection between them
        let mut input = self.rand_node();
        let mut out = self.rand_node();
        //check if the nodes are on the same layer and if the connection already exists
        //if so, find new nodes
        let mut tries = 0; //hacky but idc
        let max_tries = 100;
        while input[0] == out[0] || self.nodes[input[0]][input[1]].outgoing_edges.iter().any(|x| x.out == out)  {
            input = self.rand_node();
            out = self.rand_node();
            tries += 1;
            if tries >= max_tries {
                return false;
            }
        }
        self.add_connection(input, out);
        true
    }


    pub fn add_connection(&mut self, mut input: [usize; 2], mut out: [usize; 2]) {
        //check the direction of the connection
        if input[0] > out[0] {
            //swap the nodes if the connection is in the wrong direction
            let temp = input;
            input = out;
            out = temp;
        }
        //add connection between two nodes
        let weight = Self::rand_wb();
        self.nodes[input[0]][input[1]].outgoing_edges.push(Edge {
            weight: weight,
            input: input,
            out: out,
        });
        self.nodes[out[0]][out[1]].incoming_edges.push(Edge {
            weight: weight,
            input: input,
            out: out,
        });
        self.edge_count += 1;
    }

    pub fn add_node_rand(&mut self) -> bool {
        //find a random edge and add a node to it
        let edge = self.rand_edge();
        self.add_node(edge);
        true // idk if anything could fail
    }

    pub fn add_node(&mut self, edge: [usize; 4]) {
        let mut edge = edge;
        //check the distance between the two nodes
        let distance = (edge[0] as i32 - edge[2] as i32).abs() as u32;
        //if distance is 1, add new layer, otherwise add node to a random existing layer in between
        let layer = if distance == 1 {
            edge[0] + 1
        } else {
            (rand::random::<u32>() % (distance - 1) + edge[0] as u32 + 1) as usize
        };
        if distance == 1 {
            //add new layer
            self.nodes.insert(layer, Vec::new());
            self.layer_sizes.insert(layer, 0);
        }
        //add new node
        self.nodes[layer].push(Node {
            bias: Self::rand_wb(),
            incoming_edges: Vec::new(),
            outgoing_edges: Vec::new(),
        });
        self.layer_sizes[layer] += 1;
        if distance == 1 {
            edge[2] += 1;
            //update the layer indices of the nodes
            //this could be done drastically more efficiently TODO
            for i in 0..self.layer_sizes.len() {
                for j in 0..self.layer_sizes[i] {
                    for k in 0..self.nodes[i][j].outgoing_edges.len() {
                        if self.nodes[i][j].outgoing_edges[k].out[0] >= layer {
                            self.nodes[i][j].outgoing_edges[k].out[0] += 1;
                        }
                        if self.nodes[i][j].outgoing_edges[k].input[0] >= layer {
                            self.nodes[i][j].outgoing_edges[k].input[0] += 1;
                        }
                    }
                    for k in 0..self.nodes[i][j].incoming_edges.len() {
                        if self.nodes[i][j].incoming_edges[k].out[0] >= layer {
                            self.nodes[i][j].incoming_edges[k].out[0] += 1;
                        }
                        if self.nodes[i][j].incoming_edges[k].input[0] >= layer {
                            self.nodes[i][j].incoming_edges[k].input[0] += 1;
                        }
                    }
                }
            }
        }
        //add edges between the new node and the two nodes
        let weight = Self::rand_wb();
        let new_node = [layer, self.nodes[layer].len() - 1];
        self.nodes[edge[0]][edge[1]].outgoing_edges.push(Edge {
            weight: weight,
            input: [edge[0], edge[1]],
            out: new_node,
        });
        self.nodes[layer][new_node[1]].incoming_edges.push(Edge {
            weight: weight,
            input: [edge[0], edge[1]],
            out: new_node,
        });
        let weight = Self::rand_wb();
        self.nodes[layer][new_node[1]].outgoing_edges.push(Edge {
            weight: weight,
            input: new_node,
            out: [edge[2], edge[3]],
        });
        self.nodes[edge[2]][edge[3]].incoming_edges.push(Edge {
            weight: weight,
            input: new_node,
            out: [edge[2], edge[3]],
        });
        //remove the old edge
        self.nodes[edge[0]][edge[1]].outgoing_edges.retain(|x| x.out != [edge[2], edge[3]]);
        self.nodes[edge[2]][edge[3]].incoming_edges.retain(|x| x.input != [edge[0], edge[1]]);
        self.edge_count += 1;
    }

    pub fn change_weight_rand(&mut self) -> bool {
        //find a random edge and change its weight
        let edge = self.rand_edge();
        self.change_weight(edge);
        true // idk if anything could fail
    }

    pub fn change_weight(&mut self, edge: [usize; 4]) {
        //change the weight of an edge
        let weight = Self::rand_wb();
        //println!("{:?}", self.nodes[edge[0]][edge[1]].outgoing_edges);
        //println!("{:?}", self.nodes[edge[2]][edge[3]].incoming_edges);
        self.nodes[edge[0]][edge[1]].outgoing_edges.iter_mut().find(|x| x.out == [edge[2], edge[3]]).unwrap().weight = weight;
        self.nodes[edge[2]][edge[3]].incoming_edges.iter_mut().find(|x| x.input == [edge[0], edge[1]]).unwrap().weight = weight;
    }

    pub fn change_bias_rand(&mut self) -> bool{
        //find a random node and change its bias
        let node = self.rand_node();
        self.change_bias(node);
        true // idk if anything could fail
    }

    pub fn change_bias(&mut self, node: [usize; 2]) {
        //change the bias of a node
        let bias = Self::rand_wb();
        self.nodes[node[0]][node[1]].bias = bias;
    }
    pub fn shift(value: f64) -> f64 {
        (value + (rand::random::<f64>() * 2. - 1.).powi(2) / 2.).max(-1.).min(1.)
    }


    pub fn shift_weight_rand(&mut self) -> bool {
        //find a random edge and shift its weight
        let edge = self.rand_edge();
        self.shift_weight(edge);
        true // idk if anything could fail
    }

    pub fn shift_weight(&mut self, edge: [usize; 4]) {
        //shift the weight of an edge
        let shift = Self::shift(self.nodes[edge[0]][edge[1]].outgoing_edges.iter().find(|x| x.out == [edge[2], edge[3]]).unwrap().weight);
        self.nodes[edge[0]][edge[1]].outgoing_edges.iter_mut().find(|x| x.out == [edge[2], edge[3]]).unwrap().weight = shift;
        self.nodes[edge[2]][edge[3]].incoming_edges.iter_mut().find(|x| x.input == [edge[0], edge[1]]).unwrap().weight = shift;
    }

    pub fn shift_bias_rand(&mut self) -> bool{
        //find a random node and shift its bias
        let node = self.rand_node();
        self.shift_bias(node);
        true // idk if anything could fail
    }

    pub fn shift_bias(&mut self, node: [usize; 2]) {
        //shift the bias of a node
        let shift = Self::shift(self.nodes[node[0]][node[1]].bias);
        self.nodes[node[0]][node[1]].bias = shift;
    }

}

impl Agent {
    pub fn new(input_nodes: usize, output_nodes: usize) -> Self {
        Self {
            fitness: 0.0,
            nn: NeuralNetwork::new(input_nodes, output_nodes),
            rank: -1,
        }
    }
    
    pub fn mutate(&mut self, mutations: usize) -> Self {
        let mutation_types = [
            MutationType {
                mutation: NeuralNetwork::add_connection_rand,
                weight: 5.,
            },
            MutationType {
                mutation: NeuralNetwork::add_node_rand,
                weight: 2.5,
            },
            MutationType {
                mutation: NeuralNetwork::change_weight_rand,
                weight: 7.5,
            },
            MutationType {
                mutation: NeuralNetwork::change_bias_rand,
                weight: 3.,
            },
            MutationType {
                mutation: NeuralNetwork::shift_weight_rand,
                weight: 20.,
            },
            MutationType {
                mutation: NeuralNetwork::shift_bias_rand,
                weight: 7.5,
            }];
        for _ in 0..mutations {
            let mut rng = rand::thread_rng();
            let mut sum = 0.0;
            for i in 0..mutation_types.len() {
                sum += mutation_types[i].weight;
            }
            let mut rand = rng.gen_range(0.0..sum);
            let mut i = 0;
            while rand > mutation_types[i].weight {
                rand -= mutation_types[i].weight;
                i += 1;
            }
            (mutation_types[i].mutation)(&mut self.nn);
        }
        self.clone()
    }
}

