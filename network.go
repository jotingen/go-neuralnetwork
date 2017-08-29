package network

import (
	"github.com/jotingen/go-neuron"
)

type Network struct {
	Neurons [][]neuron.Neuron `json:"Neuron"`
	Output  [][]float64
	Error   [][]float64
	Delta   [][]float64
}

func New(layer []int)  (n Network) {

	//Build neurons
	for l := range layer {
		var neuronLayer []neuron.Neuron
		var outputLayer []float64
		var errorLayer []float64
		var deltaLayer []float64
		for n := 0; n < layer[l]; n++ {
			neuronLayer = append(neuronLayer, neuron.Neuron{})
			outputLayer = append(outputLayer, 0.0)
			errorLayer = append(errorLayer, 0.0)
			deltaLayer = append(deltaLayer, 0.0)
		}
		n.Neurons = append(n.Neurons, neuronLayer)
		n.Output = append(n.Output, outputLayer)
		n.Error = append(n.Error, errorLayer)
		n.Delta = append(n.Delta, deltaLayer)
	}

	//Initialize weights
	n.Calc([]float64{0, 0})

	return n

}

func (n *Network) Calc(inputs []float64) (outputs []float64) {
	for i := range n.Neurons {
		outputs = nil
		if i == 0 {
			//first layer uses inputs
			for m := range n.Neurons[i] {
				n.Output[i][m] = n.Neurons[i][m].Calc(inputs)
				outputs = append(outputs, n.Output[i][m])
			}
		} else {
			//next layers use previous layer
			for m := range n.Neurons[i] {
				n.Output[i][m] = n.Neurons[i][m].Calc(n.Output[i-1])
				outputs = append(outputs, n.Output[i][m])
			}
		}
		inputs = outputs
	}
	return
}

func (n *Network) Train(inputs []float64, target []float64) {
	for i := len(n.Neurons) - 1; i >= 0; i-- {
		for m := 0; m < len(n.Neurons[i]); m++ {
			if i == len(n.Neurons)-1 {
				//Output Layer
				n.Error[i][m] = target[m] - n.Output[i][m]
				n.Delta[i][m] = n.Error[i][m] * n.Neurons[i][m].Derivative(n.Output[i][m])

			} else {
				//Remaining Layers
				n.Error[i][m] = 0
				for j := 0; j < len(n.Neurons[i+1]); j++ {
					n.Error[i][m] += n.Delta[i+1][j] * n.Neurons[i+1][j].Weight[m]
				}
				n.Delta[i][m] = n.Error[i][m] * n.Neurons[i][m].Derivative(n.Output[i][m])
			}
		}
	}

	learningRate := 0.05
	for i := len(n.Neurons) - 1; i >= 0; i-- {
		for m := 0; m < len(n.Neurons[i]); m++ {
			if i == 0 {
				//Input Layer
				for w := range inputs {
					n.Neurons[i][m].Weight[w] += learningRate * inputs[w] * n.Delta[i][m]
				}
			} else {
				//Remaining Layers
				for w := range n.Neurons[i][m].Weight {
					if w == len(n.Neurons[i][m].Weight)-1 {
						//Bias is last
						n.Neurons[i][m].Weight[w] += learningRate * 1 * n.Delta[i][m]
					} else {
						n.Neurons[i][m].Weight[w] += learningRate * n.Output[i-1][w] * n.Delta[i][m]
					}
				}
			}
		}
	}
}
