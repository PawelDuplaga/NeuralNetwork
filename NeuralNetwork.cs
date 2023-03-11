using MatrixAlgebraForDoubles;
using System.Runtime.CompilerServices;

public class NeuralNetwork
{
    private int numberOfHiddenLayers { get; set; }
    public List<Matrix> LAYERS  = new List<Matrix>();
    public List<Matrix> WEIGHTS = new List<Matrix>();

    public NeuralNetwork(int numberOfHiddenLayers, int defaultSizeOfLayers)
    {
        this.numberOfHiddenLayers = numberOfHiddenLayers;
        for(int i =0; i< numberOfHiddenLayers; i++)
        {
            LAYERS.Add(new Matrix(1,defaultSizeOfLayers));

        }
    }


    public Matrix GetInputs() => LAYERS.First();
    public Matrix GetOutput() => LAYERS.Last(); 
    

}