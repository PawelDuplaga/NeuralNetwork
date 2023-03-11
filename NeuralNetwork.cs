using MatrixAlgebraForDoubles;
using System.Runtime.CompilerServices;

public class NeuralNetwork
{
    private int numberOfHiddenLayers { get; set; }
    public List<Matrix> LAYERS = new List<Matrix>();
    public List<Matrix> WEIGHTS = new List<Matrix>();

    enum ActivationFunction
    {
        ReLU
    }

    private Dictionary<ActivationFunction, double> ActivateFunctions =
        new Dictionary<ActivationFunction, double>() { };
 


    public NeuralNetwork(int numberOfHiddenLayers, int defaultSizeOfLayers)
    {
        this.numberOfHiddenLayers = numberOfHiddenLayers;
        for(int i =0; i< numberOfHiddenLayers+1; i++)
        {
            LAYERS.Add(new Matrix(1,defaultSizeOfLayers));
            WEIGHTS.Add(new Matrix(defaultSizeOfLayers,defaultSizeOfLayers));
        }
        LAYERS.Add(new Matrix(1, defaultSizeOfLayers));
    }

    public Matrix GetInputs() => LAYERS.First();
    public Matrix GetOutput() => LAYERS.Last();

    public delegate double ActivateFunc(double x);

    public static double ActivateFunctionTransform(double x, ActivateFunc acfunc)
    {
        return acfunc(x);
    }

    





    //public Matrix RunNetwork(double[] inputs)
    //{
    //    if (inputs.Length != GetInputs().columns)
    //        throw new ArgumentException("Wrong number of inputs given");    


    //}



    public NeuralNetwork RandomizeWeigghts()
    {
        for(int i =0; i < this.WEIGHTS.Count; i++)
            this.WEIGHTS[i].FillRandomInRange(0, 1);
    
        return this;
    }

    public void PrintNeuralNetwork()
    {
        for(int i = 0; i < WEIGHTS.Count; i++)
        {
            LAYERS[i].PrintMatrix();
            Console.Write("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
            WEIGHTS[i].PrintMatrix();
            Console.Write("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
        }
        LAYERS[LAYERS.Count - 1].PrintMatrix();

    }


    private class Helpers
    {

        public static double ReLU(double x)
        {
            return Math.Max(x, 0);
        }



    }

}