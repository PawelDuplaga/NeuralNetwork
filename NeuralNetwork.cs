using MatrixAlgebraForDoubles;
using System.Runtime.CompilerServices;
using System.Security.Cryptography.X509Certificates;

public class NeuralNetwork
{
    private int numberOfHiddenLayers { get; set; }
    public List<Matrix> LAYERS = new List<Matrix>();
    public List<Matrix> WEIGHTS = new List<Matrix>();
    public List<Func<double,double>> CUSTOM_ACTIVATION_FUNCTIONS = new List<Func<double,double>>();


    public enum ActivationFunction
    {
        ReLU ,
        Sigmoid,
        Custom,
        None
    }

    public Dictionary<ActivationFunction, Func<double, double>> ActivationFuncDict
        = new Dictionary<ActivationFunction, Func<double, double>>()
            {
                { ActivationFunction.ReLU, ActivationFunctions.ReLU() },
                { ActivationFunction.Sigmoid, ActivationFunctions.Sigmoid() }        
            };


    public NeuralNetwork(int numberOfHiddenLayers, int defaultSizeOfLayers)
    {
        this.numberOfHiddenLayers = numberOfHiddenLayers;
        for(int i =0; i< numberOfHiddenLayers+1; i++)
        {
            LAYERS.Add(new Matrix(1,defaultSizeOfLayers));
            WEIGHTS.Add(new Matrix(defaultSizeOfLayers,defaultSizeOfLayers));
        }
        LAYERS.Add(new Matrix(1, defaultSizeOfLayers));

        for(int i = 0; i < LAYERS.Count; i++)
        {
            //default NO FUNCTION
            CUSTOM_ACTIVATION_FUNCTIONS.Add(x => x);
        }
    }


    /// <summary>
    /// Constructor for neural netork using additional parameter - default Function
    /// <para> Default function is setted like this :</para>
    /// <br>InputLayer   func = x => x </br>
    /// <br>HiddenLayer1 func = defaultFunction</br>
    /// <br>HiddenLayer2 func = defaultFunction</br>
    /// <br>...</br>
    /// <br>HiddenLayerN func = defaultFunction</br>
    /// <br>OutputLayer  func = defaultFunction</br>
    /// </summary>
    /// <param name="numberOfHiddenLayers">lol</param>
    /// <param name="defaultSizeOfLayers"></param>
    /// <param name="defaultFunction"></param>
    public NeuralNetwork(int numberOfHiddenLayers, int defaultSizeOfLayers, ActivationFunction defaultFunction)
    {
        this.numberOfHiddenLayers = numberOfHiddenLayers;
        for (int i = 0; i < numberOfHiddenLayers + 1; i++)
        {
            LAYERS.Add(new Matrix(1, defaultSizeOfLayers));
            WEIGHTS.Add(new Matrix(defaultSizeOfLayers, defaultSizeOfLayers));
        }
        LAYERS.Add(new Matrix(1, defaultSizeOfLayers));

        for(int i = 0; i < LAYERS.Count; i++)
        {
            //default NO FUNCTION
            CUSTOM_ACTIVATION_FUNCTIONS.Add(x => x);
        }
    }


    public Matrix GetInputs() => LAYERS.First();
    public Matrix GetOutput() => LAYERS.Last();






    private Func<double,double> RunActivationFunction(double x, ActivationFunction enum_) 
    {
        switch (enum_) 
        {
            case ActivationFunction.ReLU    : return Helpers.ReLU();
            case ActivationFunction.Sigmoid : return Helpers.Sigmoid();
            default                         : return Helpers.Sigmoid();
        }

    }

    private double RunCustomActivationFunction(double x, Func<double,double> func)
    {
        return func(x);
    }

    public NeuralNetwork RandomizeWeigghts()
    {
        for(int i =0; i < this.WEIGHTS.Count; i++)
        {
            this.WEIGHTS[i].FillRandomInRange(0, 1);
        }
    
        return this;
    }

    public void PrintNeuralNetwork()
    {
        for(int i = 0; i < WEIGHTS.Count; i++)
        {
            LAYERS[i].PrintMatrix();
            Console.Write("       ");
            WEIGHTS[i].PrintMatrix();
            Console.Write("       ");
        }
        LAYERS[LAYERS.Count - 1].PrintMatrix();

    }


    private class ActivationFunctions
    {

        public static Func<double,double> ReLU() => x => Math.Max(x,0);
        public static Func<double,double> Sigmoid() => x => x;




    }

}