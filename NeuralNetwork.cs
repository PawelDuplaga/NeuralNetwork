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
                { ActivationFunction.ReLU, ActivationFunctions.ReLU },
                { ActivationFunction.Sigmoid, ActivationFunctions.Sigmoid },
                { ActivationFunction.None, ActivationFunctions.None }
            
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

        public static Func<double, double> ReLU = x => Math.Max(x,0);

        public static Func<double, double> LeakyReLU = x => x > 0 ? x : 0.01 * x;

        public static Func<double, double> ELU = x => x > 0 ? x : Math.Exp(x) - 1;

        public static Func<double, double> Sigmoid = x => 1.0 / (1.0 + Math.Exp(-x));

        public static Func<double, double> HardSigmoid = x => Math.Max(0, Math.Min(1, 0.2 * x + 0.5));

        public static Func<double, double> Tanh = x => Math.Tanh(x);

        public static Func<double, double> Softplus = x => Math.Log(1 + Math.Exp(x));

        public static Func<double, double> Softsign = x => x / (1 + Math.Abs(x));

        public static Func<double, double> GELU = x => 0.5 * x * (1 + Math.Tanh(Math.Sqrt(2 / Math.PI) * (x + 0.044715 * Math.Pow(x, 3))));

        public static Func<double, double> Swish = x => x * Sigmoid(x);

        public static Func<double, double> BinaryStep = x => x < 0 ? 0 : 1;

        public static Func<double, double> None = x => x;

    }

}