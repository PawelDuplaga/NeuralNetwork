using MatrixAlgebraForDoubles;
using System.Runtime.CompilerServices;
using System.Security.Cryptography.X509Certificates;

public class NeuralNetwork
{
    private int numberOfHiddenLayers { get; set; }
    public List<Matrix> LAYERS = new List<Matrix>();
    public List<Matrix> WEIGHTS = new List<Matrix>();
    public Func<double, double>[] ACTIVATION_FUNCTIONS;


    public enum ActivationFunction
    {
        HardSigmoid,
        BinaryStep,
        LeakyReLU,
        Softplus,
        Softsign,
        Sigmoid,
        Swish,
        Tanh,
        ReLU,
        GELU,
        ELU,
        None
    }

    public Dictionary<ActivationFunction, Func<double, double>> ActivationFuncDict
        = new Dictionary<ActivationFunction, Func<double, double>>()
            {
                { ActivationFunction.HardSigmoid, ActivationFunctions.HardSigmoid },
                { ActivationFunction.BinaryStep, ActivationFunctions.BinaryStep },
                { ActivationFunction.LeakyReLU,ActivationFunctions.LeakyReLU},
                { ActivationFunction.Softplus, ActivationFunctions.Softplus },
                { ActivationFunction.Softsign, ActivationFunctions.Softsign },
                { ActivationFunction.Sigmoid, ActivationFunctions.Sigmoid },
                { ActivationFunction.Swish, ActivationFunctions.Swish },
                { ActivationFunction.Tanh, ActivationFunctions.Tanh },
                { ActivationFunction.ReLU, ActivationFunctions.ReLU },
                { ActivationFunction.GELU, ActivationFunctions.GELU},
                { ActivationFunction.ELU, ActivationFunctions.ELU },
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

        ACTIVATION_FUNCTIONS = new Func<double, double>[LAYERS.Count];
        for (int i = 0; i < LAYERS.Count; i++)
        {
            ACTIVATION_FUNCTIONS[i] = ActivationFuncDict[ActivationFunction.None];
        }
    }

    public NeuralNetwork(int numberOfHiddenLayers, int defaultSizeOfLayers, ActivationFunction HiddenLayersActivationFunction)
    {
        this.numberOfHiddenLayers = numberOfHiddenLayers;
        for (int i = 0; i < numberOfHiddenLayers + 1; i++)
        {
            LAYERS.Add(new Matrix(1, defaultSizeOfLayers));
            WEIGHTS.Add(new Matrix(defaultSizeOfLayers, defaultSizeOfLayers));
        }
            LAYERS.Add(new Matrix(1, defaultSizeOfLayers));

        ACTIVATION_FUNCTIONS = new Func<double, double>[LAYERS.Count];
        for (int i = 0; i < LAYERS.Count; i++)
        {
            ACTIVATION_FUNCTIONS[i] = ActivationFuncDict[HiddenLayersActivationFunction];
        }
    }

    public NeuralNetwork(int[] NumberOfPerceptronsInLayers , ActivationFunction HiddenLayersActivationFunction)
    {
        this.numberOfHiddenLayers = numberOfHiddenLayers;
        for (int i = 0; i < numberOfHiddenLayers + 1; i++)
        {
            LAYERS.Add(new Matrix(1, NumberOfPerceptronsInLayers[i]));
            WEIGHTS.Add(new Matrix(NumberOfPerceptronsInLayers[i], NumberOfPerceptronsInLayers[i+1]));
        }
        LAYERS.Add(new Matrix(1, NumberOfPerceptronsInLayers[NumberOfPerceptronsInLayers.Count() - 1]));

        ACTIVATION_FUNCTIONS = new Func<double, double>[LAYERS.Count];
        for (int i = 0; i < LAYERS.Count; i++)
        {
            ACTIVATION_FUNCTIONS[i] = ActivationFuncDict[HiddenLayersActivationFunction];
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


    private static class ActivationFunctions
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

    //by Pawel Duplaga

}