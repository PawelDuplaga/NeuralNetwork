using MatrixAlgebraForDoubles;
using System;
using System.Runtime.CompilerServices;
using System.Security.Cryptography.X509Certificates;

public class NeuralNetwork
{
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
                { ActivationFunction.BinaryStep,  ActivationFunctions.BinaryStep },
                { ActivationFunction.LeakyReLU,   ActivationFunctions.LeakyReLU },
                { ActivationFunction.Softplus,    ActivationFunctions.Softplus },
                { ActivationFunction.Softsign,    ActivationFunctions.Softsign },
                { ActivationFunction.Sigmoid,     ActivationFunctions.Sigmoid },
                { ActivationFunction.Swish,       ActivationFunctions.Swish },
                { ActivationFunction.Tanh,        ActivationFunctions.Tanh },
                { ActivationFunction.ReLU,        ActivationFunctions.ReLU },
                { ActivationFunction.GELU,        ActivationFunctions.GELU },
                { ActivationFunction.ELU,         ActivationFunctions.ELU },
                { ActivationFunction.None,        ActivationFunctions.None }
            };


    public Matrix InputLayer => LAYERS.First();
    public Matrix OutputLayer => LAYERS.Last();
    public int NumberOfLayers => LAYERS.Count;

    #region CONSTUCTORS

    public NeuralNetwork(int numberOfHiddenLayers, int defaultSizeOfLayers)
    {
        for (int i = 0; i < numberOfHiddenLayers + 1; i++)
        {
            LAYERS.Add(new Matrix(1, defaultSizeOfLayers));
            WEIGHTS.Add(new Matrix(defaultSizeOfLayers, defaultSizeOfLayers));
        }
        LAYERS.Add(new Matrix(1, defaultSizeOfLayers));

        ACTIVATION_FUNCTIONS = new Func<double, double>[LAYERS.Count];
        for (int i = 0; i < LAYERS.Count; i++)
        {
            ACTIVATION_FUNCTIONS[i] = ActivationFuncDict[ActivationFunction.None];
        }
    }

    public NeuralNetwork(int numberOfHiddenLayers, int defaultSizeOfLayers,
        ActivationFunction HiddenLayersActivationFunction)
    {
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

    public NeuralNetwork(int numberOfHiddenLayers, int defaultSizeOfLayers,
        ActivationFunction HiddenLayersActivationFunction,
        ActivationFunction InputLayerActivationFunction,
        ActivationFunction OutputLayerActivationFunction)
    {
        for (int i = 0; i < numberOfHiddenLayers + 1; i++)
        {
            LAYERS.Add(new Matrix(1, defaultSizeOfLayers));
            WEIGHTS.Add(new Matrix(defaultSizeOfLayers, defaultSizeOfLayers));
        }
        LAYERS.Add(new Matrix(1, defaultSizeOfLayers));

        ACTIVATION_FUNCTIONS = new Func<double, double>[LAYERS.Count];
        ACTIVATION_FUNCTIONS[0] = ActivationFuncDict[InputLayerActivationFunction];
        ACTIVATION_FUNCTIONS[LAYERS.Count - 1] = ActivationFuncDict[OutputLayerActivationFunction];
        for (int i = 1; i < LAYERS.Count - 1; i++)
        {
            ACTIVATION_FUNCTIONS[i] = ActivationFuncDict[HiddenLayersActivationFunction];
        }
    }

    public NeuralNetwork(int[] NumberOfPerceptronsInLayers)
    {
        for (int i = 0; i < NumberOfPerceptronsInLayers.Length - 1; i++)
        {
            LAYERS.Add(new Matrix(1, NumberOfPerceptronsInLayers[i]));
            WEIGHTS.Add(new Matrix(NumberOfPerceptronsInLayers[i], NumberOfPerceptronsInLayers[i + 1]));
        }
        LAYERS.Add(new Matrix(1, NumberOfPerceptronsInLayers[NumberOfPerceptronsInLayers.Count() - 1]));

        ACTIVATION_FUNCTIONS = new Func<double, double>[LAYERS.Count];
        for (int i = 0; i < LAYERS.Count; i++)
        {
            ACTIVATION_FUNCTIONS[i] = ActivationFuncDict[ActivationFunction.None];
        }
    }

    public NeuralNetwork(int[] NumberOfPerceptronsInLayers,
        ActivationFunction AllLayersActivationFunction)
    {
        for (int i = 0; i < NumberOfPerceptronsInLayers.Length - 1; i++)
        {
            LAYERS.Add(new Matrix(1, NumberOfPerceptronsInLayers[i]));
            WEIGHTS.Add(new Matrix(NumberOfPerceptronsInLayers[i], NumberOfPerceptronsInLayers[i + 1]));
        }
        LAYERS.Add(new Matrix(1, NumberOfPerceptronsInLayers[NumberOfPerceptronsInLayers.Count() - 1]));

        ACTIVATION_FUNCTIONS = new Func<double, double>[LAYERS.Count];
        for (int i = 0; i < LAYERS.Count; i++)
        {
            ACTIVATION_FUNCTIONS[i] = ActivationFuncDict[AllLayersActivationFunction];
        }
    }

    public NeuralNetwork(int[] NumberOfPerceptronsInLayers,
        ActivationFunction HiddenLayersActivationFunction,
        ActivationFunction InputLayerActivationFunction,
        ActivationFunction OutputLayerActivationFunction)
    {
        for (int i = 0; i < NumberOfPerceptronsInLayers.Length - 1; i++)
        {
            LAYERS.Add(new Matrix(1, NumberOfPerceptronsInLayers[i]));
            WEIGHTS.Add(new Matrix(NumberOfPerceptronsInLayers[i], NumberOfPerceptronsInLayers[i + 1]));
        }
        LAYERS.Add(new Matrix(1, NumberOfPerceptronsInLayers[NumberOfPerceptronsInLayers.Count() - 1]));

        ACTIVATION_FUNCTIONS = new Func<double, double>[LAYERS.Count];
        ACTIVATION_FUNCTIONS[0] = ActivationFuncDict[InputLayerActivationFunction];
        ACTIVATION_FUNCTIONS[LAYERS.Count - 1] = ActivationFuncDict[OutputLayerActivationFunction];
        for (int i = 1; i < LAYERS.Count - 1; i++)
        {
            ACTIVATION_FUNCTIONS[i] = ActivationFuncDict[HiddenLayersActivationFunction];
        }
    }

    #endregion

    public Matrix this[int i]
    {
        get => LAYERS[i];
        set => LAYERS[i] = value;
    }

    public double[] RunNetwork(double[] inputs)
    {
        if (inputs.Length != this.InputLayer.columns)
            throw new Exception($"Wrong number or inputs, Correct number of inputs is {this.InputLayer.columns}");

        if (inputs == null)
            throw new ArgumentNullException();

        for(int i = 0; i< inputs.Length; i++)
        {
            this.InputLayer[0,i] = inputs[i];
        }

        for(int i =0; i<this.NumberOfLayers-1; i++)
        {
            LAYERS[i+1] =  LAYERS[i] * WEIGHTS[i];
        }

        double[] output = new double[this.OutputLayer.columns];
        
        for(int i =0; i < output.Length; i++)
        {
            output[i] = this.OutputLayer[0,i];
        }

        return output;
    }


    public void SetActivationFunctionForLayer(int indexOfLayer, ActivationFunction enum_)
    {
        ACTIVATION_FUNCTIONS[indexOfLayer] = ActivationFuncDict[enum_];
    }

    public void SetActivationFunctionForLayer(int indexOfLayer, Func<double,double> customFunc)
    {
        ACTIVATION_FUNCTIONS[indexOfLayer] = customFunc;
    }




    public NeuralNetwork RandomizeWeigghts(double minValue, double maxValue)
    {
        for(int i =0; i < this.WEIGHTS.Count; i++)
        {
            this.WEIGHTS[i].FillRandomInRange(minValue, maxValue);
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