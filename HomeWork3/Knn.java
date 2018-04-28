package HomeWork3;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

class DistanceCalculator {

    private static final int INFINITY = Integer.MAX_VALUE;

    /**
    * We leave it up to you wheter you want the distance method to get all relevant
    * parameters(lp, efficient, etc..) or have it has a class variables.
    */
    public double distance (Instance one, Instance two, int p, Knn.DistanceCheck distanceCheck) {
        double distance = 0;
        if (distanceCheck == distanceCheck.Efficient) {
            if (p == INFINITY) {
                distance = efficientLInfinityDistance(one, two);
            } else {
                distance = efficientLpDisatnce(one, two, p);
            }
        } else if (p == INFINITY) {
            distance = lInfinityDistance(one, two);
        } else {
            distance = lpDisatnce(one, two, p);
        }
        return distance;
    }

    /**
     * Returns the Lp distance between 2 instances.
     * @param one
     * @param two
     */
    private double lpDisatnce(Instance one, Instance two, int p) {
        double distance = 0;
        int d = one.numAttributes();
        for (int i = 0; i < d; i++) {
            distance += Math.pow(one.value(i) - two.value(i), p);
        }
        return Math.pow(distance, 1 / p);
    }

    /**
     * Returns the L infinity distance between 2 instances.
     * @param one
     * @param two
     * @return
     */
    private double lInfinityDistance(Instance one, Instance two) {
        double distance = 0;
        int d = one.numAttributes();
        double maxDistance = Math.abs(one.value(0) - two.value(0));
        for (int i = 1; i < d; i++) {
            distance = Math.abs(one.value(i) - two.value(i));
            if (maxDistance < distance) {
                maxDistance = distance;
            }
        }
        return distance;
    }

    /**
     * Returns the Lp distance between 2 instances, while using an efficient distance check.
     * @param one
     * @param two
     * @return
     */
    private double efficientLpDisatnce(Instance one, Instance two, int p) {
        double distance = 0;
        
    }

    /**
     * Returns the Lp distance between 2 instances, while using an efficient distance check.
     * @param one
     * @param two
     * @return
     */
    private double efficientLInfinityDistance(Instance one, Instance two) {
        return 0.0;
    }
}

public class Knn implements Classifier {

    /**
     * State of Knn class
     */

    private Instances m_trainingInstances;
    private int K;
    private int P;
    private DistanceCheck distanceCheck;
    private int folds;
    private Weights weights;


    @Override
    public void buildClassifier(Instances instances) {
        // Fallback to defaults
        this.K = 2;
        this.P = 2;
        this.folds = 10;
        this.distanceCheck = DistanceCheck.Regular;
        this.weights = Weights.Uniform;
    }

    /**
     * Build the knn classifier. In our case, simply stores the given instances for 
     * later use in the prediction.
     * @param instances
     */
    public void buildClassifier(Instances instances, int K, int P, int folds, DistanceCheck distanceCheck, Weights weights) throws Exception {
        this.m_trainingInstances = instances;
        this.K = K;
        this.P = P;
        this.folds = folds;
        this.distanceCheck = distanceCheck;
        this.weights = weights;
    }

    /**
     * Returns the knn prediction on the given instance.
     * @param instance
     * @return The instance predicted value.
     */
    public double regressionPrediction(Instance instance) {
        // Steps:
        // 1. find knn :: Instances
        // 2. calcAvgError :: Double
        // return it.

        return 0.0;
    }

    /**
     * Caclcualtes the average error on a give set of instances.
     * The average error is the average absolute error between the target value and the predicted
     * value across all insatnces.
     * @param insatnces
     * @return
     */
    public double calcAvgError (Instances insatnces){
        return 0.0;
    }

    /**
     * Calculates the cross validation error, the average error on all folds.
     * @param insances Insances used for the cross validation
     * @param num_of_folds The number of folds to use.
     * @return The cross validation error.
     */
    public double crossValidationError(Instances insances, int num_of_folds){
        return 0.0;
    }


    /**
     * Finds the k nearest neighbors.
     * @param instance
     */
    public Instances findNearestNeighbors(Instance instance) {
        // Steps:
        // 1. forEach i in instances => D(i, instances)
        // 2. save k min D
        return null;
    }

    /**
     * Cacluates the average value of the given elements in the collection.
     * @param
     * @return
     */
    public double getAverageValue (Instances instances) {
        return 0.0;
    }

    /**
     * Calculates the weighted average of the target values of all the elements in the collection
     * with respect to their distance from a specific instance.
     * @return
     */
    public double getWeightedAverageValue(Instances instances) {
        return 0.0;
    }


    @Override
    public double[] distributionForInstance(Instance arg0) throws Exception {
        // TODO Auto-generated method stub - You can ignore.
        return null;
    }

    @Override
    public Capabilities getCapabilities() {
        // TODO Auto-generated method stub - You can ignore.
        return null;
    }

    @Override
    public double classifyInstance(Instance instance) {
        // TODO Auto-generated method stub - You can ignore.
        return 0.0;
    }

    public enum DistanceCheck {
        Regular,
        Efficient
    }

    public enum Weights {
        Uniform,
        Weighted
    }

}
