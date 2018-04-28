package HomeWork3;

import javafx.util.Pair;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemoveRange;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.PriorityQueue;
import java.util.Random;

class DistanceCalculator {

    public static final int INFINITY = Integer.MAX_VALUE;

    /**
    * We leave it up to you wheter you want the distance method to get all relevant
    * parameters(lp, efficient, etc..) or have it has a class variables.
    */
    public double distance (Instance one, Instance two, int p, Knn.DistanceCheck distanceCheck, double maxDistance) {
        double distance = 0;
        if (distanceCheck == distanceCheck.Efficient) {
            if (p == INFINITY) {
                distance = efficientLInfinityDistance(one, two, maxDistance);
            } else {
                distance = efficientLpDisatnce(one, two, p, maxDistance);
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
        int d = one.numAttributes() - 1;
        for (int i = 0; i < d; i++) {
            distance += Math.pow(Math.abs(one.value(i) - two.value(i)), p);
        }

        return Math.pow(distance, (double) (1 / p));
    }

    /**
     * Returns the L infinity distance between 2 instances.
     * @param one
     * @param two
     * @return
     */
    private double lInfinityDistance(Instance one, Instance two) {
        double distance = 0;
        int d = one.numAttributes() - 1;
        double maxDistance = Math.abs(one.value(0) - two.value(0));
        for (int i = 1; i < d; i++) {
            distance = Math.abs(one.value(i) - two.value(i));
            if (maxDistance < distance) {
                maxDistance = distance;
            }
        }
        return maxDistance;
    }

    /**
     * Returns the Lp distance between 2 instances, while using an efficient distance check.
     * @param one
     * @param two
     * @return
     */
    private double efficientLpDisatnce(Instance one, Instance two, int p, double maxDistance) {
        double distance = 0;
        int d = one.numAttributes() - 1;
        for (int i = 0; i < d; i++) {
            distance += Math.pow(one.value(i) - two.value(i), p);
            if (Math.pow(distance, (double) 1 / p) > maxDistance) {
                break;
            }
        }
        return Math.pow(distance, (double) (1 / p));
    }

    /**
     * Returns the Lp distance between 2 instances, while using an efficient distance check.
     * @param one
     * @param two
     * @return
     */
    private double efficientLInfinityDistance(Instance one, Instance two, double maxDistance) {
        double distance = 0;
        int d = one.numAttributes() - 1;
        double maxDistanceSeen = Math.abs(one.value(0) - two.value(0));
        for (int i = 1; i < d; i++) {
            if (maxDistanceSeen < maxDistance) {
                distance = Math.abs(one.value(i) - two.value(i));
                if (maxDistanceSeen < distance) {
                    maxDistanceSeen = distance;
                }
            } else {
                break;
            }
        }
        return maxDistanceSeen;
    }
}

public class Knn implements Classifier {

    private static final int INFINITY = Integer.MAX_VALUE;

    /**
     * State of Knn class
     */

    private Instances m_trainingInstances;
    private int K;
    private int P;
    private DistanceCheck distanceCheck;
    private int folds;
    private Weights weights;
    private DistanceCalculator distanceCalculator = new DistanceCalculator();


    @Override
    public void buildClassifier(Instances instances) throws Exception {
        // Fallback to defaults
        this.m_trainingInstances = instances;
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
        ArrayList<Pair<Instance, Double>> knn = findNearestNeighbors(instance);
        double sum = 0.0;
        double sumOfWeights = 0.0;
        for (int i = 0; i < knn.size(); i++) {
            if(this.weights == Weights.Uniform) {
                sum += knn.get(i).getKey().classValue();
                sumOfWeights = knn.size();
            }
            else {
                double distance = knn.get(i).getValue();

                if(distance != 0) {
                    double weight = ((double) 1 / Math.pow(distance, 2));
                    sumOfWeights += weight;
                    sum += weight * (knn.get(i).getKey().classValue());
                }
            }
        }
        return knn.size() > 1 ? sum / sumOfWeights : knn.get(0).getKey().classValue();
    }

    /**
     * Caclcualtes the average error on a give set of instances.
     * The average error is the average absolute error between the target value and the predicted
     * value across all instances.
     * @param instances
     * @return
     */
    public double calcAvgError (Instances instances){
        double sum = 0.0;

        for(int i=0;i<instances.size();i++) {
            Instance instance = instances.get(i);
            sum += Math.abs(instance.classValue() - regressionPrediction(instance));
        }

        return sum / instances.size();
    }

    /**
     * Calculates the cross validation error, the average error on all folds.
     * @param instances instances used for the cross validation
     * @param num_of_folds The number of folds to use.
     * @return The cross validation error.
     */
    public double crossValidationError(Instances instances, int num_of_folds) throws Exception {
        // Shuffle the data
        Random rand = new Random();
        Instances randData = new Instances(instances);
        randData.randomize(rand);

        int instancesPerBin = (int) Math.ceil((double) randData.size() / num_of_folds);
        double sum = 0;

        for(int i=0;i<num_of_folds - 1;i++) {
            Instances testData = removeInstancesFromData(randData, i * instancesPerBin, ((i + 1) * instancesPerBin), true);
            Instances trainingData = removeInstancesFromData(randData, i * instancesPerBin, ((i + 1) * instancesPerBin), false);

            this.m_trainingInstances = trainingData;
            sum += calcAvgError(testData);
        }

        Instances testData = removeInstancesFromData(randData, (num_of_folds - 1) * instancesPerBin, randData.size(), true);
        Instances trainingData = removeInstancesFromData(randData, (num_of_folds - 1) * instancesPerBin, randData.size(), false);

        this.m_trainingInstances = trainingData;
        sum += calcAvgError(testData);



        return sum / num_of_folds;
    }

    private Instances removeInstancesFromData(Instances data, int from, int to, boolean inverted) throws Exception {
        RemoveRange filter = new RemoveRange();
        if(!inverted){
            String[] options = { "-R" , createStringFromTo(from + 1, to + 1)};
            filter.setOptions(options);
        }
        else {
            String[] options = { "-R" , createStringFromTo(from + 1, to + 1), "-V"};
            filter.setOptions(options);
        }
        filter.setInputFormat(data);

        return Filter.useFilter(data, filter);
    }

    private String createStringFromTo(int from, int to) {
        String string = "";
        for (int i = from; i < to - 1; i++) {
            string += i + ",";
        }

        string += to - 1;

        return string;
    }


    /**
     * Finds the k nearest neighbors.
     * @param instance
     */
    public ArrayList<Pair<Instance, Double>> findNearestNeighbors(Instance instance) {

        PriorityQueue<Pair<Instance, Double>> minHeap = new PriorityQueue<>((one, two) -> (int)(one.getValue() - two.getValue()));
        PriorityQueue<Pair<Instance, Double>> maxHeap = new PriorityQueue<>((one, two) -> (int)(two.getValue() - one.getValue()));
        buildFirstKNodes(minHeap, maxHeap, instance);
        findKMins(minHeap, maxHeap, instance);

        return extractKMins(minHeap);

    }

    private ArrayList<Pair<Instance, Double>> extractKMins(PriorityQueue<Pair<Instance, Double>> minHeap) {
        ArrayList<Pair<Instance, Double>> kNN = new ArrayList<>(this.K);

        for(int i=0;i<this.K;i++) {
            kNN.add(minHeap.poll());
        }
        return kNN;
    }

    private void buildFirstKNodes(PriorityQueue<Pair<Instance, Double>> minHeap, PriorityQueue<Pair<Instance, Double>> maxHeap, Instance instance) {
        for(int i=0;i<this.K;i++) {
            Pair<Instance, Double> heapNode = getHeapNode(instance, this.m_trainingInstances.get(i), Double.POSITIVE_INFINITY);
            minHeap.add(heapNode);
            maxHeap.add(heapNode);
        }
    }

    private void findKMins(PriorityQueue<Pair<Instance, Double>> minHeap, PriorityQueue<Pair<Instance, Double>> maxHeap, Instance instance) {
        for (int i = this.K; i < this.m_trainingInstances.size(); i++) {
            double maxDistance = this.distanceCheck == DistanceCheck.Regular ? Double.POSITIVE_INFINITY : maxHeap.peek().getValue();
            Pair<Instance, Double> heapNode = getHeapNode(instance, this.m_trainingInstances.get(i),  maxDistance);
            if (heapNode.getValue() < maxHeap.peek().getValue()) {
                minHeap.add(heapNode);
                maxHeap.add(heapNode);
                Pair<Instance, Double> nodeToRemove = maxHeap.poll();
                minHeap.remove(nodeToRemove);
            }
        }
    }



    private Pair<Instance, Double> getHeapNode(Instance instance, Instance  instanceToCompare, double maxDistance) {
        double distance = distanceCalculator.distance(instance, instanceToCompare, this.P, this.distanceCheck, maxDistance);
        return new Pair<>(instanceToCompare, distance);
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
