package HomeWork3;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import weka.core.Instances;

public class MainHW3 {

	private static int bestK;
	private static int bestW;
	private static int bestLp;



	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}

	public static Instances loadData(String fileName) throws IOException {
		BufferedReader datafile = readDataFile(fileName);
		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}

	public static void main(String[] args) throws Exception {
		Instances data = loadData("./HomeWork3/Data/auto_price.txt");
		FeatureScaler featureScaler = new FeatureScaler();
		Instances scaledData = featureScaler.scaleData(data);
		System.out.println("----------------------------");
		System.out.println("Results for original dataset: ");
		System.out.println("----------------------------");
		checkError(data, 10, true, Knn.DistanceCheck.Regular);
		System.out.println();
		System.out.println("----------------------------");
		System.out.println("Results for 159 folds: ");
		System.out.println("----------------------------");
		checkError(data, data.size(), false, Knn.DistanceCheck.Regular);
		checkError(data, data.size(), false, Knn.DistanceCheck.Efficient);
		System.out.println();
		System.out.println("----------------------------");
		System.out.println("Results for 50 folds: ");
		System.out.println("----------------------------");
		checkError(data, 50, false, Knn.DistanceCheck.Regular);
		checkError(data, 50, false, Knn.DistanceCheck.Efficient);
		System.out.println();
		System.out.println("----------------------------");
		System.out.println("Results for 10 folds: ");
		System.out.println("----------------------------");
		checkError(data, 10, false, Knn.DistanceCheck.Regular);
		checkError(data, 10, false, Knn.DistanceCheck.Efficient);
		System.out.println();
		System.out.println("----------------------------");
		System.out.println("Results for 5 folds: ");
		System.out.println("----------------------------");
		checkError(data, 5, false, Knn.DistanceCheck.Regular);
		checkError(data, 5, false, Knn.DistanceCheck.Efficient);
		System.out.println();
		System.out.println("----------------------------");
		System.out.println("Results for 3 folds: ");
		System.out.println("----------------------------");
		checkError(data, 3, false, Knn.DistanceCheck.Regular);
		checkError(data, 3, false, Knn.DistanceCheck.Efficient);
		System.out.println();
		System.out.println("----------------------------");
		System.out.println("Results for scaled dataset: ");
		System.out.println("----------------------------");
		checkError(scaledData, 10, true, Knn.DistanceCheck.Regular);
		System.out.println();
		System.out.println("----------------------------");
		System.out.println("Results for 159 folds: ");
		System.out.println("----------------------------");
		checkError(scaledData, scaledData.size(), false, Knn.DistanceCheck.Regular);
		checkError(scaledData, scaledData.size(), false, Knn.DistanceCheck.Efficient);
		System.out.println();
		System.out.println("----------------------------");
		System.out.println("Results for 50 folds: ");
		System.out.println("----------------------------");
		checkError(scaledData, 50, false, Knn.DistanceCheck.Regular);
		checkError(scaledData, 50, false, Knn.DistanceCheck.Efficient);
		System.out.println();
		System.out.println("----------------------------");
		System.out.println("Results for 10 folds: ");
		System.out.println("----------------------------");
		checkError(scaledData, 10, false, Knn.DistanceCheck.Regular);
		checkError(scaledData, 10, false, Knn.DistanceCheck.Efficient);
		System.out.println();
		System.out.println("----------------------------");
		System.out.println("Results for 5 folds: ");
		System.out.println("----------------------------");
		checkError(scaledData, 5, false, Knn.DistanceCheck.Regular);
		checkError(scaledData, 5, false, Knn.DistanceCheck.Efficient);
		System.out.println();
		System.out.println("----------------------------");
		System.out.println("Results for 3 folds: ");
		System.out.println("----------------------------");
		checkError(scaledData, 3, false, Knn.DistanceCheck.Regular);
		checkError(scaledData, 3, false, Knn.DistanceCheck.Efficient);



	}

	private static void checkError(Instances data, int folds, boolean changeGlobals, Knn.DistanceCheck distanceCheck) throws Exception{
		Knn knn = new Knn();
		long sumOfTime = 0;
		int maxK = 0;
		int maxW = 0;
		int maxLp = 0;
		double minError = Double.POSITIVE_INFINITY;

		if(changeGlobals) {
			for (int k = 1; k <= 20; k++) {
				for (int w = 0; w <= 1; w++) {
					for (int lp = 1; lp <= 3; lp++) {
						knn.buildClassifier(data, k, lp, folds, distanceCheck, getWeight(w));
						double error = knn.crossValidationError(data, folds);
						if(error < minError) {
							maxK = k;
							maxW = w;
							maxLp = lp;
							minError = error;
						}
					}
					knn.buildClassifier(data, k, DistanceCalculator.INFINITY, folds, distanceCheck, getWeight(w));
					double error = knn.crossValidationError(data, folds);
					if(error < minError) {
						maxK = k;
						maxW = w;
						maxLp = DistanceCalculator.INFINITY;
						minError = error;
					}
				}
			}
			bestK = maxK;
			bestLp = maxLp;
			bestW = maxW;
			System.out.println("Cross validation error with K = " + maxK + ", lp = " + InfinityToText(maxLp) + ", majority function = " + getWeight(maxW) +" for auto_price data is: " + minError);
		}

		else {
			knn.buildClassifier(data, bestK, bestLp, folds, distanceCheck, getWeight(bestW));
			long start = System.nanoTime();
			double error = knn.crossValidationError(data, folds);
			long end = System.nanoTime();
			sumOfTime += (end - start) / folds;
			System.out.println("Cross validation error of "+ distanceCheck + " knn on auto_price dataset is "+ error +
					" and the average elapsed time is " + sumOfTime + "\n" +
					"The total elapsed time is: " + sumOfTime * folds);
		}
	}

	private static String InfinityToText (int num) {
		return num == DistanceCalculator.INFINITY ? "Infinity" : (num + "");
	}


	private static Knn.Weights getWeight(int num) {
		return num == 0 ? Knn.Weights.Uniform : Knn.Weights.Weighted;
	}

}
