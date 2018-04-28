package HomeWork3;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import weka.core.Instances;

public class MainHW3 {

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
		checkError(data);
		System.out.println();
		System.out.println("----------------------------");
		System.out.println("Results for scaled dataset: ");
		System.out.println("----------------------------");
		checkError(scaledData);

	}

	private static void checkError(Instances data) throws Exception{
		Knn knn = new Knn();
		int maxK = 0;
		int maxW = 0;
		int maxLp = 0;
		double minError = Double.POSITIVE_INFINITY;
		for (int k = 1; k <= 20; k++) {
			for (int w = 0; w <= 1; w++) {
				for (int lp = 1; lp <= 3; lp++) {
					knn.buildClassifier(data, k, lp, 10, Knn.DistanceCheck.Regular, getWeight(w));
					double error = knn.crossValidationError(data, 10);
					if(error < minError) {
						maxK = k;
						maxW = w;
						maxLp = lp;
						minError = error;
					}
				}
				knn.buildClassifier(data, k, DistanceCalculator.INFINITY, 10, Knn.DistanceCheck.Regular, getWeight(w));
				double error = knn.crossValidationError(data, 10);
				if(error < minError) {
					maxK = k;
					maxW = w;
					maxLp = DistanceCalculator.INFINITY;
					minError = error;
				}
			}
		}

		System.out.println("Cross validation error with K = " + maxK + ", lp = " + maxLp + ", majority function = " + getWeight(maxW) +" for auto_price data is: " + minError);
	}

	private static Knn.Weights getWeight(int num) {
		return num == 0 ? Knn.Weights.Uniform : Knn.Weights.Weighted;
	}

}
