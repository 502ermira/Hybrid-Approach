package Project6;


import weka.clusterers.ClusterEvaluation;
import weka.clusterers.FarthestFirst;
import weka.clusterers.SimpleKMeans;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.Standardize;

public class HybridApproach {
    
    public static void main(String[] args) throws Exception {
        // Load dataset
        DataSource source = new DataSource("C:\\Users\\Admin\\Desktop\\fina.arff");
        Instances data = source.getDataSet();

        // Remove class attribute
 /*       Remove remove = new Remove();
        remove.setAttributeIndices("" + (data.classIndex() + 1));
        remove.setInputFormat(data);
        Instances dataWithoutClass = Filter.useFilter(data, remove);
       System.out.println("1shi: "+dataWithoutClass.numAttributes()); */
       
        // Apply clustering to entire dataset
        SimpleKMeans ff = new SimpleKMeans();
        ff.setNumClusters(1000);
        ff.buildClusterer(data);
        Instances clusteredData = new Instances(data, data.numInstances());
 
        clusteredData.setClassIndex(clusteredData.numAttributes() - 1);
        for (int i = 0; i < data.numInstances(); i++) {
            Instance inst = data.instance(i);
            int clusterIndex = ff.clusterInstance(inst);
            clusteredData.add(ff.getClusterCentroids().instance(clusterIndex));
            
        }

        data.setClassIndex(data.numAttributes() - 1);
        // Split data into training and test sets
        int trainSize = (int) Math.round(data.numInstances() * 0.8);
        int testSize = data.numInstances() - trainSize;
        Instances trainData = new Instances(data, 0, trainSize);
        Instances testData = new Instances(data, trainSize, testSize);

        // Train and evaluate the Naive Bayes classifier
        NaiveBayes nb = new NaiveBayes();
        nb.buildClassifier(clusteredData);
        Evaluation eval = new Evaluation(testData);
        eval.evaluateModel(nb, testData);
        
     
        NaiveBayes nb1 = new NaiveBayes();
        nb1.buildClassifier(trainData);
        Evaluation eval1 = new Evaluation(testData);
        eval1.evaluateModel(nb1, testData);
        

        // Output evaluation results
        System.out.println("Performance of the algorithm on the clustered data");
       System.out.println(eval.toSummaryString());
       System.out.println("Performance of the algorithm on the original dataset");
       System.out.println(eval1.toSummaryString());
    }
}