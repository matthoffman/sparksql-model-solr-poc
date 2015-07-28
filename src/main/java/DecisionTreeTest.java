/**
 * Created by ganeshkumar on 7/17/15.
 */

import java.io.IOException;
import java.lang.annotation.ElementType;
import java.util.HashMap;

import com.lucidworks.spark.SolrRDD;
import com.lucidworks.spark.SolrSupport;
import org.apache.solr.client.solrj.SolrServerException;
import org.apache.solr.common.SolrDocument;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.api.java.*;
import org.apache.spark.SparkContext;
import org.apache.spark.sql.types.*;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.SQLContext;
import scala.Tuple2;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.SparkConf;
import java.util.*;
import org.apache.solr.client.solrj.SolrQuery;
import org.apache.solr.common.SolrInputDocument;
import org.apache.*;
import org.apache.solr.client.solrj.impl.HttpSolrServer;
import org.apache.solr.client.solrj.response.QueryResponse;
import org.apache.solr.common.SolrDocumentList;
import scala.collection.mutable.ArrayBuffer;

import java.util.Map.Entry;


public class DecisionTreeTest {

    public static void main(String args[]) throws IOException, SolrServerException {
        SparkConf sparkConf = new SparkConf().setAppName("JavaDecisionTree");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);
        SQLContext sqlCtx = new org.apache.spark.sql.SQLContext(sc);
        // Load and parse the data file.
        String datapath = "data/mllib/sample_libsvm_data.txt";

        JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(sc.sc(), datapath).toJavaRDD();
        // Split the data into training and test sets (30% held out for testing)
        JavaRDD<LabeledPoint>[] splits = data.randomSplit(new double[]{0.7, 0.3});
        JavaRDD<LabeledPoint> trainingData = splits[0];
        JavaRDD<LabeledPoint> testData = splits[1];
        // Set parameters.
        //  Empty categoricalFeaturesInfo indicates all features are continuous.
        Integer numClasses = 2;
        Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();
        String impurity = "gini";
        Integer maxDepth = 5;
        Integer maxBins = 32;

        // Train a DecisionTree model for classification.
        final DecisionTreeModel model = DecisionTree.trainClassifier(trainingData, numClasses,
                categoricalFeaturesInfo, impurity, maxDepth, maxBins);

        // Evaluate model on test instances and compute test error
        JavaPairRDD<Double, Double> predictionAndLabel =
                testData.mapToPair(new PairFunction<LabeledPoint, Double, Double>() {
                    @Override
                    public Tuple2<Double, Double> call(LabeledPoint p) {
                        return new Tuple2<Double, Double>(model.predict(p.features()), p.label());
                    }
                });
        Double testErr =
                1.0 * predictionAndLabel.filter(new Function<Tuple2<Double, Double>, Boolean>() {
                    @Override
                    public Boolean call(Tuple2<Double, Double> pl) {
                        return !pl._1().equals(pl._2());
                    }
                }).count() / testData.count();
        System.out.println("Test Error: " + testErr);
        System.out.println("Learned classification tree model:\n" + model.toDebugString());

        // Save and load model
        model.save(sc.sc(), "myModelPathDT");
        // Get dataframes of the model to send it to Solr
        DataFrame df = sqlCtx.load("myModelPathDT/data/");
        DataFrame df1 = sqlCtx.jsonFile("myModelPathDT/metadata/");
        DataFrameSolr dfs = new DataFrameSolr("http://localhost:8888/solr/lucidworks");
        dfs.sendToSolr(df, df1);
        DataFrame pdf = dfs.readParquetDF( sc, sqlCtx);
        DataFrame mdf = dfs.readMetadataDF(sc, sqlCtx);
        System.out.println("Print Reconstructed Schema"+ pdf.schema().treeString());
        System.out.println("Print Original Schema"+ df.schema().treeString());

        System.out.println("Print Reconstructed metadata schema"+ mdf.schema().treeString());
        System.out.println("Print Original Schema"+ df1.schema().treeString());

        System.out.println("Print reconstructed metadata frame" + mdf.showString(10));
        System.out.println("Print original metadata frame" + df1.showString(10));

        mdf.toJSON().saveAsTextFile("test.parquet1/metadata");

        pdf.saveAsParquetFile("test.parquet1/data");
        final DecisionTreeModel sameModel1 = DecisionTreeModel.load(sc.sc(), "test.parquet1");
        JavaPairRDD<Double, Double> predictionAndLabel1 =
                testData.mapToPair(new PairFunction<LabeledPoint, Double, Double>() {
                    @Override
                    public Tuple2<Double, Double> call(LabeledPoint p) {
                        return new Tuple2<Double, Double>(sameModel1.predict(p.features()), p.label());
                    }
                });
        Double testErr1 =
                1.0 * predictionAndLabel1.filter(new Function<Tuple2<Double, Double>, Boolean>() {
                    @Override
                    public Boolean call(Tuple2<Double, Double> pl) {
                        return !pl._1().equals(pl._2());
                    }
                }).count() / testData.count();
        System.out.println("Test Error: " + testErr1);
        System.out.println("Learned classification tree model:\n" + sameModel1.toDebugString());

    }



}
