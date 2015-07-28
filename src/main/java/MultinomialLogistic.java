/**
 * Created by ganeshkumar on 7/9/15.
 */

import org.apache.solr.client.solrj.SolrQuery;
import org.apache.solr.common.SolrInputDocument;

import org.apache.spark.mllib.linalg.*;

import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import scala.Tuple2;

import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.sql.*;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import com.lucidworks.spark.*;
import org.apache.solr.common.SolrDocument;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Matrices;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.distributed.RowMatrix;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;

import java.util.*;


public class MultinomialLogistic {
    private static class FieldDescription {
        public String constraint;
        public String type;
        public String name;
    }

    public static void main(String[] args) throws Exception {
        SparkConf conf = new SparkConf().setAppName("SVM Classifier Example");
        SparkContext sc = new SparkContext(conf);
        String path = "data/mllib/sample_libsvm_data.txt";

        JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(sc, path).toJavaRDD();
        SQLContext sqlCtx = new org.apache.spark.sql.SQLContext(sc);
        // Split initial RDD into two... [60% training data, 40% testing data].
        JavaRDD<LabeledPoint>[] splits = data.randomSplit(new double[] {0.6, 0.4}, 11L);
        JavaRDD<LabeledPoint> training = splits[0].cache();
        JavaRDD<LabeledPoint> test = splits[1];
        final LogisticRegressionModel model = new LogisticRegressionWithLBFGS()
                .setNumClasses(2)
                .run(training.rdd());
        JavaRDD<Tuple2<Object, Object>> predictionAndLabels = test.map(
                new Function<LabeledPoint, Tuple2<Object, Object>>() {
                    public Tuple2<Object, Object> call(LabeledPoint p) {
                        Double prediction = model.predict(p.features());
                        return new Tuple2<Object, Object>(prediction, p.label());
                    }
                }
        );

        // Get evaluation metrics.
        MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());
        double precision = metrics.precision();
        System.out.println("Original Precision = " + precision);

        // Save and load model

        model.save(sc, "myTestpath");
        // Read the parquet file and metadata. Metadata is in Json form by default.
        DataFrame df = sqlCtx.load("myTestpath/data/");
        DataFrame df1 = sqlCtx.jsonFile("myTestpath/metadata/");
        //Reload

        JavaSparkContext jsc = new JavaSparkContext(sc);
        DataFrameSolr dfs = new DataFrameSolr("http://localhost:8888/solr/lucidworks");
        dfs.sendToSolr(df, df1);
        DataFrame pdf = dfs.readParquetDF( jsc, sqlCtx);
        DataFrame mdf = dfs.readMetadataDF(jsc, sqlCtx);
        System.out.println("Print Reconstructed Schema"+ pdf.schema().treeString());
        System.out.println("Print Original Schema"+ df.schema().treeString());

        System.out.println("Print Reconstructed Metadata Schema"+ mdf.schema().treeString());
        System.out.println("Print Original Schema"+ df1.schema().treeString());

        System.out.println("Print reconstructed metadata frame" + mdf.showString(10));
        System.out.println("Print original metadata frame" + df1.showString(10));

        mdf.toJSON().saveAsTextFile("test.parquet/metadata");

        pdf.saveAsParquetFile("test.parquet/data");

        final LogisticRegressionModel sameModel = LogisticRegressionModel.load(sc, "test.parquet");
        // Do prediction
        JavaRDD<Tuple2<Object, Object>> predictionAndLabels1= test.map(
                new Function<LabeledPoint, Tuple2<Object, Object>>() {
                    public Tuple2<Object, Object> call(LabeledPoint p) {
                        Double prediction = sameModel.predict(p.features());
                        return new Tuple2<Object, Object>(prediction, p.label());
                    }
                }
        );

        // Get evaluation metrics.
        MulticlassMetrics metrics1 = new MulticlassMetrics(predictionAndLabels.rdd());
        double precision1 = metrics1.precision();
        System.out.println("Reconstructed Precision = " + precision1);



    }
}