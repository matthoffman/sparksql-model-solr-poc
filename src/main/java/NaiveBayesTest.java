/**
 * Created by ganeshkumar on 7/23/15.
 */
import org.apache.solr.client.solrj.SolrServerException;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import scala.Tuple2;

import java.io.IOException;

public class NaiveBayesTest {
    public static void main(String args[]) throws IOException, SolrServerException {
        SparkConf sparkConf = new SparkConf().setAppName("JavaDecisionTree");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);
        SQLContext sqlCtx = new org.apache.spark.sql.SQLContext(sc);
        String datapath ="data/mllib/sample_naive_bayes_data.txt";
        JavaRDD<LabeledPoint> data = MLUtils.loadLabeledData(sc.sc(), datapath).toJavaRDD();
        JavaRDD<LabeledPoint>[] splits = data.randomSplit(new double[]{0.7, 0.3});
        JavaRDD<LabeledPoint> training = splits[0]; // training set
        JavaRDD<LabeledPoint> test = splits[1]; // test set

        final NaiveBayesModel model = NaiveBayes.train(training.rdd(), 1.0);

        JavaPairRDD<Double, Double> predictionAndLabel =
                test.mapToPair(new PairFunction<LabeledPoint, Double, Double>() {
                    @Override
                    public Tuple2<Double, Double> call(LabeledPoint p) {
                        return new Tuple2<Double, Double>(model.predict(p.features()), p.label());
                    }
                });
        double accuracy = predictionAndLabel.filter(new Function<Tuple2<Double, Double>, Boolean>() {
            @Override public Boolean call(Tuple2<Double, Double> pl) {
                return pl._1().equals(pl._2());
            }
        }).count() / (double) test.count();

// Save and load model

        model.save(sc.sc(), "myModelPathNB");
        NaiveBayesModel sameModel = NaiveBayesModel.load(sc.sc(), "myModelPathNB");

        DataFrame df = sqlCtx.load("myModelPathNB/data/");
        org.apache.spark.sql.types.StructField sftest = new org.apache.spark.sql.types.StructField("test", DataTypes.StringType, true, Metadata.empty());
        DataFrameSolr dfs = new DataFrameSolr("http://localhost:8888/solr/lucidworks");
        DataFrame df1 = sqlCtx.jsonFile("myModelPathNB/metadata/");
        System.out.println("GK Original Schema"+ df.schema().treeString());
        dfs.sendToSolr(df, df1);
        DataFrame pdf = dfs.readParquetDF( sc, sqlCtx);
        DataFrame mdf = dfs.readMetadataDF(sc, sqlCtx);
        System.out.println("Print Reconstructed Schema"+ pdf.schema().treeString());
        System.out.println("Print Original Schema"+ df.schema().treeString());

        System.out.println("Print Reconstructed Metadata Schema "+ mdf.schema().treeString());
        System.out.println("Print Original Schema"+ df1.schema().treeString());

        System.out.println("Print reconstructed metadata frame" + mdf.showString(10));
        System.out.println("Print original metadata frame" + df1.showString(10));

        System.out.println("Print reconstructed parquet data frame" + pdf.showString(10));
        System.out.println("Print original parquet data frame" + df.showString(10));
        mdf.toJSON().saveAsTextFile("test.parquet2/metadata");

        pdf.saveAsParquetFile("test.parquet2/data");

        final NaiveBayesModel model1 = NaiveBayesModel.load(sc.sc(),"test.parquet2" );

        JavaPairRDD<Double, Double> predictionAndLabel1 =
                test.mapToPair(new PairFunction<LabeledPoint, Double, Double>() {
                    @Override
                    public Tuple2<Double, Double> call(LabeledPoint p) {
                        return new Tuple2<Double, Double>(model1.predict(p.features()), p.label());
                    }
                });
        double accuracy1 = predictionAndLabel1.filter(new Function<Tuple2<Double, Double>, Boolean>() {
            @Override public Boolean call(Tuple2<Double, Double> pl) {
                return pl._1().equals(pl._2());
            }
        }).count() / (double) test.count();
        System.out.println("Original: "+accuracy);
        System.out.println("Reconstructed: "+accuracy1);
    }
}
