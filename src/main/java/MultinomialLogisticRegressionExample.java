/**
 * Created by ganeshkumar on 7/9/15.
 */
import org.apache.avro.generic.GenericData;
import org.apache.solr.client.solrj.SolrQuery;
import org.apache.solr.common.SolrInputDocument;
import org.apache.solr.common.SolrInputField;
import org.apache.spark.mllib.linalg.*;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.streaming.api.java.JavaDStream;
import scala.Function1;
import scala.Tuple2;

import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.sql.*;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import parquet.format.*;
import parquet.hadoop.*;
import com.lucidworks.spark.*;
import org.apache.solr.common.SolrDocument;
import org.apache.solr.common.SolrInputDocument;
import org.apache.solr.client.solrj.SolrClient;

import java.util.*;
import java.util.Collection.*;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import org.apache.solr.*;

import parquet.Log;
import parquet.example.data.Group;
import parquet.hadoop.ParquetInputSplit;
import parquet.hadoop.example.ExampleInputFormat;
import parquet.schema.MessageTypeParser;
import parquet.schema.Type;
import parquet.schema.GroupType;

import java.io.IOException;
import java.lang.reflect.Field;
import java.io.IOException;


public class MultinomialLogisticRegressionExample {
    private static class FieldDescription {
        public String constraint;
        public String type;
        public String name;
    }

    public static void main(String[] args) throws Exception {
        SparkConf conf = new SparkConf().setAppName("SVM Classifier Example");
        SparkContext sc = new SparkContext(conf);
        //String path = "data/mllib/sample_libsvm_data.txt";
        String path = "data/gktest.txt";
        System.out.println("Here1");
        JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(sc, path).toJavaRDD();
        //JavaRDD data = MLUtils.loadVectors(sc, path).toJavaRDD();
        System.out.println("Here2");
        SQLContext sqlCtx = new org.apache.spark.sql.SQLContext(sc);
        // Split initial RDD into two... [60% training data, 40% testing data].
        JavaRDD<LabeledPoint>[] splits = data.randomSplit(new double[] {0.6, 0.4}, 11L);
        JavaRDD<LabeledPoint> training = splits[0].cache();
        JavaRDD<LabeledPoint> test = splits[1];
        System.out.println("Here3");
        // Run training algorithm to build the model.
/*        LogisticRegression lr = new LogisticRegression();
        lr.setMaxIter(10)
                .setRegParam(0.01);
        DataFrame training1 = sqlCtx.createDataFrame(training, LabeledPoint.class);
        DataFrame test1 = sqlCtx.createDataFrame(test, LabeledPoint.class);
        LogisticRegressionModel model1 = lr.fit(training1);*/
        final LogisticRegressionModel model = new LogisticRegressionWithLBFGS()
                .setNumClasses(2)
                .run(training.rdd());
        System.out.println("Here4");
        //DataFrame results = model1.transform(test1);
        // Compute raw scores on the test set.
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
        System.out.println("Precision = " + precision);
        //JavaSchemaRDD parquetFile = sqlContext.parquetFile("people.parquet");

        // Save and load model

        model.save(sc, "myTestpath");



        //DataFrame df = sqlCtx.parquetFile("/Users/ganeshkumar/Downloads/spark-1.3.1-bin-hadoop2.6/myTestpath/data/part-r-00001.parquet");
        DataFrame df = sqlCtx.load("/Users/ganeshkumar/Downloads/spark-1.3.1-bin-hadoop2.6/myTestpath/data/");
        DataFrame df1 = sqlCtx.jsonFile("/Users/ganeshkumar/Downloads/spark-1.3.1-bin-hadoop2.6/myTestpath/metadata/");
        System.out.println("GK Debug1" + df1.showString(3));
        System.out.println("GK Debug2" + df.showString(3));
        //String fileSchema = ((ParquetInputSplit)sc.getInputSplit()).getFileSchema();
        //df.select("weights","intercept","threshold").saveAsParquetFile("/Users/ganeshkumar/test.parquet/data");
        df1.select("class","numClasses","numFeatures","version").toJSON().saveAsTextFile("/Users/ganeshkumar/test.parquet/metadata");
        //.saveAsParquetFile("/Users/ganeshkumar/test.parquet/data");
        String message = model.toString();
        System.out.println("GK print message"+message);
        //final LogisticRegressionModel sameModel = LogisticRegressionModel.load(sc, "/Users/ganeshkumar/test.parquet/");
        System.out.println("GK DataFrame1 to Json" +
                df.toJavaRDD().collect());
        System.out.println("GK DataFrame2 to Json" +
                df1.toJavaRDD().collect());
        System.out.println("GK DataFrame1 to Schema" +
                df.schema());
        System.out.println("GK DataFrame2 to Schema" +
                df1.schema());
        StructType sch = df.schema();
        String[] s4 = (java.lang.String[]) df1.toJSON().collect();
        System.out.println("GK DF1 to json"+ df1.showString(3) + s4[0]);
        SolrInputField field1 = new SolrInputField("test");
        //SolrInputField field2 = new SolrInputField("id");
        Map<String,String> xyza = new HashMap<String, String>();
        //
        //xyza.put("id", field2);
        //xyza.put("test", "1");
        SolrInputDocument s = new SolrInputDocument();
        SolrSupport ss = new SolrSupport();

        java.util.List<Row> s5 = df.toJavaRDD().collect();
        Iterator<Row> s6 = s5.iterator();
        StructType s1 = df.schema();
        scala.collection.Iterator<StructField> s2 = s1.toIterator();
        int i = 0;
        Row s7 = null;
        //xyza.put("test_s","1");
        //
        //s.setField("id",12345);
        //s.setField("test_s", 1);
        while (s2.hasNext())
        {

            StructField s3 = s2.next();

            if (s6.hasNext())
            {
                s7 = s6.next();
            }

            System.out.println("GK Loop"+ s3 );
            System.out.println("GK Loop"+ s3.name() + s3.dataType() + s3.nullable() );
            List<Object> checkchild = new ArrayList<Object>();
            checkchild.add(s7.get(i));
            System.out.println("GK check size "+checkchild.size());
            if (checkchild.size() > 1)
            {
                SolrInputDocument schild = new SolrInputDocument();
                xyza.put(s3.name()+"_s", schild.get(0).toString());
                //schild.addField("feature1",schild.get(0));
            }
            else {
                System.out.println("GK check hashmap"+ s3.name() + s7.get(i).toString() );
                //xyza.put(s3.name()+"_s","1");
                xyza.put(s3.name()+"_s",s7.get(i).toString());
                //s.setField(s3.name(),s7.get(i).toString());
                //s.setField("", 1);
                //s7.get(i)s3.name()
            }
            i = i + 1;

        }
        System.out.println("GK Check Keyset & values"+ xyza.keySet() + xyza);
        s = ss.autoMapToSolrInputDoc(java.util.UUID.randomUUID().toString(),xyza.keySet(),xyza);
        Iterator it1 = xyza.entrySet().iterator();
        while (it1.hasNext()) {
            Map.Entry pair = (Map.Entry)it1.next();
            //System.out.println(pair.getKey() + " = " + pair.getValue());
            s.setField(pair.getKey().toString(),pair.getValue());
            it1.remove(); // avoids a ConcurrentModificationException

        }
       // s.setField("test_s", "1");
        System.out.println("GK Print SolrDocument" + s.getFieldNames() + s.getFieldValue("weights") + s.getFieldValue("intercept") + s.getFieldValue("threshold"));
        ArrayList<SolrInputDocument> a = new ArrayList<SolrInputDocument>();
        a.add(s);
        //JavaDRDD<SolrDocument> send = JavaSparkContext.parallelize(a);
        JavaSparkContext jsc = new JavaSparkContext(sc);

        JavaRDD<SolrInputDocument> send = jsc.parallelize(a);

        System.out.println("GK Final before send"+ send.collect());

        ss.indexDocs("localhost:9888","lucidworks5", 10, send);
        SolrRDD sr = new SolrRDD("localhost:9888","lucidworks5");
        // sq = new SolrQuery();
        //SolrQuery sq = sr.toQuery("&fl=weights_s,intercept_s,threshold_s&fq=id:54e71964-f64c-496a-b5b5-455b7837fe27");
        SolrQuery sq = sr.toQuery("id:54e71964-f64c-496a-b5b5-455b7837fe27");
        System.out.println("GK Print Solr Query"+ sq);
        //sq.setFilterQueries("id:54e71964-f64c-496a-b5b5-455b7837fe27");
        //sq.addField("weights_s");
        //sq.addField("intercept_s");
        //sq.addField("threshold_s");
        JavaRDD<SolrDocument> dfj = sr.queryShards(jsc, sq);
        final List<SolrDocument> str1 = dfj.collect();
        JavaRDD<Row> dfjr = dfj.map(new Function<SolrDocument, Row>() {
            @Override
            public Row call(SolrDocument solrDocument) throws Exception {
                ArrayList<Object> str = new ArrayList<Object>();
                //str.add(solrDocument.getFieldValue("weights_s"));
                str.add(Vectors.dense(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0));
                str.add(Double.parseDouble(solrDocument.getFieldValue("intercept_s").toString()));
                str.add(Double.parseDouble(solrDocument.getFieldValue("threshold_s").toString()));
                /*Map<String, Object> hm = solrDocument.getFieldValueMap();
                Iterator it2 = hm.entrySet().iterator();
                while (it2.hasNext()) {
                    Map.Entry pair = (Map.Entry)it2.next();
                    str.add(pair.getValue());
                    //s.setField(pair.getKey().toString(), pair.getValue());
                    it2.remove(); // avoids a ConcurrentModificationException

                }*/

                Row r = RowFactory.create(str.toArray());

                //r.apply(2);
                return r;
            }
        });
        System.out.println("GK Print Row" + dfjr.collect());
        System.out.println("GK Print Row Schema" + dfjr.collect().get(0).schema());
        DataFrame df3 = sqlCtx.createDataFrame(dfjr, sch);
        System.out.println("GK Print DF3 sch" + sch);
        System.out.println("GK Print DF3" + df3.showString(20));
        //sqlCtx.createDataFrame(dfjr, sch);
        System.out.println("GK Print Solr Query Output"+ dfj.collect());
        DataFrame df2 = sr.queryForRows(sqlCtx, "id:54e71964-f64c-496a-b5b5-455b7837fe27");

        System.out.println("GK Read DataFrame" + df2.showString(20));
        System.out.println("GK Read DataFrame Schema" + df2.schema());
        //df2.select("weights_s","intercept_s","threshold_s").toDF("weights", "intercept", "threshold").saveAsParquetFile("/Users/ganeshkumar/test.parquet/data");
        df3.select("weights", "intercept", "threshold").saveAsParquetFile("/Users/ganeshkumar/test.parquet/data");
        final LogisticRegressionModel sameModel = LogisticRegressionModel.load(sc, "/Users/ganeshkumar/test.parquet/");
        //Column ageCol = df.select("weights").col("age");
        //LogisticRegressionModel xyz = new LogisticRegressionModel(,df.select("intercept") );
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
        System.out.println("Precision = " + precision1);

        ArrayList<FieldDescription> fields = new ArrayList<FieldDescription>();
        List<String> elements = Arrays.asList(message.split("\n"));
        Iterator<String> it = elements.iterator();


}
}