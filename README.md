# sparksql-model-solr-poc
POC to store machine learning models in Solr

In most of the technologies part of hadoop ecosystem (such as hive, spark etc) , the recommended format to store machine learning models is parquet format (developed by ASF). This POC is an attempt to read, parse and store the parquet model in Solr and read it again for prediction.

Spark SQL already provides the mechanism to convert parquet file into a DataFrame.

Features:

1) Send spark dataframes to Solr
2) Read back the dataframes from Solr into Spark

Prerequisites:

Install spark. [Download link: http://www.apache.org/dyn/closer.cgi/spark/spark-1.4.1/spark-1.4.1.tgz]

Procedure:

1)	Clone the project into your local repo.

Git clone https://github.com/ganeshk7/sparksql-model-solr-poc.git

2)	There are three test machine learning algorithms tested with sample data. The classes are DecisionTreeTest.java, MultinomialLogistic.java and NaiveBayesTest.java

a)	All these three classes have hard-coded input file path. Change the variable “datapath” to point it to your own data file. Please note that the file has to be in a <LabelledPoint> format. (Not required in general. But to test the POC, we used this). An example of LabelledPoint format is given below

		0 1:1 2:2 3:3 4:4 5:5 6:6 7:7 8:8 9:9
1 1:8 2:7 3:6 4:4 5:5 6:6 7:1 8:2 9:3

0 and 1 are class labels. Totally number of features is 9 (labeled from 1 to 9).

For simplicity, you can use the sample data files provided by Spark as provided in the java file already.
b)	You need to have Solr up and running with at-least one core. The variable “dfs” has the solr connection details. Replace the same with the Solr port and collection name that you are running on.

3)	Build the jar: This will also download the necessary dependencies from the pom.xml

mvn clean install


4)	Run the Jar using the spark-submit command line. Example given below.

./bin/spark-submit –class <<ClassName>> --master local[4] –jars <<ExternalJars>> <<Jar containing your class>>

We need to include the external Jar “spark-solr-1.0-SNAPSHOT.jar” as well.

./bin/spark-submit --class DecisionTreeTest --master local[4] --jars /Users/ganeshkumar/spark-solr/target/spark-solr-1.0-SNAPSHOT.jar /Users/ganeshkumar/myCode/ML1/target/ml1-1.0.jar
 
./bin/spark-submit --class NaiveBayesTest --master local[4] --jars /Users/ganeshkumar/spark-solr/target/spark-solr-1.0-SNAPSHOT.jar /Users/ganeshkumar/myCode/ML1/target/ml1-1.0.jar
 
./bin/spark-submit --class MultinomialLogistic --master local[4] --jars /Users/ganeshkumar/spark-solr/target/spark-solr-1.0-SNAPSHOT.jar /Users/ganeshkumar/myCode/ML1/target/ml1-1.0.jar


Note: This works for the basic data type conversions and Vector type. However, coversion is yet to be implemented for Labelled Point, Local Matrix and Distributed Matrix.
