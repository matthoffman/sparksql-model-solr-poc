import org.apache.solr.client.solrj.SolrQuery;
import org.apache.solr.client.solrj.SolrServerException;
import org.apache.solr.client.solrj.impl.HttpSolrServer;
import org.apache.solr.client.solrj.response.QueryResponse;
import org.apache.solr.common.SolrDocument;
import org.apache.solr.common.SolrDocumentList;
import org.apache.solr.common.SolrInputDocument;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.*;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.types.*;
import scala.Tuple2;
import scala.collection.mutable.ArrayBuffer;

import java.io.IOException;
import java.util.*;

/**
 * Created by ganeshkumar on 7/23/15.
 */
public class DataFrameSolr {

    private String solrConnectString = "";

    public DataFrameSolr(String solrConnectString ) {
        this.setsolrConnectString(solrConnectString);
    }

    public String getsolrConnectString()
    {
        return this.solrConnectString;
    }

    public  void setsolrConnectString(String s)
    {
        this.solrConnectString = s;
    }

    public void sendToSolr(DataFrame parquetDF, DataFrame metadataDF) throws IOException, SolrServerException {
        HttpSolrServer solr = new HttpSolrServer( this.solrConnectString);
        // Get max version of the model stored and increment it by 1
        Integer version = getMaxModelVersion(this.solrConnectString) + 1;
        solr.add( convertToSolrDocuments(parquetDF, "parquet", version) );
        solr.add( convertToSolrDocuments(metadataDF, "metadata", version) );
        solr.commit();
    }

    public DataFrame readParquetDF( JavaSparkContext sc, SQLContext sqlCtx) throws IOException, SolrServerException {
        HttpSolrServer solr = new HttpSolrServer( this.solrConnectString);
        Integer version = getMaxModelVersion(this.solrConnectString);
        SolrQuery q = new SolrQuery("root_s:root AND category:schema AND content_type:parquet AND __lwversion_i:"+version);
        QueryResponse rsp = solr.query( q );
        SolrDocumentList docs = rsp.getResults();

        StructType st = readSchema(docs, solr);
        // Print it the reconstructed version and the original version

        // Get the latest version of the model - just the parquet data
        SolrQuery sq = new SolrQuery("root_s:root AND category:data AND content_type:parquet AND __lwversion_i:"+version);

        SolrDocumentList sld = solr.query(sq).getResults();

        JavaRDD<Row> dfjr = sc.parallelize(readData(sld, solr, st));

        // Create a data frame based on the RDD of rows and the schema s1

        DataFrame df3 = sqlCtx.createDataFrame(dfjr, st);
        return df3;
    }

    public DataFrame readMetadataDF(JavaSparkContext sc, SQLContext sqlCtx) throws IOException, SolrServerException {
        HttpSolrServer solr = new HttpSolrServer( this.solrConnectString);
        Integer version = getMaxModelVersion(this.solrConnectString);
        SolrQuery qm = new SolrQuery("root_s:root AND category:schema AND content_type:metadata AND __lwversion_i:"+version);
        QueryResponse rspm = solr.query( qm );
        SolrDocumentList docsm = rspm.getResults();

        StructType stm = readSchema(docsm, solr);

        // Get the metadata data
        SolrQuery sqm = new SolrQuery("root_s:root AND category:data AND content_type:metadata AND __lwversion_i:"+version);

        SolrDocumentList sldm = solr.query(sqm).getResults();

        JavaRDD<Row> dfjrm = sc.parallelize(readData(sldm, solr, stm));

        // Create a data frame based on the RDD of rows and the schema s1

        DataFrame dfm = sqlCtx.createDataFrame(dfjrm, stm);
        return dfm;
    }

    private static Integer getMaxModelVersion(String solrConnectString)
    {
        HttpSolrServer solr = new HttpSolrServer( solrConnectString);
        SolrQuery sqversionMax = new SolrQuery("root_s:root");
        sqversionMax.setSort(new SolrQuery.SortClause("__lwversion_i", "desc"));
        sqversionMax.setRows(1);
        sqversionMax.setFields("__lwversion_i");
        Integer version = 0;
        try
        {
            version = (Integer) solr.query(sqversionMax).getResults().get(0).get("__lwversion_i");
        }
        catch (Exception E)
        {
            System.out.println("No previous version found. Defaulting version to 0");
        }
        return version;
    }

    public static ArrayList<SolrInputDocument> convertToSolrDocuments(DataFrame df, String cType, Integer version)
    {
        SolrInputDocument s = new SolrInputDocument();
        StructType styp = df.schema();
        String id = java.util.UUID.randomUUID().toString();
        s.addField("id",id);
        s.addField("root_s", "root");
        s.addField("__lwversion_i", version);
        int level = 0;
        s.addField("content_type", cType);
        s.addField("category", "schema");
        ArrayList<SolrInputDocument> solrDocumentList = new ArrayList<SolrInputDocument>();
        String idData = null;
        for (int i =0; i < df.count(); i++)
        {
            solrDocumentList.add(new SolrInputDocument());
            idData = java.util.UUID.randomUUID().toString();
            solrDocumentList.get(i).addField("id",idData);
            solrDocumentList.get(i).addField("__lwversion_i", version);
            solrDocumentList.get(i).addField("root_s", "root");
            solrDocumentList.get(i).addField("content_type", cType);
            solrDocumentList.get(i).addField("category", "data");
        }
        recurseWrite(styp, s, level, solrDocumentList, df.collectAsList(), 0);
        ArrayList<SolrInputDocument> a = new ArrayList<SolrInputDocument>();

        a.add(s);
        for (int i =0; i< solrDocumentList.size(); i++)
        {
            a.add(solrDocumentList.get(i));
        }
        return a;
    }

    public static StructType readSchema(SolrDocumentList docs, HttpSolrServer solr) throws IOException, SolrServerException {
        List<StructField> fields = new ArrayList<StructField>();
        StructType st= (StructType) recurseRead(docs.get(0), solr, fields).dataType();
        return st;
    }

    public static ArrayList<Row> readData(SolrDocumentList docs, HttpSolrServer solr, StructType st) throws IOException, SolrServerException {
        ArrayList<Row> finalstr = new ArrayList<Row>();
        for(int k=0; k<docs.size(); k++)
        {
            ArrayList<Object> str = new ArrayList<Object>();
            finalstr.add(recurseDataRead(docs.get(k), solr, str, st));
        }
        return finalstr;
    }

    public static void recurseWrite(StructType st, SolrInputDocument s, int l, ArrayList<SolrInputDocument> solrDocumentList, List<org.apache.spark.sql.Row> df, int counter)
    {
        scala.collection.Iterator x = st.iterator();
        int linkCount = 0;
        while (x.hasNext())
        {
            StructField sf = (StructField) x.next();
            if (sf.dataType().typeName().toString().toLowerCase().equals("struct"))
            {
                linkCount = linkCount + 1;
                SolrInputDocument sc = new SolrInputDocument();
                ArrayList<SolrInputDocument> solrDocumentList1 = new ArrayList<SolrInputDocument>();
                String id = java.util.UUID.randomUUID().toString();
                sc.addField("id",id);
                s.addField("links"+linkCount +"_s", id);
                List<org.apache.spark.sql.Row> df1 = new ArrayList<org.apache.spark.sql.Row>();
                l = l + 1;

                sc.addField("child_doc_name_s",sf.name());
                sc.addField("category","schema");
                for (int inner=0; inner<solrDocumentList.size(); inner++) {
                    solrDocumentList1.add(new SolrInputDocument());
                }
                for (int inner=0; inner<solrDocumentList.size(); inner++) {
                    String idChild = java.util.UUID.randomUUID().toString();
                    solrDocumentList1.get(inner).addField("id",idChild );
                    solrDocumentList.get(inner).addField("links"+linkCount +"_s",idChild );

                    solrDocumentList1.get(inner).addField("child_doc_name_s",sf.name());
                    solrDocumentList1.get(inner).addField("category","data");
                    df1.add((org.apache.spark.sql.Row) df.get(inner).get(counter));
                }
                recurseWrite((StructType) sf.dataType(), sc, l, solrDocumentList1, df1, 0);
                s.addChildDocument(sc);
                for (int inner=0; inner<solrDocumentList.size(); inner++)
                {
                    solrDocumentList.get(inner).addChildDocument(solrDocumentList1.get(inner));
                }
            }
            else
            {

                if (!sf.dataType().typeName().toLowerCase().equals("array"))
                {
                    s.addField(sf.name() + "_s", sf.dataType().typeName());
                }
                else
                {
                    //s.addField(sf.name() + "_s", sf.dataType().typeName() + ":" + ((ArrayType) sf.dataType()).elementType().typeName());
                    s.addField(sf.name() + "_s", getArraySchema(sf.dataType()));
                }

                for (int inner=0; inner<solrDocumentList.size(); inner++)
                {
                    if (df.get(inner) != null) {
                        if (!sf.dataType().isPrimitive() && sf.dataType().typeName().equals("array"))
                        {
                            //Tuple2<String,Object> suffixandvalue = getArrayToString(sf.dataType(), df.get(inner).get(counter));
                            solrDocumentList.get(inner).addField(sf.name() + "_s", getArrayToString(sf.dataType(), df.get(inner).get(counter)));
                        }
                        else
                        {
                            if (df.get(inner).get(counter) != null)
                            {
                                solrDocumentList.get(inner).addField(sf.name() + "_s", df.get(inner).get(counter));
                            }
                            else
                            {
                                solrDocumentList.get(inner).addField(sf.name() + "_s", "0");
                            }
                        }
                    }

                }
            }
            counter = counter + 1;
        }
    }

    public static Row recurseDataRead(SolrDocument doc, HttpSolrServer solr, ArrayList<Object> x, StructType st)
    {
        Boolean recurse = true;

        ArrayList<Object> str = new ArrayList<Object>();
        String id1 = doc.get("id").toString();
        //ArrayList<String> ids = (ArrayList<String>) doc.get("links");


        Map<String, Object> x1 = doc.getFieldValueMap();
        {
            Object[] x2 = x1.keySet().toArray();
            for (int i = 0; i < x2.length; i++) {
                if (x2[i].toString().startsWith("links"))
                {
                    String id = doc.get(x2[i].toString()).toString();
                    if (id != null)
                    {
                        SolrQuery q1 = new SolrQuery("id:" + id);
                        QueryResponse rsp1 = null;
                        try {
                            rsp1 = solr.query(q1);
                        } catch (Exception E) {
                            recurse = false;
                        }
                        if (recurse) {
                            //l = l + 1;
                            SolrDocumentList docs1 = rsp1.getResults();
                            ArrayList<Object> str1 = new ArrayList<Object>();
                            x.add(recurseDataRead(docs1.get(0), solr, str1, st));
                        }
                    }
                }
                if (x2[i].toString().substring(x2[i].toString().length() - 2, x2[i].toString().length()).equals("_s") && !x2[i].toString().equals("child_doc_name_s") && !x2[i].toString().equals("root_s")  && !x2[i].toString().startsWith("links")) {
                    String type = getFieldTypeMapping(st,x2[i].toString().substring(0,x2[i].toString().length()-2));
                    if (!type.equals(""))
                    {
                        //GK FieldTypeMappingbyte:short:integer:long:float:double:decimal:string:binary:boolean:timestamp:date:array:map:struct:string
                        if (type.equals("integer"))
                        {
                            x.add(convertToInteger(x1.get(x2[i]).toString()));
                        }
                        else if (type.equals("double"))
                        {
                            x.add(convertToDouble(x1.get(x2[i]).toString()));
                        }
                        else if (type.equals("float"))
                        {
                            x.add(convertToFloat(x1.get(x2[i]).toString()));
                        }
                        else if (type.equals("short"))
                        {
                            x.add(convertToShort(x1.get(x2[i]).toString()));
                        }
                        else if (type.equals("long"))
                        {
                            x.add(convertToLong(x1.get(x2[i]).toString()));
                        }
                        else if (type.equals("decimal"))
                        {

                        }
                        else if (type.equals("boolean"))
                        {
                            x.add(convertToBoolean(x1.get(x2[i]).toString()));
                        }
                        else if (type.equals("timestamp"))
                        {

                        }
                        else if (type.equals("date"))
                        {

                        }
                        else if (type.equals("vecto"))
                        {
                            x.add(convertToVector(x1.get(x2[i]).toString()));
                        }
                        else if (type.contains(":"))
                        {
                           //List<Object> debug = Arrays.asList(getArrayFromString(type, x1.get(x2[i]).toString(), 0, new ArrayList<Object[]>()));
                            x.add(getArrayFromString(type, x1.get(x2[i]).toString(), 0, new ArrayList<Object[]>()));
                        }
                        else
                        {
                            x.add(x1.get(x2[i]));
                        }
                    }
                    else {
                        x.add(x1.get(x2[i]));
                    }
                }
            }
        }
        if (x.size()>0) {
            Object[] array = new Object[x.size()];

            x.toArray(array);

            return RowFactory.create(array);
        }
        return null;
    }

    public static StructField recurseRead(SolrDocument doc, HttpSolrServer solr, List<StructField> fldr) throws IOException, SolrServerException
    {
        Boolean recurse = true;
        List<StructField> fld = new ArrayList<StructField>();
        String id1 = doc.get("id").toString();

        String finalName = null;
        for (Map.Entry<String, Object> field : doc.entrySet())
        {
            String name = field.getKey();
            Object value = field.getValue();
            if (name.startsWith("links"))
            {
                //ArrayList<String> ids = (ArrayList<String>) doc.get(name);
                String id = doc.get(name).toString();
                if (id != null)
                {

                    SolrQuery q1 = new SolrQuery("id:" + id);
                    QueryResponse rsp1 = null;
                    try
                    {
                        rsp1 = solr.query(q1);
                    } catch (Exception E)
                    {
                        recurse = false;
                    }
                    if (recurse)
                    {
                        SolrDocumentList docs1 = rsp1.getResults();
                        List<StructField> fld1 = new ArrayList<StructField>();
                        fldr.add(recurseRead(docs1.get(0), solr, fld1));
                    }

                }
            }
            if (name.substring(name.length()-2,name.length()).equals("_s")  && !name.equals("root_s") && !name.startsWith("links")) {
                if (name.substring(0, name.length() - 2).equals("child_doc_name")) {
                    finalName = field.getValue().toString();
                } else
                {
                    fldr.add(new StructField(name.substring(0, name.length() - 2), getsqlDataType(field.getValue().toString()), true, Metadata.empty()));
                }

            }
            System.out.println("\t" + name + "=" + value);

        }
        StructField[] farr = new StructField[fldr.size()];
        farr = fldr.toArray(farr);
        StructType st2 = new StructType(farr);
        if (finalName == null)
        {
            finalName = "root";
        }
        return new StructField(finalName, st2, true,  Metadata.empty());
    }


    public static Object getmllibDataType(String s)
    {
        if (s.toLowerCase().equals("vecto"))
        {
            return Vector.class.toString();
        }
        return String.class;
    }
    public static DataType getsqlDataType(String s)
    {

        if (s.toLowerCase().equals("double"))
        {
            return DataTypes.DoubleType;
        }
        if (s.toLowerCase().equals("byte"))
        {
            return DataTypes.ByteType;
        }
        if (s.toLowerCase().equals("short"))
        {
            return DataTypes.ShortType;
        }
        if (((s.toLowerCase().equals("int")) || (s.toLowerCase().equals("integer"))))
        {
            return DataTypes.IntegerType;
        }
        if (s.toLowerCase().equals("long"))
        {
            return DataTypes.LongType;
        }
        if (s.toLowerCase().equals("String"))
        {
            return DataTypes.StringType;
        }
        if (s.toLowerCase().equals("boolean"))
        {
            return DataTypes.BooleanType;
        }
        if (s.toLowerCase().equals("timestamp"))
        {
            return DataTypes.TimestampType;
        }
        if (s.toLowerCase().equals("date"))
        {
            return DataTypes.DateType;
        }
        if (s.toLowerCase().equals("vecto"))
        {
            return new VectorUDT();
        }
        if (s.contains(":") && s.split(":")[0].toLowerCase().equals("array"))
        {
            return getArrayTypeRecurse(s,0);
        }
        return DataTypes.StringType;
    }

    public static DataType getArrayTypeRecurse(String s, int fromIdx)
    {
        if (s.contains(":") && s.split(":")[1].toLowerCase().equals("array"))
        {
            fromIdx = s.indexOf(":", fromIdx);
            s = s.substring(fromIdx+1, s.length());

            return DataTypes.createArrayType(getArrayTypeRecurse(s,fromIdx));
        }
        return DataTypes.createArrayType(getsqlDataType(s.split(":")[1]));
    }

    public static Object[] getArrayFromString(String type, String s, int fromIdx, ArrayList<Object[]> ret)
    {
        if(type.contains(":") && type.split(":")[1].equals("array"))
        {
            fromIdx = type.indexOf(":", fromIdx);
            type = type.substring(fromIdx+1, type.length());
            String[] items = s.replaceFirst("\\[", "").substring(0,s.replaceFirst("\\[", "").lastIndexOf("]")).split("\\],");
            ArrayList<Object[]> ret1 = new ArrayList<Object[]>();
            for (int i=0; i<items.length; i++)
            {
                if (i == items.length -1 )
                {
                    ret1.add(getArrayFromString(type, items[i], fromIdx, ret1));
                }
                else
                {
                    ret1.add(getArrayFromString(type, items[i] + "]", fromIdx, ret1));
                }
            }
            ret.add(ret1.toArray());
            return ret1.toArray();

        }
        String[] items = s.replaceFirst("\\[", "").substring(0,s.replaceFirst("\\[", "").lastIndexOf("]")).split(",");
        if (type.split(":")[1].equals("integer"))
        {

            //x.add(convertToIntegerArray(items));
            return convertToIntegerArray(items);
        }
        else if(type.split(":")[1].equals("double"))
        {
            //x.add(convertToDoubleArray(items));
            return convertToDoubleArray(items);
        }
        else if(type.split(":")[1].equals("float"))
        {
            //x.add(convertToFloatArray(items));
            return convertToFloatArray(items);
        }
        else if(type.split(":")[1].equals("short"))
        {
            //x.add(convertToShortArray(items));
            return convertToShortArray(items);
        }
        else if(type.split(":")[1].equals("long"))
        {
            //x.add(convertToLongArray(items));
            return convertToLongArray(items);
        }
        else
        {
            //x.add(items);
            return items;
        }
        //return new String[]{};
    }

    public static String getArraySchema(DataType dType)
    {
        if (((ArrayType) dType).elementType().typeName().equals("array"))
        {
            return dType.typeName() + ":" + getArraySchema(((ArrayType) dType).elementType());
        }
        else
        {
            return dType.typeName() + ":" + ((ArrayType) dType).elementType().typeName();
        }


    }

    public static Object getArrayToString(DataType dataType, Object value)
    {
        if (dataType.typeName().equals("array"))

        {
            ArrayType a = (ArrayType) dataType;
            DataType e = a.elementType();
            ArrayBuffer ab = (ArrayBuffer) value;
            Object[] d ;
            if (ab.size() > 0)
            {

                    d = new Object[ab.size()];
                    for (int i = 0; i < ab.array().length; i++) {
                        if (e.typeName().equals("array"))
                        {
                            d[i] = getArrayToString(e, ab.array()[i]);
                        }
                        else {
                            d[i] = (Double) ab.array()[i];
                        }

                    }

            }
            else
            {
                d = new Double[]{};
            }
            return Arrays.toString(d);
        }
        return "";
    }

    public static String getFieldTypeMapping(StructType s, String fieldName)
    {
        scala.collection.Iterator x = s.iterator();
        while (x.hasNext())
        {
            StructField f = (StructField) x.next();
            if (f.name().equals(fieldName) && !f.dataType().typeName().toString().toLowerCase().equals("struct"))
            {
                if(f.dataType().typeName().toLowerCase().equals("array"))
                {
                    if (((ArrayType) f.dataType()).elementType().typeName().toLowerCase().equals("array"))
                    {
                        return (f.dataType().typeName() + ":" + (getFieldTypeMapping((ArrayType) (((ArrayType) f.dataType()).elementType()), fieldName)));
                    }

                    else
                    {
                        return (f.dataType().typeName() + ":" + ((ArrayType) f.dataType()).elementType().typeName());
                    }
                }
                else
                {
                    return f.dataType().typeName();
                }
            }
            else
            {
                if (f.dataType().typeName().toString().toLowerCase().equals("struct"))
                {
                    String fieldType = getFieldTypeMapping((StructType) f.dataType(), fieldName);
                    if (!fieldType.equals(""))
                    {
                        return fieldType;
                    }
                }

            }
        }
        return "";
    }

    public static String getFieldTypeMapping(ArrayType d, String fieldName)
    {
        if (d.elementType().typeName().toLowerCase().equals("array"))
        {
            getFieldTypeMapping((ArrayType) d.elementType(), fieldName);
        }
        return (d.typeName() + ":" + d.elementType().typeName());
    }

    public static Integer convertToInteger(String s)
    {
        return Integer.parseInt(s);
    }

    public static Double convertToDouble(String s)
    {
        return Double.parseDouble(s);
    }

    public static Float convertToFloat(String s)
    {
        return Float.parseFloat(s);
    }

    public static Short convertToShort(String s)
    {
        return Short.parseShort(s);
    }

    public static Long convertToLong(String s)
    {
        return Long.parseLong(s);
    }

    public static Boolean convertToBoolean(String s)
    {
        return Boolean.parseBoolean(s);
    }

    public static Integer[] convertToIntegerArray(String[] s)
    {
        Integer[] results = new Integer[s.length];

        for (int i = 0; i < s.length; i++) {
            try {
                results[i] = Integer.parseInt(s[i]);
            } catch (NumberFormatException nfe)
            {

            };
        }
        return results;
    }

    public static Double[] convertToDoubleArray(String[] s)
    {
        Double[] results = new Double[s.length];

        for (int i = 0; i < s.length; i++) {
            try {
                results[i] = Double.parseDouble(s[i]);
            } catch (NumberFormatException nfe)
            {

            };
        }
        return results;
    }

    public static Float[] convertToFloatArray(String[] s)
    {
        Float[] results = new Float[s.length];

        for (int i = 0; i < s.length; i++) {
            try {
                results[i] = Float.parseFloat(s[i]);
            } catch (NumberFormatException nfe)
            {

            };
        }
        return results;
    }

    public static Short[] convertToShortArray(String[] s)
    {
        Short[] results = new Short[s.length];

        for (int i = 0; i < s.length; i++) {
            try {
                results[i] = Short.parseShort(s[i]);
            } catch (NumberFormatException nfe)
            {

            };
        }
        return results;
    }

    public static Long[] convertToLongArray(String[] s)
    {
        Long[] results = new Long[s.length];

        for (int i = 0; i < s.length; i++) {
            try {
                results[i] = Long.parseLong(s[i]);
            } catch (NumberFormatException nfe)
            {

            };
        }
        return results;
    }

    public static Boolean[] convertToBooleanArray(String[] s)
    {
        Boolean[] results = new Boolean[s.length];

        for (int i = 0; i < s.length; i++) {
            try {
                results[i] = Boolean.parseBoolean(s[i]);
            } catch (NumberFormatException nfe)
            {

            };
        }
        return results;
    }

    /*public static Double[] convertToDouble2DArray(String[] s)
    {
        String[] items = s.replaceFirst("\\[", "").substring(0,s.replaceFirst("\\[", "").lastIndexOf("]")).split("\\],");
        Double[][] results = new Double[s.length][];

        for (int i = 0; i < s.length; i++) {
            try {
                results[i] = Double.parseDouble(s[i]);
            } catch (NumberFormatException nfe)
            {

            };
        }
        return results;
    }*/

    public static org.apache.spark.mllib.linalg.Vector convertToVector(String s)
    {
        return Vectors.parse(s);
    }




}
