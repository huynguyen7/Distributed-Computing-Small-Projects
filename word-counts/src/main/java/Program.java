import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

import java.util.Arrays;

public class Program {
    public static void main(String[] args) {

        // path to the folder that contains text files.
        final String inputPath = "./src/main/resources";
        final String outputPath = "./output";

        wordCount(inputPath, outputPath);
    }

    private static void wordCount(final String inputPath, final String outputPath) {
        // setMaster -> "local" means Spark threads should run on localhost.
        SparkConf sparkConf = new SparkConf().setAppName("Word Counts Program")
                .setMaster(String.format("local[%d]", Runtime.getRuntime().availableProcessors()));

        // start Spark context.
        JavaSparkContext sc = new JavaSparkContext(sparkConf);
        // sc.setLogLevel("WARN");

        // read all text files inside path directory to RDD.
        JavaRDD<String> lines = sc.textFile(inputPath);

        // convert each line into a list of words.
        JavaRDD<String> words = lines.flatMap(line -> Arrays.asList(line.split(" ")).iterator());

        // create intermediate <key-value> pairs for every word (default value is 1).
        JavaPairRDD<String, Integer> intermediatePairs = words.mapToPair(word -> new Tuple2<String, Integer>(word, 1));

        // get counts for each word.
        JavaPairRDD<String, Integer> wordCounts = intermediatePairs.reduceByKey((x, y) -> x + y);

        // save to ./output
        wordCounts.map(a -> String.format("%s : %d", a._1, a._2))
                .saveAsTextFile(outputPath);

        // stop Spark context.
        sc.stop();
    }
}
