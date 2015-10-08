import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Scanner;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.classify.WeightedPropertyVectorWritable;
import org.apache.mahout.clustering.iterator.ClusterWritable;
import org.apache.mahout.clustering.kmeans.KMeansDriver;
import org.apache.mahout.clustering.kmeans.Kluster;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.NamedVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

public class Main {
    
    // Write data to sequence files in Hadoop (write the vector to sequence file)
    public static void writePointsToFile(List<Vector> points, String fileName,
            							FileSystem fs,Configuration conf) throws IOException {
        
                    Path path = new Path(fileName);
                    SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, path, 
                    									LongWritable.class, VectorWritable.class);
                    long recNum = 0;
                    VectorWritable vec = new VectorWritable();
                    
                    for (Vector point : points) {
                        vec.set(point);
                        writer.append(new LongWritable(recNum++), vec);
                    }
                    
                    writer.close();
    }
    
    public static void dbg(String msg) {
    	System.out.println(msg);
    }
    
    public static List getPointsFromFile() throws FileNotFoundException {
    	List points = new ArrayList();
    	
    	BufferedReader br = null;
    	String line = "";
     
    	try {
    		br = new BufferedReader(new FileReader("assets/All.csv"));
    		while ((line = br.readLine()) != null) {
     
    			String[] tokens = line.split(",");
    			String id = tokens[0];
    			double[] tmp = new double[3];
    			tmp[0] = Integer.parseInt(tokens[1])*1000;
    			tmp[1] = Integer.parseInt(tokens[2])*1000;
    			tmp[2] = Integer.parseInt(tokens[3]);
    			
    			NamedVector vector = new NamedVector(new DenseVector(tmp), id);
                points.add(vector);
     
    		}
     
    	} catch (FileNotFoundException e) {
    		e.printStackTrace();
    	} catch (IOException e) {
    		e.printStackTrace();
    	} finally {
    		if (br != null) {
    			try {
    				br.close();
    			} catch (IOException e) {
    				e.printStackTrace();
    			}
    		}
    	}
        
        return points;
    }
    
    public static void main(String args[]) throws Exception {
        
        // specify the number of clusters 
        int k = 5;
        
        // read the values (features) - generate vectors from input data
        
          List vectors = getPointsFromFile();
          
          // Create input directories for data
          File testData = new File("testdata");
          
          if (!testData.exists()) {
            testData.mkdir();
          }
          testData = new File("testdata/points");
          if (!testData.exists()) {
            testData.mkdir();
          }
          
          // Write initial centers
          Configuration conf = new Configuration();
          
          FileSystem fs = FileSystem.get(conf);

          // Write vectors to input directory
          writePointsToFile(vectors,
              "testdata/points/file1.txt", fs, conf);
          
          Path path = new Path("testdata/clusters/part-00000");
          
          SequenceFile.Writer writer = 
        		  new SequenceFile.Writer(fs, conf, path, Text.class, Kluster.class);
          
          for (int i = 0; i < k; i++) {
            Vector vec = (Vector) vectors.get(i);
            
            // write the initial center here as vec
            Kluster cluster = new Kluster(vec, i, new EuclideanDistanceMeasure());
            writer.append(new Text(cluster.getIdentifier()), cluster);
          }
          
          writer.close();
        
          //openFileOutput("output/cpoints.txt");
          File file = new File("output/cPoints.txt");
          //if(!file.isExist()) {
          //    file.create();
          //}
          PrintWriter fos = new PrintWriter(file);
          //fos.write(content.getBytes());
            
          
          // Run K-means algorithm
        KMeansDriver.run(conf, new Path("testdata/points"), new Path("testdata/clusters"),
        new Path("output"), 0.1, 5, true, 0.2, false);
          SequenceFile.Reader reader
              = new SequenceFile.Reader(fs,
                  new Path("output/" + Cluster.CLUSTERED_POINTS_DIR
                      + "/part-m-00000"), conf);
        IntWritable key = new IntWritable();
        
        printCluster(conf, new Path("output/clusters-5-final/part-r-00000"));
		


	
        
        // Read output values
        WeightedPropertyVectorWritable value = new WeightedPropertyVectorWritable();
        while (reader.next(key, value)) {
        	Vector vec = value.getVector();
        	double tmp = vec.get(0);
            //System.out.println(value + " belongs to cluster " + key.toString() + "\n");
            
            fos.write("Cluster " + key.toString() + " : " + value.toString()  + "\n");
        }
        /*while (reader.next(key, value)) {
            String key0="100";
            
            	int i=0;
            		if (key0 !=key.toString()){
            			key0=key.toString();
            			i++;
            			fos.write("\n" + "Cluster "+ i +" :" + "\n");
            		}
               fos.write(value.toString());   */ 
        
          reader.close();
          
          fos.close();
        }
        
        /*BufferedReader br1 = null;
    	String line = "";
        
        br1 = new BufferedReader(new FileReader("output/cpoints.txt"));
		while ((line = br1.readLine()) != null) {
 
			String[] tokens = line.split("\t");
			String cNumber = tokens[0];
			double[] tmp = new double[6];
			tmp[0] = Integer.parseInt(tokens[1]);
			tmp[1] = Integer.parseInt(tokens[2]);
			tmp[2] = Integer.parseInt(tokens[3]);
			tmp[3] = Integer.parseInt(tokens[4]);
			tmp[4] = Integer.parseInt(tokens[5]);
			tmp[5] = Integer.parseInt(tokens[6]);
			System.out.println(tmp[5]);

 
		}*/
		
    

private static void printCluster(Configuration conf, Path path) throws IOException {
	FileSystem fs = FileSystem.get(conf);
	SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);
	
	IntWritable key_output = new IntWritable();
	// Read output values
	ClusterWritable value_output = new ClusterWritable();
	
	File file1 = new File("output/cReport.txt");
    //if(!file.isExist()) {//¿ÉÄÜÊÇfile.exist()
    //    file.create();
    //}
    PrintWriter fos1 = new PrintWriter(file1);
    
   File file2 = new File("output/cResult.txt");
    //if(!file.isExist()) {
    //    file.create();
    //}
    PrintWriter fos2 = new PrintWriter(file2);
    
    fos1.write("Clusters  :" + 
    		"\n\n");
	
	while (reader.next(key_output, value_output)) {
	//	NamedVector vec = (NamedVector) value_output;
	//	double[] tmp = new double[3];
	//	tmp[0] = Math.round(vec.get(0));
	//	tmp[1] = Math.round(vec.get(1));
	//	tmp[2] = Math.round(vec.get(2) * 1000);
		
	//	Vector vec2 = new DenseVector(tmp);
		//System.out.println(value_output.getValue().toString() + " / " + key_output.toString());
        //while (reader.next(key_output, value_output)) {
			
		    long n = value_output.getValue().getNumObservations();
		    long total = value_output.getValue().getTotalObservations();
		    long totalN=8323;
		
			Vector vec = value_output.getValue().getCenter();
			long age = Math.round(vec.get(0)/1000);
			long edu = Math.round(vec.get(1)/1000);
			long income = Math.round(vec.get(2));
			
			Vector vec1 = value_output.getValue().getRadius();
			long ageR = Math.round(vec1.get(0)/1000);
			long eduR = Math.round(vec1.get(1)/1000);
			long incomeR = Math.round(vec1.get(2));
			
			long ageLeft=age-ageR;
			long ageRight=age+ageR;
			long eduLeft=edu-eduR;
			long eduRight=edu+eduR;
			long incomeLeft=income-incomeR;
			long incomeRight=income+incomeR;
			
				        
			

	        fos1.write("information for cluster " + key_output.get() + " :" + 
	        		"\n age = " + age +  " years old" + 
	        		"\n education years = " + edu + " years" +  
	        		"\n income = " + income + " dollars" + 
	        		"\n car numbers = " + n + " cars" +
	        		"\n car ratio = " + n*100/totalN + "%" +
	        		"\n" +
	        		"\n number range :" +
	        		"\n age range : from " + ageLeft +  " to " + ageRight + " years old" + 
	        		"\n eduation range : from " + eduLeft +  " to " + eduRight + " years" + 
	        		"\n income range : from " + incomeLeft +  " to " + incomeRight + " dollars" + 
	        		"\n\n");
	        System.out.println(value_output.getValue().toString() + "\n");
	        fos2.write(value_output.getValue().toString() + "\n");
	        fos1.write("\n");
	        
	        
		}
	fos1.close();
	fos2.close();

	//}
	reader.close();
	
}
          

}