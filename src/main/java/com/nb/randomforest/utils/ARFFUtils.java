package com.nb.randomforest.utils;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.nio.file.Paths;
import java.util.ArrayList;

import org.apache.commons.lang3.StringUtils;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.nb.randomforest.entity.EventFeature;

import weka.core.Attribute;
import weka.core.Instances;

/**
 *
 */
public class ARFFUtils {

    public static void buildARFFV1(String sourcePath, String dumpName) throws Exception {
        ObjectMapper mapper = new ObjectMapper();
        ArrayList<Attribute> attributes = MyAttributeBuilder.buildMyAttributesV1();
    
        // 2. create Instances object
        Instances instances = new Instances(dumpName, attributes, 1);
        instances.setClassIndex(instances.numAttributes() - 1);
        System.out.println("ClassIndex:" + instances.classIndex());
        
        // 3. dump header
        File sourceFile = new File(sourcePath);
        File dumpedFile = new File(Paths.get(sourceFile.getParentFile().getAbsolutePath(), dumpName + ".arff").toString());
        BufferedReader br = new BufferedReader(new FileReader(sourceFile));
        BufferedWriter bw = new BufferedWriter(new FileWriter(dumpedFile));
        
        bw.write(instances.toString());
        String line = null;
        while ((line = br.readLine()) != null) {
            String[] datas = line.split("\t");
            if ("SUCCESS".equals(datas[3])) {
                String label = datas[2];
                String mStr = datas[4];
                String cStr = datas[5];
                try {
                    EventFeature feature = new EventFeature(
                        mapper.readTree(mStr),
                        mapper.readTree(cStr),
                        StringUtils.equals(label, "DIFF") ? "DIFF" : "EVENT"
                    );
//                    System.out.println(mapper.writeValueAsString(feature));
                    bw.write(feature.toStringV1());
                    bw.write("\n");
                } catch (Exception e) {
                    e.printStackTrace();
                    System.out.println(mStr);
                    System.out.println(cStr);
                    System.out.println(label);
                    break;
                }
            }
        }
        bw.close();
    }
    
    
    public static void buildARFFV2(String sourcePath, String dumpName) throws Exception {
        ObjectMapper mapper = new ObjectMapper();
        ArrayList<Attribute> attributes = MyAttributeBuilder.buildMyAttributesV2();
        
        // 2. create Instances object
        Instances instances = new Instances(dumpName, attributes, 1);
        instances.setClassIndex(instances.numAttributes() - 1);
        System.out.println("ClassIndex:" + instances.classIndex());
        
        // 3. dump header
        File sourceFile = new File(sourcePath);
        File dumpedFile = new File(Paths.get(sourceFile.getParentFile().getAbsolutePath(), dumpName + ".arff").toString());
//        FileUtils.writeStringToFile(dumpFile, instances.toString(), Charset.defaultCharset(), false);
        BufferedReader br = new BufferedReader(new FileReader(sourceFile));
        BufferedWriter bw = new BufferedWriter(new FileWriter(dumpedFile));
        
        bw.write(instances.toString());
        String line = null;
        while ((line = br.readLine()) != null) {
            String[] datas = line.split("\t");
            if ("SUCCESS".equals(datas[3])) {
                String label = datas[2];
                String mStr = datas[4];
                String cStr = datas[5];
                try {
                    EventFeature feature = new EventFeature(
                        mapper.readTree(mStr),
                        mapper.readTree(cStr),
                        StringUtils.equals(label, "DIFF") ? "DIFF" : "EVENT"
                    );
                    bw.write(feature.toStringV2());
                    bw.write("\n");
                } catch (Exception e) {
                    e.printStackTrace();
                    System.out.println(mStr);
                    System.out.println(cStr);
                    System.out.println(label);
                    break;
                }
            }
        }
        bw.close();
    }
    
    public static void buildARFFV3(String sourcePath, String dumpName) throws Exception {
        ObjectMapper mapper = new ObjectMapper();
        ArrayList<Attribute> attributes = MyAttributeBuilder.buildMyAttributesV3();
    
        // 2. create Instances object
        Instances instances = new Instances(dumpName, attributes, 1);
        instances.setClassIndex(instances.numAttributes() - 1);
        System.out.println("ClassIndex:" + instances.classIndex());
        
        // 3. dump header
        File sourceFile = new File(sourcePath);
        File dumpedFile = new File(Paths.get(sourceFile.getParentFile().getAbsolutePath(), dumpName + ".arff").toString());
        BufferedReader br = new BufferedReader(new FileReader(sourceFile));
        BufferedWriter bw = new BufferedWriter(new FileWriter(dumpedFile));
        
        bw.write(instances.toString());
        String line = null;
        while ((line = br.readLine()) != null) {
            String[] datas = line.split("\t");
            if ("SUCCESS".equals(datas[3])) {
                String label = datas[2];
                String mStr = datas[4];
                String cStr = datas[5];
                try {
                    EventFeature feature = new EventFeature(
                        mapper.readTree(mStr),
                        mapper.readTree(cStr),
                        label
                    );
//                    System.out.println(mapper.writeValueAsString(feature));
                    bw.write(feature.toStringV1());
                    bw.write("\n");
                } catch (Exception e) {
                    e.printStackTrace();
                    System.out.println(mStr);
                    System.out.println(cStr);
                    System.out.println(label);
                    break;
                }
            }
        }
        bw.close();
    }
    
    
    public static void main(String[] args) throws Exception {
        // buildARFFV3("/mnt/nlp/albert/clustering/data/dedup_train_data_v3/train_fields", "train");
        buildARFFV3("/mnt/nlp/albert/clustering/data/dedup_train_data_v3/test_fields", "test");
    }
}
