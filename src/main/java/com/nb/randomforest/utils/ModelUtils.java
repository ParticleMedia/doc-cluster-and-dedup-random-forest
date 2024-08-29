package com.nb.randomforest.utils;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.stream.Collectors;

import org.apache.commons.lang3.StringUtils;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.nb.randomforest.entity.EventFeature;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ArffLoader;
import weka.core.converters.ConverterUtils.DataSource;

public class ModelUtils {
    
    public static final ObjectMapper mapper = new ObjectMapper();
    

    /**
     * Load data set
     */
    public static Instances getDataSet(String fileName) throws IOException {
        int classIdx = 1;
        ArffLoader loader = new ArffLoader();
        loader.setSource(new FileInputStream(new File(fileName)));
        Instances dataSet = loader.getDataSet();
        dataSet.setClassIndex(classIdx);
        return dataSet;
    }
    
    
    /**
     * 半监督学习 : 基于构建的半监督数据 + 训练数据 训练模型
     * 训练方式一 : 先基于半监督数据训练, 再基于训练数据训练 传统 fine tuning 方式, 但是不适用 RandomForest?
     * 训练方式二 : 混合半监督数据 + 训练数据, 通过调整混合比例及方式起到 fine tuning 的作用
     */
    public static void trainModelSemiSupervised(String semiSupFile, String trainFile, String testFile) throws Exception {
        //semi supervised data set
        Instances trainingDataSetSemiSupervised = getDataSet(semiSupFile);
        trainingDataSetSemiSupervised.setClassIndex(trainingDataSetSemiSupervised.numAttributes() - 1);
        
        //
        Instances trainingDataSet = getDataSet(trainFile);
        trainingDataSet.setClassIndex(trainingDataSet.numAttributes() - 1);
        
        //test data set
        Instances testingDataSet = getDataSet(testFile);
        testingDataSet.setClassIndex(trainingDataSetSemiSupervised.numAttributes() - 1);
    
        RandomForest forest = new RandomForest();
        forest.setBatchSize("200");
        forest.setNumIterations(200);
        forest.setDebug(false);
        forest.setNumFeatures(4); // random feature num = log_2{feature num}
        forest.setComputeAttributeImportance(true);
        //
        forest.buildClassifier(trainingDataSetSemiSupervised);
        
        //NOTE : 无效
        forest.setClassifier(forest.getClassifier());
        forest.buildClassifier(trainingDataSet);
    
        Evaluation eval = new Evaluation(trainingDataSetSemiSupervised);
        eval.evaluateModel(forest, testingDataSet);
        /** Print the algorithm summary */
        System.out.println("** Decision Tress Evaluation with Datasets **");
        System.out.println(eval.toSummaryString());
        System.out.print(" the expression for the input data as per alogorithm is ");
        System.out.println(forest);
    
        // dump random forest model to file
        SerializationHelper.write(Paths.get(new File(trainFile).getParent(),"forest.model").toString(), forest);
    }
    

    /**
     * Model training and estimate based on train, test file.
     */
    public static RandomForest trainModel(
        String rootPath, Instances trainingDataSet, Integer randomTreeNum, Integer featureNum, Integer maxDepth
    ) throws Exception {
        RandomForest forest = new RandomForest();
        //some forest params
        forest.setNumIterations(randomTreeNum);
        forest.setNumFeatures(featureNum); // random feature num = log_2{feature num}
        forest.setMaxDepth(maxDepth);
        
        //some debug info
        forest.setDebug(false);
        forest.setComputeAttributeImportance(true);
        
        //build classifier
        forest.buildClassifier(trainingDataSet);
        SerializationHelper.write(Paths.get(rootPath,"forest.model").toString(), forest);
        return forest;
    }
    
    
    /**
     * 基于 DOC Field 文件计算 Precision, Recall
     */
    public static void predictABTestBasedDocField(RandomForest forest, String sourceFile) throws Exception {
        return;
    }
    
    
    /**
     * 基于 ARFF 文件计算 Precision, Recall
     */
    public static void predictABTestBasedARFF(RandomForest forest, String sourceFile, String docPairFile) throws Exception {
        DataSource dataSource = new DataSource(sourceFile);
        String badpath = Paths.get(Paths.get(sourceFile).getParent().toString(), "badcase").toString();
        BufferedWriter bw = new BufferedWriter(new FileWriter(new File(badpath)));
        Instances testDataset = dataSource.getDataSet();
        testDataset.setClassIndex(testDataset.numAttributes() - 1);
    
        BufferedReader br = new BufferedReader(new FileReader(new File(docPairFile)));
        String line = null;
        int j = 22;
        Map<Integer, String> indexDocMap = new HashMap<>();
        while ((line = br.readLine()) != null) {
            String[] docPairFields = line.split("\t");
            indexDocMap.put(j, docPairFields[0] + "\t" + docPairFields[1]);
            j++;
        }

        //Real Positive
        int rpEVT = 0;
        int rpDUP = 0;
        //True Positive
        int tpDIF = 0;
        int tpEVT = 0;
        int tpDUP = 0;
        //Predict Positive
        int ppEVT = 0;
        int ppDUP = 0;
        
        Attribute clsAttribute = testDataset.classAttribute();
        int i = 22;//打印时便于找到 arff 中对应的结果
        //逐个计算
        bw.write("Master\t\tCandidate\tDIF_WEIGHT\tEVT_WEIGHT\tDUP_WEIGHT\tPredict\tReal");
        bw.write("\n");
        for (Instance instance : testDataset) {
            double[] distribute = forest.distributionForInstance(instance);
            double difScr = distribute[0];
            double evtScr = distribute[1];
            double dupScr = distribute[2];
            
            String pCls = "";
            String rCls = clsAttribute.value(Double.valueOf(instance.value(instance.classIndex())).intValue());
            if (difScr < 0.56d && dupScr < 0.7d) {//EVENT
                ppEVT++;
                pCls = "EVENT";
                if (StringUtils.equals(rCls, "EVENT")) {
                    tpEVT++;
                }
            } else if (dupScr >= 0.7d) {//DUP
                ppDUP++;
                pCls = "DUP";
                if (StringUtils.equals(rCls, "DUP")) {
                    tpDUP++;
                }
            } else {
                pCls = "DIFF";
            }
            
            if (StringUtils.equals(rCls, "DUP")) {
                rpDUP++;
            } else if (StringUtils.equals(rCls, "EVENT")) {
                rpEVT++;
            }
            bw.write(indexDocMap.get(i) + "\t" + String.valueOf(difScr) + "\t" + String.valueOf(evtScr) + "\t" + String.valueOf(dupScr) + "\t" + pCls + "\t" + rCls);
            bw.write("\n");
            i++;
        }
        bw.close();
        
        double tprDUP = tpDUP / Double.valueOf(ppDUP);
        double rcrDUP = tpDUP / Double.valueOf(rpDUP);
        System.out.println("## TPR DUP    : " + String.valueOf(tpDUP) + "/" + String.valueOf(ppDUP) + "=" + String.valueOf(tprDUP));
        System.out.println("## RCR DUP    : " + String.valueOf(tpDUP) + "/" + String.valueOf(rpDUP) + "=" + String.valueOf(rcrDUP));
        double tprEVT = tpEVT / Double.valueOf(ppEVT);
        double rcrEVT = tpEVT / Double.valueOf(rpEVT);
        System.out.println("## TPR EVT    : " + String.valueOf(tpEVT) + "/" + String.valueOf(ppEVT) + "=" + String.valueOf(tprEVT));
        System.out.println("## RCR EVT    : " + String.valueOf(tpEVT) + "/" + String.valueOf(rpEVT) + "=" + String.valueOf(rcrEVT));
    }
    
    
    /**
     * 基于 Feature V1(16) + Estimate Data => Model Estimate
     *
     */
    public static void predictEstimateDataFeatureV0(RandomForest forest, String docPairFile) throws Exception {
        ObjectMapper objectMapper = new ObjectMapper();
        
        ArrayList<Attribute> attributes = new ArrayList<>();
        ArrayList<String> attVals = new ArrayList<>();
        // - numeric
        attributes.add(new Attribute("TitleDist"));
        attributes.add(new Attribute("SameSRC"));
        attributes.add(new Attribute("CWordSpan"));
        attributes.add(new Attribute("EpochSpan"));
        attributes.add(new Attribute("ParghSpan"));
        attributes.add(new Attribute("SimhashDist"));
        attributes.add(new Attribute("KWSRatio"));
        attributes.add(new Attribute("ChannelRatio"));
        attributes.add(new Attribute("COrgOverlapRatio"));
        attributes.add(new Attribute("CLocOverlapRatio"));
        attributes.add(new Attribute("CPrsOverlapRatio"));
        attributes.add(new Attribute("TOrgOverlapRatio"));
        attributes.add(new Attribute("TLocOverlapRatio"));
        attributes.add(new Attribute("TPrsOverlapRatio"));
        attributes.add(new Attribute("CatOverlapRatio"));
        attributes.add(new Attribute("GEOOverlapRatio"));
        // - nominal
        attVals.add("DIFF");
        attVals.add("EVENT");
        attributes.add(new Attribute("Label", attVals));
        //
        //Real Positive
        int rpDUP = 0;
        int rpEVT = 0;
        //True Positive
        int tpDIF = 0;
        int tpEVT = 0;
        int tpDUP = 0;
        //Predict Positive
        int ppEVT = 0;
        int ppDUP = 0;
        //
        BufferedReader br = new BufferedReader(new FileReader(new File(docPairFile)));
        String badpath = Paths.get(Paths.get(docPairFile).getParent().toString(), "badcase").toString();
        BufferedWriter bw = new BufferedWriter(new FileWriter(new File(badpath)));
        bw.write("Master\t\tCandidate\tDIF_WEIGHT\tEVT_WEIGHT\tPredict\tReal");
        bw.write("\n");
        String line = null;
        while ((line = br.readLine()) != null) {
            String[] docPairFields = line.split("\t");
            String mDoc = docPairFields[0];
            String cDoc = docPairFields[1];
            String pCls = "";
            String rCls = docPairFields[2];
            JsonNode mNode = objectMapper.readTree(docPairFields[4]);
            JsonNode cNode = objectMapper.readTree(docPairFields[5]);
            EventFeature feature = new EventFeature(mNode, cNode, rCls);
            Instances instances = new Instances(UUID.randomUUID().toString(), attributes, 1);
            instances.setClassIndex(instances.numAttributes() - 1);
            instances.add(feature.toInstanceV0());
            double[] distribute = forest.distributionsForInstances(instances)[0];
            double difScr = distribute[0];
            double evtScr = distribute[1];
            if (evtScr > 0.92) {
                ppDUP++;
                pCls = "DUP";
                if (StringUtils.equals(rCls, pCls)) {
                    tpDUP++;
                }
            } else if (evtScr > 0.58) {//EVENT
                if (!StringUtils.equals(rCls, "DUP")) {
                    ppEVT++;
                }
                pCls = "EVENT";
                if (StringUtils.equals(rCls, "EVENT")) {
                    tpEVT++;
                }
            } else {
                pCls = "DIFF";
            }
    
            if (StringUtils.equals(rCls, "EVENT")) {
                rpEVT++;
            }
            if (StringUtils.equals(rCls, "DUP")) {
                rpDUP++;
            }
    
            if (!StringUtils.equals(rCls, pCls)) {
                bw.write(mDoc + "\t" + cDoc + "\t" + String.valueOf(difScr) + "\t" + String.valueOf(evtScr) + "\t" + pCls + "\t" + rCls);
                bw.write("\n");
            }
        }
        
        bw.close();

        double tprDUP = tpDUP / Double.valueOf(ppDUP);
        double rcrDUP = tpDUP / Double.valueOf(rpDUP);
        System.out.println("## TPR DUP    : " + String.valueOf(tpDUP) + "/" + String.valueOf(ppDUP) + "=" + String.valueOf(tprDUP));
        System.out.println("## RCR DUP    : " + String.valueOf(tpDUP) + "/" + String.valueOf(rpDUP) + "=" + String.valueOf(rcrDUP));

        double tprEVT = tpEVT / Double.valueOf(ppEVT);
        double rcrEVT = tpEVT / Double.valueOf(rpEVT);
        System.out.println("## TPR EVT    : " + String.valueOf(tpEVT) + "/" + String.valueOf(ppEVT) + "=" + String.valueOf(tprEVT));
        System.out.println("## RCR EVT    : " + String.valueOf(tpEVT) + "/" + String.valueOf(rpEVT) + "=" + String.valueOf(rcrEVT));
    }
    
    
    /**
     * 基于 Feature V1(55) + Estimate Data => Model Estimate
     *
     */
    public static void predictEstimateDataFeatureV1(RandomForest forest, String docPairFile) throws Exception {
        ObjectMapper objectMapper = new ObjectMapper();
        
        ArrayList<Attribute> attributes = MyAttributeBuilder.buildMyAttributesV1();
        //Real Positive
        int rpDUP = 0;
        int rpEVT = 0;
        //True Positive
        int tpDIF = 0;
        int tpEVT = 0;
        int tpDUP = 0;
        //Predict Positive
        int ppEVT = 0;
        int ppDUP = 0;
        //
        BufferedReader br = new BufferedReader(new FileReader(new File(docPairFile)));
        String badpath = Paths.get(Paths.get(docPairFile).getParent().toString(), "badcase").toString();
        BufferedWriter bw = new BufferedWriter(new FileWriter(new File(badpath)));
        bw.write("Master\t\tCandidate\tDIF_WEIGHT\tEVT_WEIGHT\tPredict\tReal");
        bw.write("\n");
        String line = null;
        while ((line = br.readLine()) != null) {
            String[] docPairFields = line.split("\t");
            String mDoc = docPairFields[0];
            String cDoc = docPairFields[1];
            String pCls = "";
            String rCls = docPairFields[2];
            JsonNode mNode = objectMapper.readTree(docPairFields[4]);
            JsonNode cNode = objectMapper.readTree(docPairFields[5]);
            EventFeature feature = new EventFeature(mNode, cNode, rCls);
            Instances instances = new Instances(UUID.randomUUID().toString(), attributes, 1);
            instances.setClassIndex(instances.numAttributes() - 1);
            instances.add(feature.toInstanceV1());
            double[] distribute = forest.distributionsForInstances(instances)[0];
            double difScr = distribute[0];
            double evtScr = distribute[1];
            if (evtScr > 0.88) {
                ppDUP++;
                pCls = "DUP";
                if (StringUtils.equals(rCls, pCls)) {
                    tpDUP++;
                }
                
                ppEVT++;
                if (StringUtils.equals(rCls, "EVENT") || StringUtils.equals(rCls, "DUP")) {
                    tpEVT++;
                }
            } else if (evtScr > 0.45) {//EVENT
                ppEVT++;
                pCls = "EVENT";
                if (StringUtils.equals(rCls, "EVENT") || StringUtils.equals(rCls, "DUP")) {
                    tpEVT++;
                }
            } else {
                pCls = "DIFF";
            }
            
            if (StringUtils.equals(rCls, "EVENT")) {
                rpEVT++;
            }
            if (StringUtils.equals(rCls, "DUP")) {
                rpDUP++;
                rpEVT++;
            }
            
            if (!StringUtils.equals(rCls, pCls)) {
                bw.write(mDoc + "\t" + cDoc + "\t" + String.valueOf(difScr) + "\t" + String.valueOf(evtScr) + "\t" + pCls + "\t" + rCls);
                bw.write("\n");
            }
        }
        
        bw.close();
        
        double tprDUP = tpDUP / Double.valueOf(ppDUP);
        double rcrDUP = tpDUP / Double.valueOf(rpDUP);
        System.out.println("## TPR DUP    : " + String.valueOf(tpDUP) + "/" + String.valueOf(ppDUP) + "=" + String.valueOf(tprDUP));
        System.out.println("## RCR DUP    : " + String.valueOf(tpDUP) + "/" + String.valueOf(rpDUP) + "=" + String.valueOf(rcrDUP));
        
        double tprEVT = tpEVT / Double.valueOf(ppEVT);
        double rcrEVT = tpEVT / Double.valueOf(rpEVT);
        System.out.println("## TPR EVT    : " + String.valueOf(tpEVT) + "/" + String.valueOf(ppEVT) + "=" + String.valueOf(tprEVT));
        System.out.println("## RCR EVT    : " + String.valueOf(tpEVT) + "/" + String.valueOf(rpEVT) + "=" + String.valueOf(rcrEVT));
    }
    
    
    /**
     * 基于 Feature V1(53) + Estimate Data => Model Estimate
     *
     */
    public static void predictEstimateDataFeatureV2(RandomForest forest, String docPairFile) throws Exception {
        ObjectMapper objectMapper = new ObjectMapper();
        
        ArrayList<Attribute> attributes = MyAttributeBuilder.buildMyAttributesV2();
        //
        //Real Positive
        int rpDUP = 0;
        int rpEVT = 0;
        //True Positive
        int tpDIF = 0;
        int tpEVT = 0;
        int tpDUP = 0;
        //Predict Positive
        int ppEVT = 0;
        int ppDUP = 0;
        //
        BufferedReader br = new BufferedReader(new FileReader(new File(docPairFile)));
        String badpath = Paths.get(Paths.get(docPairFile).getParent().toString(), "badcase").toString();
        BufferedWriter bw = new BufferedWriter(new FileWriter(new File(badpath)));
        bw.write("Master\t\tCandidate\tDIF_WEIGHT\tEVT_WEIGHT\tPredict\tReal");
        bw.write("\n");
        String line = null;
        while ((line = br.readLine()) != null) {
            String[] docPairFields = line.split("\t");
            String mDoc = docPairFields[0];
            String cDoc = docPairFields[1];
            String pCls = "";
            String rCls = docPairFields[2];
            JsonNode mNode = objectMapper.readTree(docPairFields[4]);
            JsonNode cNode = objectMapper.readTree(docPairFields[5]);
            EventFeature feature = new EventFeature(mNode, cNode, rCls);
            Instances instances = new Instances(UUID.randomUUID().toString(), attributes, 1);
            instances.setClassIndex(instances.numAttributes() - 1);
            instances.add(feature.toInstanceV2());
            double[] distribute = forest.distributionsForInstances(instances)[0];
            double difScr = distribute[0];
            double evtScr = distribute[1];
            if (evtScr > 0.98765) {
                ppDUP++;
                pCls = "DUP";
                if (StringUtils.equals(rCls, pCls)) {
                    tpDUP++;
                }
            } else if (evtScr > 0.98765) {//EVENT
                if (!StringUtils.equals(rCls, "DUP")) {
                    ppEVT++;
                }
                pCls = "EVENT";
                if (StringUtils.equals(rCls, "EVENT")) {
                    tpEVT++;
                }
            } else {
                pCls = "DIFF";
            }
            
            if (StringUtils.equals(rCls, "EVENT")) {
                rpEVT++;
            }
            if (StringUtils.equals(rCls, "DUP")) {
                rpDUP++;
            }
            
            if (!StringUtils.equals(rCls, pCls)) {
                bw.write(mDoc + "\t" + cDoc + "\t" + String.valueOf(difScr) + "\t" + String.valueOf(evtScr) + "\t" + pCls + "\t" + rCls);
                bw.write("\n");
            }
        }
        
        bw.close();
        
        double tprDUP = tpDUP / Double.valueOf(ppDUP);
        double rcrDUP = tpDUP / Double.valueOf(rpDUP);
        System.out.println("## TPR DUP    : " + String.valueOf(tpDUP) + "/" + String.valueOf(ppDUP) + "=" + String.valueOf(tprDUP));
        System.out.println("## RCR DUP    : " + String.valueOf(tpDUP) + "/" + String.valueOf(rpDUP) + "=" + String.valueOf(rcrDUP));
        
        double tprEVT = tpEVT / Double.valueOf(ppEVT);
        double rcrEVT = tpEVT / Double.valueOf(rpEVT);
        System.out.println("## TPR EVT    : " + String.valueOf(tpEVT) + "/" + String.valueOf(ppEVT) + "=" + String.valueOf(tprEVT));
        System.out.println("## RCR EVT    : " + String.valueOf(tpEVT) + "/" + String.valueOf(rpEVT) + "=" + String.valueOf(rcrEVT));
    }
    
    /**
     * 基于 Feature V3(55) + Estimate Data => Model Estimate
     * But has 3 output classes DIFF EVENT DUP
     */
    public static void predictEstimateDataFeatureV3(RandomForest forest, String docPairFile, boolean callOnline) throws Exception {
        ObjectMapper objectMapper = new ObjectMapper();
        
        ArrayList<Attribute> attributes = MyAttributeBuilder.buildMyAttributesV3();
        
        int[][] confusion = new int[3][3];
        int right = 0;
        int total = 0;
        
        BufferedReader br = new BufferedReader(new FileReader(new File(docPairFile)));
        String badpath = Paths.get(Paths.get(docPairFile).getParent().toString(), callOnline ? "badcase_online" : "badcase").toString();
        BufferedWriter bw = new BufferedWriter(new FileWriter(new File(badpath)));
        bw.write("Master\tCandidate\tDIF_WEIGHT\tEVT_WEIGHT\tDUP_WEIGHT\tPredict\tReal");
        bw.write("\n");
        String line = null;
        while ((line = br.readLine()) != null) {
            String[] docPairFields = line.split("\t");
            String mDoc = docPairFields[0];
            String cDoc = docPairFields[1];
            String pCls = "";
            String rCls = docPairFields[2];
            JsonNode mNode = objectMapper.readTree(docPairFields[4]);
            JsonNode cNode = objectMapper.readTree(docPairFields[5]);
            double[] distribute = {0, 0, 0};
            double difScr, evtScr, dupScr;
            if (callOnline) {
                ObjectNode rootNode = objectMapper.createObjectNode();
                rootNode.set("master", mNode);
                ArrayNode candidatesArray = objectMapper.createArrayNode();
                candidatesArray.add(cNode);
                rootNode.set("candidates", candidatesArray);
                JsonNode res = FileUtils.post("http://doc-clu-dedup-random-forest-v2.k8s.nb-prod.com/document", rootNode);
                res = res.get(0);
                pCls = res.get("label").asText();
                double score = res.get("score").asDouble();
                difScr = pCls.equals("DIFF") ? score : 0;
                evtScr = pCls.equals("EVENT") ? score : 0;
                dupScr = pCls.equals("DUP") ? score : 0;
            } else {
                EventFeature feature = new EventFeature(mNode, cNode, rCls);
                Instances instances = new Instances(UUID.randomUUID().toString(), attributes, 1);
                instances.setClassIndex(instances.numAttributes() - 1);
                instances.add(feature.toInstanceV1());
                distribute = forest.distributionsForInstances(instances)[0];
            
                difScr = distribute[0];
                evtScr = distribute[1];
                dupScr = distribute[2];
                
                int method = 3;
                if (method == 1) {
                    if (dupScr >= evtScr && dupScr >= difScr) {
                        pCls = "DUP";
                    } else if (evtScr >= dupScr && evtScr >= difScr) {
                        pCls = "EVENT";
                    } else {
                        pCls = "DIFF";
                    }
                } else if (method == 2) {
                    if (dupScr - evtScr >= 0.0 && dupScr - difScr >= 0.2) {
                        pCls = "DUP";
                    } else if (evtScr - dupScr >= 0.0 && evtScr - difScr >= 0.2) {
                        pCls = "EVENT";
                    } else if (Math.abs(evtScr - dupScr) <= 0.2 && difScr < 0.3) {
                        pCls = "EVENT";
                    } else {
                        pCls = "DIFF";
                    }
                } else if (method == 3) {
                    if (dupScr >= evtScr && dupScr >= difScr) {
                        pCls = "DUP";
                    } else if (evtScr >= dupScr && evtScr - difScr >= 0.2) {
                        pCls = "EVENT";
                    } else {
                        pCls = "DIFF";
                    }
                }
            }
            
            int pIndex = pCls.equals("DIFF") ? 0 : pCls.equals("EVENT") ? 1 : 2;
            int rIndex = rCls.equals("DIFF") ? 0 : rCls.equals("EVENT") ? 1 : 2;
            confusion[rIndex][pIndex]++;
            
            if (!StringUtils.equals(rCls, pCls)) {
                bw.write(mDoc + "\t" + cDoc + "\t" + String.valueOf(difScr) + "\t"
                    + String.valueOf(evtScr) + "\t" + String.valueOf(dupScr) + "\t"
                    + pCls + "\t" + rCls);
                bw.write("\n");
            } else {
                right++;
            }
            total++;
        }
        
        bw.close();
        
        String summary = "Accuracy: " + String.valueOf((float) right/total);
        summary += "\nReal\\Predicted\tDIFF\tEVENT\tDUP";
        summary += "\nDIFF\t" + Arrays.stream(confusion[0]).mapToObj(String::valueOf).collect(Collectors.joining("\t"));
        summary += "\nEVENT\t" + Arrays.stream(confusion[1]).mapToObj(String::valueOf).collect(Collectors.joining("\t"));
        summary += "\nDUP\t" + Arrays.stream(confusion[2]).mapToObj(String::valueOf).collect(Collectors.joining("\t"));
        System.out.println(summary);
        String summpath = Paths.get(Paths.get(docPairFile).getParent().toString(), callOnline ? "summary_online" : "summary").toString();
        BufferedWriter summw = new BufferedWriter(new FileWriter(new File(summpath)));
        summw.write(summary);
        summw.close();
    }
    
    
    /**
     *
     */
    public static List<double[]> predictOnline(RandomForest forest, JsonNode masterNode, JsonNode canditNodes) throws Exception {
        ArrayList<Attribute> attributes = MyAttributeBuilder.buildMyAttributesV1();
        Instances instances;
        
        // 2. create Instances object
        instances = new Instances(UUID.randomUUID().toString(), attributes, 1);
        instances.setClassIndex(instances.numAttributes() - 1);
        for (JsonNode canditNode : canditNodes) {
            EventFeature feature = new EventFeature(masterNode, canditNode, null);
//            System.out.println(mapper.writeValueAsString(feature));
            instances.add(feature.toInstanceV1());
        }
        
        // 3.
        List<double[]> result = new ArrayList<>();
        double[][] canditResults = forest.distributionsForInstances(instances);
        for (int j = 0; j < canditResults.length; j++) {
            result.add(canditResults[j]);
        }
        return result;
    }
    
    
    /**
     *
     */
    public static List<Double> getTgtFeature(JsonNode masterNode, JsonNode canditNodes) throws Exception {
        List<Double> result = new ArrayList<>();
        ArrayList<Attribute> attributes = MyAttributeBuilder.buildMyAttributesV1();
        Instances instances;
    
        // 2. create Instances object
        instances = new Instances(UUID.randomUUID().toString(), attributes, 1);
        instances.setClassIndex(instances.numAttributes() - 1);
        for (JsonNode canditNode : canditNodes) {
            EventFeature feature = new EventFeature(masterNode, canditNode, null);
            result.add(feature.getcKWSRatio());
        }
        
        return result;
    }
    
    
    /**
     * 基于模型分数构造相似度矩阵
     */
    public static void buildModelSimMatrix() throws Exception {
        ObjectMapper mapper = new ObjectMapper();
        String modelPath = "/Users/yuxi/NB/RandomForest/_local/train/20201117/forest.model";
        RandomForest forest = (RandomForest) SerializationHelper.read(modelPath);
        
        File docListFile = new File("/Users/yuxi/NB/crumbs/experiment/doc_cluster/0XtBbBhm_evt_cluster_docs_fields");
        File simMatrixFile = new File(docListFile.getAbsolutePath() + "_matrix");
        BufferedReader br = new BufferedReader(new FileReader(docListFile));
        BufferedWriter bw = new BufferedWriter(new FileWriter(simMatrixFile));
        String line = null;
        ArrayNode allCandidate = mapper.createArrayNode();
        while ((line = br.readLine()) != null) {
            allCandidate.add(mapper.readTree(line.split("\t")[1]));
        }
    
        for (int i = 0; i < allCandidate.size(); i++) {
            JsonNode master = allCandidate.get(i);
            List<double[]> indexList = predictOnline(forest, master, allCandidate);
            StringBuilder sb = new StringBuilder();
            for (double[] weight : indexList) {
                sb.append(String.format("%.4f", weight[1]));
                sb.append(",");
            }
            sb.deleteCharAt(sb.length() - 1);
            bw.write(sb.toString());
            bw.write("\n");
        }
        bw.close();
    }
    
    
    /**
     * 基于指定特征构造相似度矩阵
     */
    public static void buildModelFeaMatrix() throws Exception {
        ObjectMapper mapper = new ObjectMapper();
        
        File docListFile = new File("/Users/yuxi/NB/crumbs/experiment/doc_cluster_ap_kws_overlap/0XsqzI0z_dup_cluster_docs_fields");
        File simMatrixFile = new File(docListFile.getAbsolutePath() + "_matrix");
        BufferedReader br = new BufferedReader(new FileReader(docListFile));
        BufferedWriter bw = new BufferedWriter(new FileWriter(simMatrixFile));
        String line = null;
        ArrayNode allCandidate = mapper.createArrayNode();
        while ((line = br.readLine()) != null) {
            allCandidate.add(mapper.readTree(line.split("\t")[1]));
        }
        
        for (int i = 0; i < allCandidate.size(); i++) {
            JsonNode master = allCandidate.get(i);
            List<Double> indexList = getTgtFeature(master, allCandidate);
            StringBuilder sb = new StringBuilder();
            for (double weight : indexList) {
                sb.append(String.format("%.4f", weight));
                sb.append(",");
            }
            sb.deleteCharAt(sb.length() - 1);
            bw.write(sb.toString());
            bw.write("\n");
        }
        bw.close();
    }
    
    
    /**
     * Inference ONLINE
     */
    public static void inferenceOnline() throws Exception {
        ObjectMapper mapper = new ObjectMapper();
        RandomForest forest = (RandomForest) SerializationHelper.read("/Users/yuxi/NB/RandomForest/_local/train/20201117/forest.model");
        String masterStr = "{ \"_id\" : \"0NmUZOpH\", \"c_word\" : 1110, \"channels\" : [ \"Facebook^^Inc.\", \"Social^^Media\", \"Social^^Life\", \"Facebook^^Privacy\", \"Social^^Engagement\" ], \"epoch\" : 1578459600, \"geotag\" : [], \"highlightkeyword_list\" : [ [ \"Cambridge^^Analytica\", 0.237125387901459 ], [ \"Mark^^Zuckerberg\", 0.00497122202215844 ], [ \"Robin^^Dunbar\", 0.00207098628466191 ], [ \"Facebook\", 0.000276432088162368 ], [ \"media\", 1.43794217017694e-06 ], [ \"Google\", 8.52768552463887e-07 ], [ \"Snowden\", 8.22066513865799e-07 ] ], \"insert_time\" : \"2020-01-08 11:40:13\", \"kw_title\" : [ \"Facebook\", \"social\", \"social^^media^^platforms\", \"social^^life\", \"Facebook^^accounts\", \"social^^networks\", \"social^^groups\", \"Facebook^^deletion\", \"people\" ], \"kws\" : [ \"Facebook\", \"social\", \"social^^media^^platforms\", \"social^^life\", \"Facebook^^accounts\", \"social^^networks\", \"social^^groups\", \"personal^^experiences\", \"Facebook^^deletion\", \"engagement\", \"people\", \"active^^users\", \"Google\", \"data^^privacy\", \"interesting^^trends\", \"digital^^forms\", \"behavioural^^influence\", \"smart^^technology\", \"Mark^^Zuckerberg\", \"perpetual^^social^^comparison\" ], \"ne_content_location\" : { \"US\" : 2 }, \"ne_content_organization\" : { \"Google\" : 1, \"Facebook\" : 25, \"Cambridge Analytica\" : 3 }, \"ne_content_person\" : { \"Robin Dunbar\" : 1, \"Snowden\" : 1, \"Donald Trump\" : 1, \"Mark Zuckerberg\" : 1 }, \"ne_title_location\" : {}, \"ne_title_organization\" : { \"Facebook\" : 1 }, \"ne_title_person\" : {}, \"paragraph_count\" : 22.0, \"simhash\" : \"78944e3a3416dfbe8a64cc47ced97ed2\", \"spacy_content_loc\" : [ \"US\", \"US\" ], \"spacy_content_num\" : [ \"2.45 billion\", \"approximately 32 %\", \"9 %\", \"35 %\", \"at least one\", \"150\", \"several thousand\", \"One\" ], \"spacy_content_org\" : [ \"Facebook\", \"Google\", \"Facebook\", \"Facebook\", \"Facebook\", \"Facebook\", \"Facebook\", \"Facebook\", \"Facebook\", \"Facebook\", \"Digital\", \"Facebook\", \"Dunbar\", \"Facebook\", \"Facebook\", \"Facebook\", \"Facebook\", \"Facebook\" ], \"spacy_content_per\" : [ \"Snowden\", \"Mark Zuckerberg 's\", \"Donald Trump\", \"Cambridge Analytica\", \"Cambridge Analytica\", \"Robin Dunbar\" ], \"spacy_content_tim\" : [ \"the previous month\", \"2018\" ], \"spacy_title_loc\" : [], \"spacy_title_num\" : [], \"spacy_title_org\" : [ \"Facebook\" ], \"spacy_title_per\" : [], \"spacy_title_tim\" : [], \"src\" : \"The Conversation\", \"stitle\" : \"Why people leave Facebook -- and what it tells us about the future of social media\", \"text_category\" : {}, \"url\" : \"https://theconversation.com/why-people-leave-facebook-and-what-it-tells-us-about-the-future-of-social-media-128952\" }";
        String canditStr =
            "[" +
                "{ \"_id\" : \"0XZqO2ay\", \"c_word\" : 362, \"channels\" : [ \"Facebook^^Inc.\", \"Free^^People\" ], \"channels_v2\" : [ \"Internet\", \"Censorship\" ], \"dup_id\" : \"0XZqO2ay\", \"epoch\" : 1604937493, \"evt_id\" : \"0XZqO2ay\", \"geotag\" : [], \"geotag_v2\" : [], \"highlightkeyword_list\" : [ [ \"Leave^^Facebook\", 0.00659497546488851 ] ], \"insert_time\" : \"2020-11-09 19:10:00\", \"kw_title\" : [ \"Leave^^Facebook\", \"Facebook\", \"people\" ], \"kws\" : [ \"Leave^^Facebook\", \"Facebook\", \"people\", \"social\", \"slander\", \"censorship\", \"includes^^blood^^relatives\", \"defamation\", \"time\", \"home\", \"victim\" ], \"nbr_id\" : \"0XZqO2ay\", \"ne_content_location\" : {}, \"ne_content_organization\" : { \"Facebook\" : 4 }, \"ne_content_person\" : {}, \"ne_title_location\" : {}, \"ne_title_organization\" : { \"Leave Facebook\" : 1 }, \"ne_title_person\" : {}, \"paragraph_count\" : 6.0, \"simhash\" : \"ce0323f8ec0319206ff9e1678fd57bd9\", \"spacy_content_loc\" : [], \"spacy_content_num\" : [ \"a dozen\" ], \"spacy_content_org\" : [ \"Facebook\", \"Facebook\" ], \"spacy_content_per\" : [], \"spacy_content_tim\" : [], \"spacy_title_loc\" : [], \"spacy_title_num\" : [], \"spacy_title_org\" : [], \"spacy_title_per\" : [], \"spacy_title_tim\" : [], \"src\" : \"FMX 94.5\", \"stitle\" : \"Are People Really Going to Leave Facebook ?\", \"text_category\" : { \"first_cat\" : { \"TechnologyElectronics\" : 0.815229921585327 }, \"second_cat\" : { \"TechnologyElectronics_Internet\" : 0.764268210077496 }, \"third_cat\" : { \"TechnologyElectronics_Internet_Other\" : 0.764268210077496 } }, \"text_category_v2\" : { \"first_cat\" : { \"TechnologyElectronics\" : 0.815229921585327 }, \"second_cat\" : { \"TechnologyElectronics_Internet\" : 0.764268210077496 }, \"third_cat\" : { \"TechnologyElectronics_Internet_Other\" : 0.764268210077496 } }, \"url\" : \"https://kfmx.com/are-people-really-going-to-leave-facebook/\" }" +
                "]";
        List<double[]> indexList = predictOnline(forest, mapper.readTree(masterStr), mapper.readTree(canditStr));
        for (double[] weight : indexList) {
            System.out.println(String.valueOf(weight[0]) + "\t" + String.valueOf(weight[1]));
        }
    }
    
    
    public static void main(String[] args) throws Exception {
        boolean doTrain = false;
        boolean doCompare = true;
        
        /** Model Training & Evaluation **/
        String rootDir = "/mnt/nlp/albert/clustering/data/dedup_train_data_v3/";
        if (doTrain) {
            String trainARFFPath = Paths.get(rootDir, "train.arff").toString();
            //    String testARFFPath = Paths.get(rootDir, "test.arff").toString();
            Instances trainingDataSet = getDataSet(trainARFFPath);
            trainingDataSet.setClassIndex(trainingDataSet.numAttributes() - 1);
            //    Instances testingDataSet = getDataSet(testARFFPath);
            //    testingDataSet.setClassIndex(trainingDataSet.numAttributes() - 1);
            trainModel(rootDir, trainingDataSet, 100, 12, 12);
        }
        
        /** Model Inference STD Estimate */
        if (doCompare) {
            String abtestModelPath = rootDir + "forest.model";
            RandomForest abtestForest = (RandomForest) SerializationHelper.read(abtestModelPath);
            // abtestForest.setPrintClassifiers(true);
            // abtestForest.setComputeAttributeImportance(false);
            // BufferedWriter bw = new BufferedWriter(new FileWriter(new File(rootDir + "forest.out")));
            // bw.write(abtestForest.toString());
            // bw.close();
            

            // String estimateDataPath = rootDir + "test_fields";
            String estimateDataPath = "/mnt/nlp/albert/clustering/data/dedup_train_data/evt_test_fields";
            predictEstimateDataFeatureV3(abtestForest, estimateDataPath, true);
            System.out.println("==========================");
        

    //        String esLocal = "/Users/yuxi/NB/RandomForest/_local/estimate/estimate_doc_pair_fields_local";
    //        String esNonlocal = "/Users/yuxi/NB/RandomForest/_local/estimate/estimate_doc_pair_fields_nonlocal";
    //        predictEstimateDataFeatureV1(onlineForest, esLocal);
    //        System.out.println("==========================");
    //        predictEstimateDataFeatureV1(onlineForest, esNonlocal);
        }
    }
    
    public static void main2(String[] args) throws Exception {
//        buildModelSimMatrix();
        buildModelFeaMatrix();
    }
}
