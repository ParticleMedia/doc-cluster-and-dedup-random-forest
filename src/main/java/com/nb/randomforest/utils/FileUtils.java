package com.nb.randomforest.utils;

import static com.mongodb.client.model.Filters.*;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.apache.commons.lang3.StringUtils;
import org.apache.http.HttpEntity;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.ContentType;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClientBuilder;
import org.bson.Document;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.mongodb.BasicDBObject;
import com.mongodb.client.MongoClient;
import com.mongodb.client.MongoClients;
import com.mongodb.client.MongoCollection;
import com.mongodb.client.MongoCursor;
import com.mongodb.client.MongoDatabase;

/**
 * @author yuxi
 * @date 2020/9/28
 * @note Generate doc training data from doc pair or doc fields.
 */
public class FileUtils {
	
	public static ObjectMapper objectMapper = new ObjectMapper();
	
	public static MongoCollection<Document> collectionStatic;
	
	public static MongoCollection<Document> collectionCenter;
	
	public static BasicDBObject fields = new BasicDBObject();
	
	static {
		String uriStatic = "mongodb://mongos.content.content:" + System.getenv("CONTENT_MONGO_PASSWORD") + "@content-mongos.mongos.nb.com/?readPreference=secondaryPreferred&authSource=admin";
		MongoClient clientStatic = MongoClients.create(uriStatic);
		MongoDatabase dbStatic = clientStatic.getDatabase("staticFeature");
		collectionStatic = dbStatic.getCollection("document");
		
		String uriCenter = "mongodb://172.31.16.143/docenter";
		MongoClient clientCenter = MongoClients.create(uriCenter);
		MongoDatabase dbCenter = clientCenter.getDatabase("docenter");
		collectionCenter = dbCenter.getCollection("document");
		
		fields.append("_id", 1)
			.append("stitle", 1).append("seg_title", 1)
			.append("src", 1)
			.append("c_word", 1)
			.append("paragraph_count", 1)
			.append("epoch", 1).append("insert_time", 1)
			.append("simhash", 1)
			.append("kws", 1).append("kw_title", 1).append("highlightkeyword_list", 1)
			.append("channels", 1).append("channels_v2", 1)
			// Content : ORG, LOC, PER, NUM, TIM
			.append("ne_content_organization", 1).append("spacy_content_org", 1)
			.append("ne_content_location", 1).append("spacy_content_loc", 1)
			.append("ne_content_person", 1).append("spacy_content_per", 1)
			.append("spacy_content_num", 1)
			.append("spacy_content_tim", 1)
			// Title   : ORG, LOC, PER, NUM, TIM
			.append("ne_title_organization", 1).append("spacy_title_org", 1)
			.append("ne_title_location", 1).append("spacy_title_loc", 1)
			.append("ne_title_person", 1).append("spacy_title_per", 1)
			.append("spacy_title_num", 1)
			.append("spacy_title_tim", 1)
			.append("text_category", 1).append("text_category_v2", 1)
			.append("geotag", 1).append("geotag_v2", 1)
			.append("url", 1)
			.append("images_phash", 1).append("faces_phash", 1);
	}
	
	
	/**
	 * 基于 doc pair 从 static_feature 或 doc_center 中抽取目标属性
	 * 已标注数据 : doc_id \t doc_id |t label => doc_id \t doc_id |t label \t isSuccess \t jstr \t jstr
	 * 未标注数据 : doc_id \t doc_id          => doc_id \t doc_id          |t isSuccess \t jstr \t jstr
	 */
	public static void dumpDocFieldsFromDocPairFile(File docPair) throws Exception {
		BufferedReader br = new BufferedReader(new FileReader(docPair));
		BufferedWriter bw = new BufferedWriter(new FileWriter(new File(docPair.getAbsolutePath() + "_fields")));
		String line = null;
		List<String[]> labelPair = new ArrayList<>();
		Set<String> idCache = new HashSet<>();
		Map<String, Document> idMap = new HashMap<>();
		while ((line = br.readLine()) != null) {
			String[] datas = line.split("\t");
			String docIDm = datas[0];
			String docIDc = datas[1];
			labelPair.add(datas);
			idCache.add(docIDm);
			idCache.add(docIDc);
		}
		MongoCursor<Document> cursor = collectionStatic.find(in("_id", idCache.toArray())).projection(fields).cursor();
		while (cursor.hasNext()) {
			Document d = cursor.next();
			String id = d.getString("_id");
			idMap.put(id, d);
		}
		for (String[] strings : labelPair) {
			String docIDm = strings[0];
			String docIDc = strings[1];
			String label = strings.length > 2 ? strings[2] : "NONE";
			Document docMaster = idMap.get(docIDm);
			Document docCandit = idMap.get(docIDc);
			bw.write(docIDm + "\t" + docIDc + "\t" + label);
			if (docMaster != null && docCandit != null) {
				bw.write("\t");
				bw.write("SUCCESS");
				bw.write("\t");
				bw.write(docMaster.toJson());
				bw.write("\t");
				bw.write(docCandit.toJson());
				bw.write("\n");
			} else {
				bw.write("\t");
				bw.write("FAILED");
				bw.write("\n");
			}
		}
		bw.close();
	}
	
	
	public static void dumpDocFieldsFromDocListFile(File docListFile) throws Exception {
		BufferedReader br = new BufferedReader(new FileReader(docListFile));
		BufferedWriter bw = new BufferedWriter(new FileWriter(new File(docListFile.getAbsolutePath() + "_fields")));
		String line = null;
		List<String> idList = new ArrayList<>();
		Set<String> idCache = new HashSet<>();
		Map<String, Document> idMap = new HashMap<>();
		while ((line = br.readLine()) != null) {
			String docID = line.trim();
			idCache.add(docID);
			idList.add(docID);
		}
		MongoCursor<Document> cursor = collectionStatic.find(in("_id", idCache.toArray())).projection(fields).cursor();
		while (cursor.hasNext()) {
			Document d = cursor.next();
			String id = d.getString("_id");
			idMap.put(id, d);
		}
		for (String docID : idList) {
			Document feature = idMap.get(docID);
			bw.write(docID + "\t" + (feature != null ? feature.toJson() : ""));
			bw.write("\n");
		}
		bw.close();
	}
	
	
	/**
	 * 基于 doc pair + mongo db 构建标注用例
	 * doc + doc + title + title + url + url
	 */
	public static void buildLabelDataFromDBByDocPair() throws Exception {
		String input = "/Users/yuxi/NB/crumbs/experiment/doc_clu_0118_caption/video_web_cap_pair";
		String output = "/Users/yuxi/NB/crumbs/experiment/doc_clu_0118_caption/video_web_cap_pair_unlabeled.tsv";
		
		BufferedReader br = new BufferedReader(new FileReader(new File(input)));
		String line = null;
		Set<String> ids = new HashSet<>();
		Map<String, Document> idMap = new HashMap<>();
		List<String[]> lines = new ArrayList<>();
		while ((line = br.readLine()) != null) {
			if (line.startsWith("\uFEFF")) {
				line = line.replace("\uFEFF", "");
			}
			String[] datas = line.split("\t");
			ids.add(datas[0]);
			ids.add(datas[1]);
			lines.add(datas);
		}
		MongoCursor<Document> cursor = collectionStatic.find(in("_id", ids.toArray())).projection(new BasicDBObject().append("url", 1).append("stitle", 1)).cursor();
		while (cursor.hasNext()) {
			Document d = cursor.next();
			String id = d.getString("_id");
			idMap.put(id, d);
		}
		BufferedWriter bw = new BufferedWriter(new FileWriter(new File(output)));
		for (String[] strs : lines) {
			String docM = strs[0];
			String docC = strs[1];
			String info = strs[2];
			try {
				bw.write(docM + "\t" + docC + "\t" + info + "\t" +
					idMap.get(docM).getString("stitle") + "\t" + idMap.get(docC).getString("stitle") + "\t" +
					idMap.get(docM).getString("url") + "\t" + idMap.get(docC).getString("url") + "\n"
				);
			} catch (Exception e) {
				e.printStackTrace();
				System.out.println(strs);
			}
		}
		bw.close();
	}
	
	
	/**
	 * 基于 #未标注数据# 的属性构造待标注文档对
	 * @input :  doc_id \t doc_id \t isSuccess \t jstr |t jstr
	 * @output : m_id \t c_id \t isFailed \t m_url \t c_url
	 */
	public static void extractDocUrlFromUnlabeledDocPair(File docPair) throws Exception {
		BufferedReader br = new BufferedReader(new FileReader(docPair));
		BufferedWriter bw = new BufferedWriter(new FileWriter(new File(docPair.getAbsolutePath() + "_urls")));
		String line = null;
		while ((line = br.readLine()) != null) {
			String[] datas = line.split("\t");
			String mID = datas[0];
			String cID = datas[1];
			String isFailed = datas[2];
			String mFeature = datas[3];
			String cFeature = datas[4];
			JsonNode mNode = objectMapper.readTree(mFeature);
			JsonNode cNode = objectMapper.readTree(cFeature);
			if ("SUCCESS".equals(isFailed)) {
				bw.write(mID);
				bw.write("\t");
				bw.write(cID);
				bw.write("\t");
				bw.write(mNode.get("url").asText());
				bw.write("\t");
				bw.write(cNode.get("url").asText());
				bw.write("\n");
			}
		}
		bw.close();
	}
	
	
	/**
	 * 无标签 Field + Label Pair => 标准 Field 文档
	 * unlabeled_doc_pair_fields + labeled_doc_pair => std_doc_pair_fields
	 */
	public static void concatFieldsWithLabel(String sourcePath) {
		try {
			File sourceFile = new File(sourcePath);
			BufferedReader unlabelFieldReader = new BufferedReader(new FileReader(sourceFile));
			BufferedReader labelPairReader = new BufferedReader(new FileReader(new File(sourceFile.getAbsolutePath() + "_label")));
			BufferedWriter bw = new BufferedWriter(new FileWriter(new File(sourceFile.getAbsolutePath() + "_fields")));
			String line = null;
			Map<String, String> docPair2Field = new HashMap<>();
			while ((line = unlabelFieldReader.readLine()) != null) {
				String[] datas = line.split("\t");
				String docM = datas[0];
				String fieldM = datas[3];
				String docC = datas[1];
				String fieldC = datas[4];
				String key = docM + "\t" + docC;
				String value = fieldM + "\t" + fieldC;
				if (docPair2Field.containsKey(key)) {
					System.out.println(key);
				}
				docPair2Field.put(key, value);
			}
			
			while ((line = labelPairReader.readLine()) != null) {
				String[] datas = line.split("\t");
				String docM = datas[0];
				String docC = datas[1];
				String label = datas[2];
				String key = docM + "\t" + docC;
				if (docPair2Field.containsKey(key)) {
					bw.write(key);
					bw.write("\t");
					bw.write(label);
					bw.write("\t");
					bw.write("SUCCESS");
					bw.write("\t");
					bw.write(docPair2Field.get(key));
					bw.write("\n");
				} else {
					System.out.println("NotIn : " + key);
				}
			}
			bw.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
		
	}
	
	
	/**
	 * 基于标注数据结果 生成 shuffled train_fields, test_fields
	 */
	public static void buildModelData(List<String> paths, String targetDir) throws Exception {
		String trainFieldsPath = Paths.get(targetDir, "train_fields").toString();
		String testFieldsPath = Paths.get(targetDir, "test_fields").toString();
		BufferedWriter trainFieldsWriter = new BufferedWriter(new FileWriter(new File(trainFieldsPath)));
		BufferedWriter testFieldsWriter = new BufferedWriter(new FileWriter(new File(testFieldsPath)));
		Random rnd = new Random();
		
		for (String path : paths) {
			BufferedReader br = new BufferedReader(new FileReader(new File(path)));
			String line = null;
			while ((line = br.readLine()) != null) {
				String[] datas = line.split("\t");
				String label = datas[2];
				String isSuccess = datas[3];
				if (!label.equals("DIFF") && !label.equals("EVENT") && !label.equals("DUP") || !isSuccess.equals("SUCCESS")) {
					System.out.println(line);
					continue;
				}
				if (rnd.nextInt(10) < 1) {
					testFieldsWriter.write(line);
					testFieldsWriter.write("\n");
				} else {
					trainFieldsWriter.write(line);
					trainFieldsWriter.write("\n");
				}
			}
		}
		
		testFieldsWriter.close();
		trainFieldsWriter.close();
	}
	
	
	/**
	 * 区分 local nonlocal, 分别写入不同文件
	 * @throws Exception
	 */
	public static void calDocPairIsLocal() throws Exception {
		BufferedReader br = new BufferedReader(new FileReader(new File("/Users/yuxi/NB/RandomForest/_local/estimate/estimate_doc_pair_fields")));
		BufferedWriter bwLocal = new BufferedWriter(new FileWriter(new File("/Users/yuxi/NB/RandomForest/_local/estimate/estimate_doc_pair_fields_local")));
		BufferedWriter bwNonlocal = new BufferedWriter(new FileWriter(new File("/Users/yuxi/NB/RandomForest/_local/estimate/estimate_doc_pair_fields_nonlocal")));
		String line = null;
		while ((line = br.readLine()) != null) {
			String[]  datas = line.split("\t");
			String mID = datas[0];
			JsonNode result = get("http://locallikelihood.ha.svc.k8sc1.nb.com:8923/debug?docid=" + mID);
			double localScore = result.get("result").get("local_score").asDouble();
			if (localScore > 0.5) {
				bwLocal.write(line);
				bwLocal.write("\n");
			} else {
				bwNonlocal.write(line);
				bwNonlocal.write("\n");
			}
		}
		bwLocal.close();
		bwNonlocal.close();
	}
	
	
	private static void buildABResultDataFromInfoLog() throws Exception {
		BufferedReader respReader = new BufferedReader(new FileReader(new File("/Users/yuxi/NB/RandomForest/_local/estimate/20201023/response.txt")));
		BufferedReader modelReader = new BufferedReader(new FileReader(new File("/Users/yuxi/NB/RandomForest/_local/estimate/20201023/model_result.txt")));
		String line = null;
		// master_candit : lr_label, rf_label
		Map<String, String[]> modelResult = new HashMap<>();
		while ((line = respReader.readLine()) != null) {
			JsonNode result = objectMapper.readTree(line.split("  ")[1]);
			JsonNode resp = result.get("RESPONSE");
			String docID = resp.get("_id").textValue();
			String nbrID = resp.get("nbr_id").textValue();
			String[] strs = new String[2];
			if (resp.get("dup_dist").doubleValue() > -1) {
				strs[0] = "DUP";
			} else if (resp.get("evt_dist").doubleValue() > -1) {
				strs[0] = "EVENT";
			} else {
				strs[0] = "DIFF";
			}
			modelResult.put(docID + "\t" + nbrID, strs);
		}
		String lastID = "";
		String lastNB = "";
		String lastLabel = "DIFF";
		double lastScore = -1;
		int count = 0;
		while ((line = modelReader.readLine()) != null) {
			String[] datas = line.split("\t");
			String docID = datas[1];
			String nbrID = datas[2];
			String label = datas[3];
			Double score = Double.valueOf(datas[4]);
			if (!docID.equals(lastID)) {
				count++;
				//dump上一步结果
				if (!lastID.equals(lastNB) || !lastLabel.equals("DIFF")) {
					String key = lastID + "\t" + lastNB;
					if (modelResult.containsKey(key)) {
						modelResult.get(key)[1] = lastLabel;
					} else {
						String[] strs = new String[2];
						strs[1] = lastLabel;
						modelResult.put(key, strs);
					}
				}
				//刷新上一步缓存
				lastID = docID;
				lastNB = docID;
				lastLabel = "DIFF";
				lastScore = -1;
			}
			//与上一步结果比较
			if (label.equals("EVENT") && score > lastScore) {
				if (score > 0.9) {
					lastNB = nbrID;
					lastLabel = "DUP";
					lastScore = score;
				} else if (score > 0.58) {
					lastNB = nbrID;
					lastLabel = "EVENT";
					lastScore = score;
				}
			}
		}
		
		for (Map.Entry<String, String[]> entry : modelResult.entrySet()) {
			System.out.println(entry.getKey() + "\t" + String.valueOf(entry.getValue()[0]) + "\t" + String.valueOf(entry.getValue()[1]));
		}
		System.out.println(count);
	}
	
	
	private static void buildLabelDataFromDBByResponseLog() throws Exception {
		String logPath = "/Users/yuxi/NB/RandomForest/_local/append_1105~1107/response_shuf.txt";
		String labelPath = "/Users/yuxi/NB/RandomForest/_local/append_1105~1107/label_doc_pair.txt";
		
		BufferedReader br = new BufferedReader(new FileReader(new File(logPath)));
		String line = null;
		Set<String> ids = new HashSet<>();
		Map<String, Document> idMap = new HashMap<>();
		List<String[]> lines = new ArrayList<>();
		while ((line = br.readLine()) != null) {
			String logString = line.split("  ")[1];
			JsonNode logNode = objectMapper.readTree(logString);
			JsonNode respNode = logNode.get("RESPONSE");
			String[] datas = new String[2];
			datas[0] = respNode.get("_id").textValue();
			datas[1] = respNode.get("nbr_id").textValue();
			ids.add(datas[0]);
			ids.add(datas[1]);
			lines.add(datas);
		}
		MongoCursor<Document> cursor = collectionStatic.find(in("_id", ids.toArray())).projection(new BasicDBObject().append("url", 1).append("stitle", 1)).cursor();
		while (cursor.hasNext()) {
			Document d = cursor.next();
			String id = d.getString("_id");
			idMap.put(id, d);
		}
		
		BufferedWriter bw = new BufferedWriter(new FileWriter(new File(labelPath)));
		for (String[] strs : lines) {
			String docM = strs[0];
			String docC = strs[1];
			String titleM = idMap.get(docM).getString("stitle");
			String titleC = idMap.get(docC).getString("stitle");
			if (!StringUtils.equals(docM, docC) && !StringUtils.equals(titleM, titleC)) {
				bw.write(docM + "\t" + docC + "\t" + titleM + "\t" + titleC + "\t" +
					idMap.get(docM).getString("url") + "\t" + idMap.get(docC).getString("url") + "\n"
				);
			}
		}
		bw.close();
	}
	
	
	/**
	 *
	 * @throws Exception
	 */
	public static void dumpDocFieldsFromDocPairFiles() throws Exception {
		List<File> files = new ArrayList<>();
//		files.add(new File("/Users/yuxi/NB/RandomForest/_local/append_0720~0920/doc_pair_0720~0920"));
//		files.add(new File("/Users/yuxi/NB/RandomForest/_local/append_0720~0920/doc_pair_0903"));
//		files.add(new File("/Users/yuxi/NB/RandomForest/_local/append_0720~0920/doc_pair_0911"));
//		files.add(new File("/Users/yuxi/NB/RandomForest/_local/append_0720~0920/doc_pair_0914"));
//		files.add(new File("/Users/yuxi/NB/RandomForest/_local/append_0926~0930/dp_0926_0930_celebrities"));
//		files.add(new File("/Users/yuxi/NB/RandomForest/_local/append_0926~0930/dp_0926_0930_economic"));
//		files.add(new File("/Users/yuxi/NB/RandomForest/_local/append_0926~0930/dp_0926_0930_event"));
//		files.add(new File("/Users/yuxi/NB/RandomForest/_local/append_0926~0930/dp_0926_0930_royal"));
//		files.add(new File("/Users/yuxi/NB/RandomForest/_local/append_0926~0930/dp_0926_0930_tech"));
//		files.add(new File("/Users/yuxi/NB/RandomForest/_local/estimate/estimate_doc_pair"));
//		files.add(new File("/Users/yuxi/NB/RandomForest/_local/train/20210303/train"));
		files.add(new File("/mnt/nlp/albert/clustering/data/dedup_train_data_v3/train"));
		files.add(new File("/mnt/nlp/albert/clustering/data/dedup_train_data_v3/test"));
		for (File file : files) {
			dumpDocFieldsFromDocPairFile(file);
		}
	}
	
	
	public static JsonNode get(String uri) {
		try {
			CloseableHttpClient client = HttpClientBuilder.create().build();
			HttpGet httpGet = new HttpGet(uri);
			CloseableHttpResponse httpResponse = client.execute(httpGet);
			BufferedReader reader = new BufferedReader(new InputStreamReader(httpResponse.getEntity().getContent()));
			String inputLine;
			StringBuilder response = new StringBuilder();
			while ((inputLine = reader.readLine()) != null) {
				response.append(inputLine);
			}
			reader.close();
			client.close();
			return objectMapper.readValue(response.toString(), JsonNode.class);
		} catch (Exception e) {
			e.printStackTrace();
			return null;
		}
	}
	
	public static JsonNode post(String uri, JsonNode params) {
		try {
			CloseableHttpClient client = HttpClientBuilder.create().build();
			HttpPost httpPost = new HttpPost(uri);
			HttpEntity entity = new StringEntity(objectMapper.writeValueAsString(params), ContentType.APPLICATION_JSON);
			httpPost.setEntity(entity);
			CloseableHttpResponse httpResponse = client.execute(httpPost);
			BufferedReader reader = new BufferedReader(new InputStreamReader(httpResponse.getEntity().getContent()));
			String inputLine;
			StringBuilder response = new StringBuilder();
			while ((inputLine = reader.readLine()) != null) {
				response.append(inputLine);
			}
			reader.close();
			client.close();
			return objectMapper.readValue(response.toString(), JsonNode.class);
		} catch (Exception e) {
			e.printStackTrace();
			return null;
		}
	}
	
	public static void main(String[] args) throws Exception {
//		buildLabelDataFromDBByDocPair();
		dumpDocFieldsFromDocPairFiles();
//		buildLabelDataFromDBByDocPair();
//		dumpDocFieldsFromDocListFile(new File("/Users/yuxi/NB/crumbs/experiment/doc_cluster/0XtBbBhm_evt_cluster_docs"));
	}
}
