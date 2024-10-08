package com.nb.randomforest.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.nb.randomforest.entity.EventFeature;
import com.nb.randomforest.entity.resource.RFModelResult;
import com.nb.randomforest.utils.MyAttributeBuilder;
import lombok.extern.slf4j.Slf4j;
import biz.k11i.xgboost.Predictor;
import biz.k11i.xgboost.util.FVec;
import org.apache.commons.lang3.BooleanUtils;
import org.apache.commons.lang3.StringUtils;
import org.ejml.data.DMatrix;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Service;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.Instance;

import java.util.*;
import java.util.concurrent.*;

/**
 * @author yuxi
 * @date 2020/10/9 
 */
@Service
@Slf4j
public class DocumentServiceMultiThread {
	
	@Autowired
	ObjectMapper mapper;
	
	@Autowired
	@Qualifier("DupXGBoost")
	Predictor dupXGBooster;
	
	@Autowired
	@Qualifier("DupRandomForest")
	RandomForest dupRandomForest;
	
	@Autowired
	@Qualifier("EvtRandomForest")
	RandomForest evtRandomForest;
	
	@Autowired
	StorageService storageService;


	ExecutorService workers = Executors.newFixedThreadPool(8);


	public void checkTextCategory(ObjectNode masterInfo, JsonNode masterNode) {
		try {
			if (masterNode.hasNonNull("text_category") && masterNode.get("text_category").hasNonNull("second_cat")) {
				JsonNode firstCatNode = masterNode.get("text_category").get("first_cat");
				Iterator<String> itrFirst = firstCatNode.fieldNames();
				while (itrFirst.hasNext()) {
					String key = itrFirst.next();
					if (StringUtils.equals("Sports", key)){
						masterInfo.set("isSports", mapper.convertValue(true, JsonNode.class));
					}
				}
				
				JsonNode secondCatNode = masterNode.get("text_category").get("second_cat");
				Iterator<String> itrSecond = secondCatNode.fieldNames();
				while (itrSecond.hasNext()) {
					String key = itrSecond.next();
					if (StringUtils.equals("BusinessEconomy_Markets", key)){
						masterInfo.set("isEconomyMarkets", mapper.convertValue(true, JsonNode.class));
					} else if (StringUtils.equals("ArtsEntertainment_Celebrities", key)) {
						masterInfo.set("isCelebrities", mapper.convertValue(true, JsonNode.class));
					} else if (StringUtils.equals("ClimateEnvironment_Weather", key)) {
						masterInfo.set("isWeather", mapper.convertValue(true, JsonNode.class));
					}
				}
			}
		} catch (Exception e) {
			log.info("EXCEPTION : checkTextCategory : " + e.getMessage(), e);
		}
	}

	public void checkFauci(ObjectNode masterInfo, JsonNode masterNode) {
		try {
			String title = masterNode.hasNonNull("seg_title") ? masterNode.get("seg_title").textValue() :
				masterNode.hasNonNull("stitle") ? masterNode.get("stitle").textValue() : "";
			title = title.toLowerCase();
			if (title.contains("fauci says")) {
				masterInfo.set("aboutFauci", mapper.convertValue(true, JsonNode.class));
			}
		} catch (Exception e) {
			log.info("EXCEPTION : checkFauci : " + e.getMessage(), e);
		}
	}

	public ObjectNode getMasterInfo(JsonNode masterNode) {
		ObjectNode masterInfo = mapper.createObjectNode();
		String masterID = masterNode.hasNonNull("_id") ? masterNode.get("_id").textValue() : "";
		masterInfo.set("masterID", mapper.convertValue(masterID, JsonNode.class));
		masterInfo.set("isCelebrities", mapper.convertValue(false, JsonNode.class));
		masterInfo.set("isEconomyMarkets", mapper.convertValue(false, JsonNode.class));
		masterInfo.set("isSports", mapper.convertValue(false, JsonNode.class));
		masterInfo.set("isWeather", mapper.convertValue(false, JsonNode.class));
		masterInfo.set("aboutFauci", mapper.convertValue(false, JsonNode.class));
		try {
			checkTextCategory(masterInfo, masterNode);
			checkFauci(masterInfo, masterNode);
			return masterInfo;
		} catch (Exception e) {
			log.info("EXCEPTION : getMasterInfo : " + e.getMessage(), e);
			return mapper.createObjectNode();
		}
	}

	public double regularizeScore(String label, double score) {
		if (label.equals("DUP")) {
			return 0.8+score*0.2;
		} else {
			return score*0.8;
		}
	}

	public void runXGBClassification(RFModelResult result,
								     String candidateID,
									 double[] feats,
									 ObjectNode masterInfo) {
		try {
			FVec fVec = FVec.Transformer.fromArray(feats, false);
			float[] predicts = dupXGBooster.predict(fVec);
			double difScr = predicts[0];
			double evtScr = predicts[1];
			double dupScr = predicts[2];
			// double score = Math.max(difScr, Math.max(evtScr, dupScr));
			log.info("XGB MODEL DEBUG: {}\t{}\t{}\t{}\t{}", masterInfo.get("masterID").textValue(), candidateID, difScr, evtScr, dupScr);
			if (difScr >= evtScr && difScr >= dupScr) {
				result.setLabel("DIFF");
				result.setScore(difScr);
			} else if (evtScr >= difScr && evtScr >= dupScr) {
				result.setLabel("EVENT");
				result.setScore(evtScr);
			} else if (dupScr >= evtScr && dupScr >= difScr) {
				result.setLabel("DUP");
				result.setScore(dupScr);
			} else {
				result.setLabel("DIFF");
				result.setScore(difScr);
			}
		} catch (Exception e) {
			log.info("EXCEPTION : runDupClassification : " + e.getMessage(), e);
		}
	}

	public void runDupClassification(RFModelResult result,
								     String candidateID,
									 EventFeature feature,
									 Instance instance,
									 ObjectNode masterInfo) {
		try {
			double score = dupRandomForest.distributionForInstance(instance)[1];
			log.info(String.format(
				"DUP MODEL DEBUG: %s\t%s\t%.5f", masterInfo.get("masterID").textValue(), candidateID, score));
			if (score > 0.58) {
				result.setLabel("DUP");
				result.setScore(regularizeScore("DUP", score));
			} else {
				result.setLabel("DIFF");
			}
		} catch (Exception e) {
			log.info("EXCEPTION : runDupClassification : " + e.getMessage(), e);
		}
	}

	public boolean nearDup(ObjectNode masterInfo, double evtScore, EventFeature feature) {
		if (masterInfo.get("isEconomyMarkets").asBoolean() && evtScore > 0.98) return true;
		if (masterInfo.get("isCelebrities").asBoolean() && evtScore > 0.95) return true;
		if (!masterInfo.get("isEconomyMarkets").asBoolean() && !masterInfo.get("isCelebrities").asBoolean()
				&& !masterInfo.get("aboutFauci").asBoolean() && evtScore > 0.9) {
			return true;
		}
		if (masterInfo.get("isWeather").asBoolean() && feature.getTitleRatio() >= 0.7
				&& feature.getTitleLength() >= 5 && evtScore > 0.85) {
			return true;
		}
		if (masterInfo.get("isSports").asBoolean() && feature.getTitleRatio() >= 0.5
				&& feature.getTitleLength() >= 5 && evtScore > 0.85) {
			return true;
		}
		if (!masterInfo.get("aboutFauci").asBoolean() && !masterInfo.get("isSports").asBoolean()
				&& !masterInfo.get("isWeather").asBoolean() && feature.getTitleRatio() >= 0.65
				&& feature.getTitleLength() >= 5 && evtScore > 0.75) {
			return true;
		}
		if (!masterInfo.get("aboutFauci").asBoolean() && !masterInfo.get("isSports").asBoolean()
				&& !masterInfo.get("isWeather").asBoolean() && feature.getSimhashDist() != null
				&& feature.getSimhashDist() < 5 && evtScore > 0.8) {
			return true;
		}
		return false;
	}

	public boolean isEvent(ObjectNode masterInfo, double evtScore, EventFeature feature) {
		if (masterInfo.get("isEconomyMarkets").asBoolean() && evtScore > 0.55) return true;
		if (masterInfo.get("isSports").asBoolean() && evtScore > 0.6) return true;
		if (masterInfo.get("isWeather").asBoolean() && evtScore > 0.6 
				&& feature.getGeoRatio() != null && feature.getGeoRatio() > 0.5) {
			return true;
		}
		if (masterInfo.get("isWeather").asBoolean() && evtScore > 0.8 && feature.getGeoRatio() == null) return true;
		if (masterInfo.get("aboutFauci").asBoolean() && evtScore > 0.85) return true;
		if (!masterInfo.get("isEconomyMarkets").asBoolean() && !masterInfo.get("isSports").asBoolean()
				&& !masterInfo.get("isWeather").asBoolean() && !masterInfo.get("aboutFauci").asBoolean()&& evtScore > 0.44) {
			return true;
		}
		return false;
	}

	public void runEvtClassification(RFModelResult result,
									 String candidateID,
									 EventFeature feature,
									 Instance instance,
									 ObjectNode masterInfo) {
		try {
			double score = evtRandomForest.distributionForInstance(instance)[1];

			if (result.getLabel().equals("DUP")) return;

			log.info(String.format(
				"EVT MODEL DEBUG: %s\t%s\t%.5f", masterInfo.get("masterID").textValue(), candidateID, score));

			String label; double scoreFinal;
			if (nearDup(masterInfo, score, feature)) {
				label = "DUP";
				scoreFinal = 0.8;
			} else if (isEvent(masterInfo, score, feature)) {
				label = "EVENT";
				scoreFinal = regularizeScore("EVENT", score);
			} else {
				label = "DIFF";
				scoreFinal = regularizeScore("DIFF", score);
			}

			result.setLabel(label);
			result.setScore(scoreFinal);
		} catch (Exception e) {
			log.info("EXCEPTION : runEvtClassification : " + e.getMessage(), e);
		}
	}

	/**
	 *
	 */
	public List<RFModelResult> calCandidatesClusterInfo(JsonNode masterNode, JsonNode candidateNodes, Boolean isDebug, String version) {
		try {
			version = version == null ? "v2" : version;
			final String versionF = version;
			ObjectNode masterInfo = getMasterInfo(masterNode);

			ArrayList<Attribute> attributes = MyAttributeBuilder.buildMyAttributesV1();
			Instances dataset = new Instances(UUID.randomUUID().toString(), attributes, 1);
			dataset.setClassIndex(dataset.numAttributes() - 1);

			Collection<Callable<RFModelResult>> tasks = new ArrayList<Callable<RFModelResult>>();
			for (JsonNode candidateNode : candidateNodes) {
				tasks.add(new Callable<RFModelResult>() {
					public RFModelResult call() {
						String candidateID = candidateNode.hasNonNull("_id") ? candidateNode.get("_id").textValue() : "";
						try {
							EventFeature feature = new EventFeature(masterNode, candidateNode, null);
							RFModelResult result =
								new RFModelResult(candidateID, "DIFF", 0.0, BooleanUtils.isTrue(isDebug) ? feature : null);
							if (versionF.equals("xgb")) {
								double[] feats = feature.toArrayV3();
								runXGBClassification(result, candidateID, feats, masterInfo);
							} else {
								Instance instance = feature.toInstanceV1();
								instance.setDataset(dataset);
								runDupClassification(result, candidateID, feature, instance, masterInfo);
								runEvtClassification(result, candidateID, feature, instance, masterInfo);
							}
							return result;
						} catch (Exception e) {
							log.info("EXCEPTION : cal model score : " + e.getMessage(), e);
							return new RFModelResult(candidateID, "DIFF", 0.0, null);
						}
					}
				});
			}

			List<RFModelResult> results = new ArrayList<RFModelResult>();
			List<Future<RFModelResult>> futures = workers.invokeAll(tasks, 10, TimeUnit.SECONDS);
			for (Future<RFModelResult> f : futures) {
				results.add(f.get());
			}
			return results;
		} catch (Exception e) {
			log.info("EXCEPTION : CAL_CANDIDATES : " + e.getMessage(), e);
			return Collections.emptyList();
		}
	}
	
	public List<RFModelResult> calCandidatesClusterDetails(String mID, List<String> cIDs) {
		try {
			Map<String, JsonNode> docNodes = storageService.findDocInfos(mID, cIDs);
			JsonNode mNode = docNodes.get(mID);
			ArrayNode cNodes = mapper.createArrayNode();
			for (String cID : cIDs) {
				cNodes.add(docNodes.get(cID));
			}
			return this.calCandidatesClusterInfo(mNode, cNodes, true, "v2");
		} catch (Exception e) {
			e.printStackTrace();
			log.info("EXCEPTION : CAL_CANDIDATES : " + e.getMessage(), e);
			return Collections.emptyList();
		}
	}
}
