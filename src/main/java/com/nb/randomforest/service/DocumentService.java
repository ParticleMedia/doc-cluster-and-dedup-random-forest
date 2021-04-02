package com.nb.randomforest.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.nb.randomforest.entity.EventFeature;
import com.nb.randomforest.entity.resource.RFModelResult;
import com.nb.randomforest.utils.MyAttributeBuilder;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.BooleanUtils;
import org.apache.commons.lang3.StringUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.Instances;

import java.util.*;

/**
 * @author yuxi
 * @date 2020/10/9
 */
@Service
@Slf4j
public class DocumentService {
	
	@Autowired
	ObjectMapper mapper;
	
	@Autowired
	RandomForest randomForest;
	
	
	@Autowired
	StorageService storageService;
	
	/**
	 *
	 */
	public List<RFModelResult> calCandidatesClusterInfo(JsonNode masterNode, JsonNode canditNodes, Boolean isDebug) {
		try {
			String mID = masterNode.hasNonNull("_id") ? masterNode.get("_id").textValue() : "";
			String title = masterNode.hasNonNull("seg_title") ? masterNode.get("seg_title").textValue() :
				masterNode.hasNonNull("stitle") ? masterNode.get("stitle").textValue() : "";
			title = title.toLowerCase();
			// 1.预处理
			Instances instances;
			List<EventFeature> features = new ArrayList<>();
			ArrayList<Attribute> attributes = MyAttributeBuilder.buildMyAttributesV1();
			// geotag rule
			boolean isLocalNews = masterNode.hasNonNull("local_score") && masterNode.get("local_score").doubleValue() > 0.5;

			boolean isCelebrities = false;
			boolean isEconomyMarkets = false;
			boolean isSports = false;
			boolean isWeather = false;
			boolean aboutFauci = false;
			if (masterNode.hasNonNull("text_category") && masterNode.get("text_category").hasNonNull("second_cat")) {
				JsonNode firstCatNode = masterNode.get("text_category").get("first_cat");
				Iterator<String> itrFirst = firstCatNode.fieldNames();
				while (itrFirst.hasNext()) {
					String key = itrFirst.next();
					if (StringUtils.equals("Sports", key)){
						isSports = true;
					}
				}
				
				JsonNode secondCatNode = masterNode.get("text_category").get("second_cat");
				Iterator<String> itrSecond = secondCatNode.fieldNames();
				while (itrSecond.hasNext()) {
					String key = itrSecond.next();
					if (StringUtils.equals("BusinessEconomy_Markets", key)){
						isEconomyMarkets = true;
					} else if (StringUtils.equals("ArtsEntertainment_Celebrities", key)) {
						isCelebrities = true;
					} else if (StringUtils.equals("ClimateEnvironment_Weather", key)) {
						isWeather = true;
					}
				}
			}
			if (title.contains("fauci says")) {
				aboutFauci = true;
			}
			
			// 2. prepare instances for RFModel
			instances = new Instances(UUID.randomUUID().toString(), attributes, 1);
			int classIndex = instances.numAttributes() - 1;
			instances.setClassIndex(classIndex);
			for (JsonNode canditNode : canditNodes) {
				EventFeature feature = new EventFeature(masterNode, canditNode, null);
				features.add(feature);
				instances.add(feature.toInstanceV1());
			}
			
			// 3.模型结果后处理 + 卡阈值
			List<RFModelResult> cls = new ArrayList<>();
			double[][] canditResults = randomForest.distributionsForInstances(instances);
			for (int j = 0; j < canditResults.length; j++) {
				JsonNode canditNode = canditNodes.get(j);
				boolean canditIsLocalNews = canditNode.hasNonNull("local_score") && canditNode.get("local_score").doubleValue()>0.5;

				double[] canditResult = canditResults[j];
				EventFeature feature = features.get(j);
				String cID = canditNode.hasNonNull("_id") ? canditNode.get("_id").textValue() : "";
				String label;
				double diffScore = canditResult[0];
				double evtScore = canditResult[1];
				double score;
				log.info(String.format("MODEL DEBUG: %s\t%s\t%.5f", mID, cID, evtScore));
				//模型结果后处理
				if ((isEconomyMarkets && evtScore > 0.98) ||
					(isCelebrities && evtScore > 0.95) ||
					//normal case 阈值
					(!isEconomyMarkets && !isCelebrities && !aboutFauci && evtScore > 0.9) ||
//					//title 相似度豁免
					(isWeather && feature.getTitleRatio() >= 0.7 && feature.getTitleLength() >= 5 && evtScore > 0.85) ||
					(isSports && feature.getTitleRatio() >= 0.5 && feature.getTitleLength() >= 5 && evtScore > 0.85) ||
					(!aboutFauci && !isSports && !isWeather && feature.getTitleRatio() >= 0.65 && feature.getTitleLength() >= 5 && evtScore > 0.75) ||
					//simhash dist 豁免
					(!aboutFauci && !isSports && !isWeather && feature.getSimhashDist() != null && feature.getSimhashDist() < 5 && evtScore > 0.8)
				) {
					label = "DUP";
					score = evtScore;
				} else if ((isEconomyMarkets && evtScore > 0.55) ||
					(isSports && evtScore > 0.6) ||
					(isWeather && evtScore > 0.6 && feature.getGeoRatio() != null && feature.getGeoRatio() > 0.5) ||
					(isWeather && evtScore > 0.8 && feature.getGeoRatio() == null) ||
					(aboutFauci && evtScore > 0.85) ||
					(!isEconomyMarkets && !isSports && !isWeather && !aboutFauci && evtScore > 0.5)
				) {
					// geotag rule
					if (isLocalNews && canditIsLocalNews && feature.getGeoRatio() != null && feature.getGeoRatio()<0.1) {
						label = "DIFF";
						log.info("No geotag overalap");
					}else {
						label = "EVENT";
					}
					score = evtScore;
				} else {
					label = "DIFF";
					score = diffScore;
				}

				cls.add(new RFModelResult(
					cID, label, score, BooleanUtils.isTrue(isDebug) ? feature : null
				));
			}
			return cls;
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
			return this.calCandidatesClusterInfo(mNode, cNodes, true);
		} catch (Exception e) {
			e.printStackTrace();
			log.info("EXCEPTION : CAL_CANDIDATES : " + e.getMessage(), e);
			return Collections.emptyList();
		}
	}
}
