package com.nb.randomforest.endpoint;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.nb.randomforest.entity.resource.RFModelResult;
import com.nb.randomforest.service.DocumentServiceMultiThread;
import com.nb.randomforest.service.DocumentService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import static org.springframework.web.bind.annotation.RequestMethod.GET;
import static org.springframework.web.bind.annotation.RequestMethod.POST;

/**
 * @author yuxi
 * @date 2020/10/9
 */
@RestController
@RequestMapping(name = "document", path = "/document")
@Slf4j
public class DocumentEndpoint {
	
	@Autowired
	ObjectMapper objectMapper;

	@Autowired
	DocumentServiceMultiThread documentService;
	
	
	/**
	 * 基于给定候选集的特征计算聚类去重结果
	 *
	 */
	@RequestMapping(method = POST)
	List<RFModelResult> calCandidatesClusterInfo(
		@RequestBody Map<String, Object> postBody
	) {
		JsonNode masterNode = objectMapper.valueToTree(postBody.get("master"));
		JsonNode candidates = objectMapper.valueToTree(postBody.get("candidates"));
		String masterID = masterNode.hasNonNull("_id") ? masterNode.get("_id").textValue() : "";
		log.info(String.format(
			"score candidates for %s, num_of_candidates: %d", masterID, candidates.size()));
		if (candidates.size() > 0) {
			return documentService.calCandidatesClusterInfo(masterNode, candidates, false);
		} else {
			return Collections.emptyList();
		}
	}
	
	
	/**
	 * 基于给定的候选集 ID, 从 DB 中获取特征, 并返回聚类去重分类结果详情
	 */
	@RequestMapping(method = POST, path = "/details")
	List<RFModelResult> calCandidatesClusterDetails(
		@RequestBody Map<String, Object> postBody
	) {
		String mID = (String) postBody.get("mID");
		List<String> cIDs = (ArrayList<String>) postBody.get("cIDs");
		return documentService.calCandidatesClusterDetails(mID, cIDs);
	}
}
