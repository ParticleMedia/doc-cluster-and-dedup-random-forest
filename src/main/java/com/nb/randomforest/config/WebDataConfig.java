package com.nb.randomforest.config;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.common.collect.Maps;
import com.mongodb.client.MongoClient;
import com.mongodb.client.MongoClients;
import com.mongodb.client.MongoCollection;
import com.mongodb.client.MongoDatabase;
import com.ning.http.client.AsyncHttpClient;
import com.ning.http.client.AsyncHttpClientConfig;
import com.ning.http.client.providers.jdk.JDKAsyncHttpProvider;
import com.pmi.serving.metrics.MetricsFactory;
import com.pmi.serving.metrics.MetricsFactoryUtil;
import com.pmi.serving.metrics.OnDemandMetricsFactory;
import com.pmi.serving.metrics.reporter.opentsdb.HttpOpenTsdbClient;
import com.pmi.serving.metrics.reporter.opentsdb.OpenTsdbClient;
import org.bson.Document;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.MediaType;
import org.springframework.http.converter.json.MappingJackson2HttpMessageConverter;
import weka.classifiers.trees.RandomForest;
import weka.core.SerializationHelper;

import java.net.InetAddress;
import java.util.Arrays;
import java.util.Map;

/**
 * @author yuxi
 * @date 2020/10/9
 */
@Configuration
public class WebDataConfig {
	
	@Bean("MetricsFactory")
	public MetricsFactory metricsFactory() throws Exception {
		Map<String, String> tags = Maps.newHashMap();
		tags.put("component", "DOC_EVENT_RF");
		String hostName = InetAddress.getLocalHost().getHostName();
		tags.put("host", hostName);
		AsyncHttpClientConfig config = new AsyncHttpClientConfig.Builder()
			.setMaxConnections(100)
			.setMaxConnectionsPerHost(100)
			.build();
		AsyncHttpClient asyncHttpClient = new AsyncHttpClient(new JDKAsyncHttpProvider(config));
		OpenTsdbClient openTsdbClient = new HttpOpenTsdbClient(asyncHttpClient, "http://opentsdb.ha.nb.com:4242/api/put");
		MetricsFactory metricsFactory = new OnDemandMetricsFactory(tags, openTsdbClient);
		MetricsFactoryUtil.register(metricsFactory);
		return MetricsFactoryUtil.getRegisteredFactory();
	}
	
	
	@Bean("RandomForest")
	public RandomForest randomForest() throws Exception {
		RandomForest forest = (RandomForest) SerializationHelper.read(WebDataConfig.class.getClassLoader().getResourceAsStream("model/forest.model"));
		return forest;
	}
	
	@Bean
	public MongoCollection<Document> mongoClient() {
		MongoClient client = MongoClients.create("mongodb://rs3.mongo.nb.com:27017");
		MongoDatabase db = client.getDatabase("staticFeature");
		return db.getCollection("document");
	}
	
	@Bean
	public MappingJackson2HttpMessageConverter mappingJackson2HttpMessageConverter(ObjectMapper objectMapper) {
		MappingJackson2HttpMessageConverter converter = new MappingJackson2HttpMessageConverter(objectMapper);
		converter.setSupportedMediaTypes(Arrays.asList(MediaType.APPLICATION_JSON, MediaType.APPLICATION_JSON_UTF8));
		return converter;
	}
}
