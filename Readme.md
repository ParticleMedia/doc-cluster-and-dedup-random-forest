# Similarity score between two documents

This repository hosts models which get called by the clustering servers. The RandomForest models for v1/v2 output the similarity score between a pair of docs (to interpret the score as 0.0 = DIFF, 1.0 = EVENT or DUP). The XGBoost model for v3 classifies a pair of docs as DIFF, EVENT, or DUP, with the three probability scores adding to 1.

1. RandomForest CI https://jenkins.n.newsbreak.com/job/doc-cluster-and-dedup-random-forest-ci/
2. RandomForest v1 CD https://jenkins.n.newsbreak.com/job/doc-cluster-and-dedup-random-forest-deploy/
3. RandomForest v2/v3 CD https://jenkins.n.newsbreak.com/job/doc-cluster-and-dedup-random-forest-v2-deploy/

# Related

1. v1 clustering server https://github.com/ParticleMedia/doc-cluster-and-dedup-service-java
2. v2/v3 clustering server https://github.com/ParticleMedia/doc-cluster-and-dedup-service-python

# 从零训练
## 步骤说明
1. 从线上获取 doc pair 格式为 : doc_id + \t + doc_id
2. 根据需求送评标注
3. 基于 MongoUtils 构造 File_Fields, 格式为 : doc_id + \t + doc_id + \t + jstr + \t + jstr
4. 基于 ARFFUtils + File_Fields 构造 File.arff
 注: 如需新增特征或查看特征计算详情, 见 FeatureUtils, 且新增特征需保证特征数量与 arff header 保持一致.
5. 基于 ModelUtils + File.arff 训练模型
