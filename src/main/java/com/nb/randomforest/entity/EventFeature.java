package com.nb.randomforest.entity;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.commons.lang3.StringUtils;
import weka.core.*;

import java.text.SimpleDateFormat;
import java.util.*;

import static com.nb.randomforest.utils.FeatureUtils.*;


/**
 * 模型训练数据特征. 包含对应的 序列化 及 实例化方法.
 */
public class EventFeature {
	
    private String label;
	
	/**
	 * Title
	 *
	 * Levenshtein Distance
	 * Overlap Ratio
	 * Average Length
	 */
	private String mTitle;
	
	private String cTitle;
	
	private Double titleDist;
	
	private Double titleRatio;
	
	private Double titleLength;
	
	/**
	 * Is Same Source
	 */
	private Double sameSRC;
    
	/**
	 * Word Count Span
	 */
	private Double cWordSpan;
	
	/**
	 * Paragraph Span
	 */
	private Double paragraphSpan;
	
	/**
	 * Time Span
	 */
	private Double epochSpan;
	
	private Double insertSpan;
	
	/**
	 * Simhash Distance
	 */
	private Double simhashDist;
	
	/**
	 * Image PHash Dist(Min)
	 */
	private Double imghashDist;
	
	/**
	 * Face PHash Dist(Min)
	 */
	private Double facehashDist;
    
    /**
     * Content Keywords
     *
     * Overlap Ratio
     * Average Length
     */
    private Double cKWSRatio;
	
	private Double cKWSLength;
	
	/**
	 * Title Keywords
	 *
	 * Overlap Ratio
	 * Average Length
	 */
	private Double tKWSRatio;
	
	private Double tKWSLength;
	
	/**
	 * HighLight Keywords
	 *
	 * Overlap Ratio
	 * Average Length
	 */
	private Double hKWSRatio;
	
	private Double hKWSLength;
	
	/**
	 * Channels
	 *
	 * Overlap Ratio
	 * Average Length
	 */
	private Double channelRatio;
	
	private Double channelLength;
	
	/**
	 * Content Organization
	 *
	 * NE : Weighted Overlap Ratio
	 * SP : Weighted Overlap Ratio
	 * NE : Average Length
	 * SP : Average Length
	 */
	private Double cOrgRatioNE;
	
	private Double cOrgRatioSP;
	
	private Double cOrgLengthNE;
	
	private Double cOrgLengthSP;
	
	/**
	 * Content Location
	 *
	 * NE : Weighted Overlap Ratio
	 * SP : Weighted Overlap Ratio
	 * NE : Average Length
	 * SP : Average Length
	 */
	private Double cLocRatioNE;
	
	private Double cLocRatioSP;
	
	private Double cLocLengthNE;
	
	private Double cLocLengthSP;
	
	/**
	 * Content Person
	 *
	 * NE : Weighted Overlap Ratio
	 * SP : Weighted Overlap Ratio
	 * NE : Average Length
	 * SP : Average Length
	 */
	private Double cPerRatioNE;
	
	private Double cPerRatioSP;
	
	private Double cPerLengthNE;
	
	private Double cPerLengthSP;
	
	/**
	 * Content NUM
	 *
	 * SP : Weighted Overlap Ratio
	 * SP : Average Length
	 */
	private Double cNUMRatioSP;
	
	private Double cNUMLengthSP;
	
	/**
	 * Content Time
	 *
	 * SP : Weighted Overlap Ratio
	 * SP : Average Length
	 */
	private Double cTimRatioSP;
	
	private Double cTimLengthSP;
	
	/**
	 * Title Organization
	 *
	 * NE : Weighted Overlap Ratio
	 * SP : Weighted Overlap Ratio
	 * NE : Average Length
	 * SP : Average Length
	 */
	private Double tOrgRatioNE;
	
	private Double tOrgRatioSP;
	
	private Double tOrgLengthNE;
	
	private Double tOrgLengthSP;
	
	/**
	 * Title Location
	 *
	 * NE : Weighted Overlap Ratio
	 * SP : Weighted Overlap Ratio
	 * NE : Average Length
	 * SP : Average Length
	 */
	private Double tLocRatioNE;
	
	private Double tLocRatioSP;
	
	private Double tLocLengthNE;
	
	private Double tLocLengthSP;
	
	/**
	 * Title Person
	 *
	 * NE : Weighted Overlap Ratio
	 * SP : Weighted Overlap Ratio
	 * NE : Average Length
	 * SP : Average Length
	 */
	private Double tPerRatioNE;
	
	private Double tPerRatioSP;
	
	private Double tPerLengthNE;
	
	private Double tPerLengthSP;
	
	/**
	 * Title NUM
	 *
	 * SP : Weighted Overlap Ratio
	 * SP : Average Length
	 */
	private Double tNUMRatioSP;
	
	private Double tNUMLengthSP;
	
	/**
	 * Title Time
	 *
	 * SP : Weighted Overlap Ratio
	 * SP : Average Length
	 */
	private Double tTimRatioSP;
	
	private Double tTimLengthSP;
	
	/**
	 * Category
	 *
	 * Overlap Ratio
	 * Average Length
	 */
	private Double catRatio;
	
	private Double catLength;
	
	/**
	 * Geotag
	 *
	 * Overlap Ratio
	 * Average Length
	 */
	private Double geoRatio;
	private Double strictGeoRatio;
	private Double geoLength;
	
	private Integer categoryIndex;
	

    public EventFeature(JsonNode masterNode, JsonNode canditNode, String label) throws Exception {
    	if (!StringUtils.isEmpty(label) && !label.equals("DIFF") && !label.equals("EVENT") && !label.equals("DUP")) {
		    throw new Exception("Invalid Label! Required DIFF, EVENT , DUP or null.");
	    }
        this.label = label;
    	
	    //Title
	    String mTitleRaw = masterNode.hasNonNull("seg_title") ? masterNode.get("seg_title").textValue() :
		    masterNode.hasNonNull("stitle") ? masterNode.get("stitle").textValue() : "";
	    String cTitleRaw = canditNode.hasNonNull("seg_title") ? canditNode.get("seg_title").textValue() :
		    canditNode.hasNonNull("stitle") ? canditNode.get("stitle").textValue() : "";
	    String mTitle = titlePreprocess(mTitleRaw);
	    List<String> mTitleList = Arrays.asList(mTitle.split(" "));
	    String cTitle = titlePreprocess(cTitleRaw);
	    List<String> cTitleList = Arrays.asList(cTitle.split(" "));
	    this.mTitle = mTitle;
	    this.cTitle = cTitle;
        this.titleDist = levenshteinDistance(mTitle, cTitle);
        this.titleRatio = overlapRatio(mTitleList, cTitleList);
        this.titleLength = averageLength(mTitleList, cTitleList);
        
        //Source
        this.sameSRC = isEqual(
        	masterNode.hasNonNull("source") ? masterNode.get("source").textValue() :
		        masterNode.hasNonNull("src") ? masterNode.get("src").textValue() : null,
	        canditNode.hasNonNull("source") ? canditNode.get("source").textValue() :
		        canditNode.hasNonNull("src") ? canditNode.get("src").textValue() : null
        );
        
        //Word Count Span
        this.cWordSpan = numSpan(
        	masterNode.hasNonNull("c_word") ? masterNode.get("c_word").asLong() : null,
	        canditNode.hasNonNull("c_word") ? canditNode.get("c_word").asLong() : null
        );
	
        //Paragraph Count Span
	    this.paragraphSpan = numSpan(
	    	masterNode.hasNonNull("paragraph_count") ? masterNode.get("paragraph_count").asLong() : null,
		    canditNode.hasNonNull("paragraph_count") ? canditNode.get("paragraph_count").asLong() : null
	    );
	    
	    //Time Span
        Long mEpoch;
        Long cEpoch;
	    if (masterNode.hasNonNull("epoch") && masterNode.get("epoch").isNumber()) {
	    	mEpoch = masterNode.get("epoch").asLong();
	    } else if (masterNode.hasNonNull("epoch")) {
		    mEpoch = masterNode.get("epoch").get("$numberLong").asLong();
	    } else {
	    	mEpoch = null;
	    }
	    if (canditNode.hasNonNull("epoch") && canditNode.get("epoch").isNumber()) {
		    cEpoch = canditNode.get("epoch").asLong();
	    } else if (canditNode.hasNonNull("epoch")) {
		    cEpoch = canditNode.get("epoch").get("$numberLong").asLong();
	    } else {
	    	cEpoch = null;
	    }
	    this.epochSpan = numSpan(mEpoch, cEpoch);
	    SimpleDateFormat df = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
	    Long mInsert;
	    Long cInsert;
	    if (masterNode.hasNonNull("insert_time")) {
		    mInsert = df.parse(masterNode.get("insert_time").asText()).getTime() / 1000;
	    } else {
		    mInsert = null;
	    }
	    if (canditNode.hasNonNull("insert_time")) {
		    cInsert = df.parse(canditNode.get("insert_time").asText()).getTime() / 1000;
	    } else {
		    cInsert = null;
	    }
	    this.insertSpan = numSpan(mInsert, cInsert);
	    
	    //Simhash Distance
	    this.simhashDist = simhashDist(masterNode.get("simhash"), canditNode.get("simhash"));
	    
	    //ImgHash + FaceHash
	    this.imghashDist = simhashDist(masterNode.get("images_phash"), canditNode.get("images_phash"));
	    this.facehashDist = simhashDist(masterNode.get("faces_phash"), canditNode.get("faces_phash"));
	    
	    //Content Keywords :
//	    List<String> mContentKWSList = new ArrayList<>();
//	    List<String> cContentKWSList = new ArrayList<>();
//	    if (masterNode.hasNonNull("kws") && masterNode.get("kws").isArray()) {
//		    masterNode.get("kws").forEach(kw -> mContentKWSList.add(kw.asText()));
//	    }
//	    if (canditNode.has("kws") && canditNode.get("kws").isArray()) {
//	    	canditNode.get("kws").forEach(kw -> cContentKWSList.add(kw.asText()));
//	    }
//	    this.cKWSRatio = overlapRatio(mContentKWSList, cContentKWSList);
//	    this.cKWSLength = averageLength(mContentKWSList, cContentKWSList);
	    //Content Keywords :
	    this.cKWSRatio = weightedOverlapRatio(masterNode.get("kws"), canditNode.get("kws"));
	    this.cKWSLength = weightedAverageLength(masterNode.get("kws"), canditNode.get("kws"));
	    
	    //Title Keywords
//	    List<String> mTitleKWSList = new ArrayList<>();
//	    List<String> cTitleKWSList = new ArrayList<>();
//	    if (masterNode.hasNonNull("kw_title") && masterNode.get("kw_title").isArray()) {
//	    	masterNode.get("kw_title").forEach(kw -> mTitleKWSList.add(kw.asText()));
//	    }
//	    if (canditNode.hasNonNull("kw_title") && canditNode.get("kw_title").isArray()) {
//		    canditNode.get("kw_title").forEach(kw -> cTitleKWSList.add(kw.asText()));
//	    }
//	    this.tKWSRatio = overlapRatio(mTitleKWSList, cTitleList);
//	    this.tKWSLength = averageLength(mTitleList, cTitleList);
	    this.tKWSRatio = weightedOverlapRatio(masterNode.get("kw_title"), canditNode.get("kw_title"));
	    this.tKWSLength = weightedAverageLength(masterNode.get("kw_title"), canditNode.get("kw_title"));
	    
	    //HigLight Keywords
	    List<String> mHighKWSList = new ArrayList<>();
	    List<String> cHighKWSList = new ArrayList<>();
	    if (masterNode.hasNonNull("highlightkeyword_list") && masterNode.get("highlightkeyword_list").isArray()) {
		    masterNode.get("highlightkeyword_list").forEach(kw_pair -> {
			    if (kw_pair.get(1).asDouble() > 0.6d) {
				    mHighKWSList.add(kw_pair.get(0).asText());
			    }
		    });
	    }
	    if (canditNode.hasNonNull("highlightkeyword_list") && canditNode.get("highlightkeyword_list").isArray()) {
		    canditNode.get("highlightkeyword_list").forEach(kw_pair -> {
		    	if (kw_pair.get(1).asDouble() > 0.6d) {
				    cHighKWSList.add(kw_pair.get(0).asText());
			    }
		    });
	    }
	    this.hKWSRatio = overlapRatio(mHighKWSList, cHighKWSList);
	    this.hKWSLength = averageLength(mHighKWSList, cHighKWSList);
	    
	    //Channel
	    List<String> mChnList = new ArrayList<>();
	    List<String> cChnList = new ArrayList<>();
	    JsonNode mChannel = masterNode.has("channels") ? masterNode.get("channels") : masterNode.get("channels_v2");
	    JsonNode cChannel = canditNode.has("channels") ? canditNode.get("channels") : canditNode.get("channels_v2");
	    if (mChannel != null && mChannel.isArray()) {
	    	mChannel.forEach(chn -> mChnList.add(chn.asText()));
	    }
	    if (cChannel != null && cChannel.isArray()) {
	    	cChannel.forEach(chn -> cChnList.add(chn.asText()));
	    }
	    this.channelRatio = overlapRatio(mChnList, cChnList);
	    this.channelLength = averageLength(mChnList, cChnList);
	    
	    //Content Organization
	    this.cOrgRatioNE = weightedOverlapRatio(masterNode.get("ne_content_organization"), canditNode.get("ne_content_organization"));
	    this.cOrgRatioSP = weightedOverlapRatio(masterNode.get("spacy_content_org"), canditNode.get("spacy_content_org"));
	    this.cOrgLengthNE = weightedAverageLength(masterNode.get("ne_content_organization"), canditNode.get("ne_content_organization"));
	    this.cOrgLengthSP = weightedAverageLength(masterNode.get("spacy_content_org"), canditNode.get("spacy_content_org"));
	    
	    //Content Location
	    this.cLocRatioNE = weightedOverlapRatio(masterNode.get("ne_content_location"), canditNode.get("ne_content_location"));
	    this.cLocRatioSP = weightedOverlapRatio(masterNode.get("spacy_content_loc"), canditNode.get("spacy_content_loc"));
	    this.cLocLengthNE = weightedAverageLength(masterNode.get("ne_content_location"), canditNode.get("ne_content_location"));
	    this.cLocLengthSP = weightedAverageLength(masterNode.get("spacy_content_loc"), canditNode.get("spacy_content_loc"));
	    
	    //Content Person
	    this.cPerRatioNE = weightedOverlapRatio(masterNode.get("ne_content_person"), canditNode.get("ne_content_person"));
	    this.cPerRatioSP = weightedOverlapRatio(masterNode.get("spacy_content_per"), canditNode.get("spacy_content_per"));
	    this.cPerLengthNE = weightedAverageLength(masterNode.get("ne_content_person"), canditNode.get("ne_content_person"));
	    this.cPerLengthSP = weightedAverageLength(masterNode.get("spacy_content_per"), canditNode.get("spacy_content_per"));
	    
	    //Content NUM
	    this.cNUMRatioSP = weightedOverlapRatio(masterNode.get("spacy_content_num"), canditNode.get("spacy_content_num"));
	    this.cNUMLengthSP = weightedAverageLength(masterNode.get("spacy_content_num"), canditNode.get("spacy_content_num"));
	    
	    //Content Time
	    this.cTimRatioSP = weightedOverlapRatio(masterNode.get("spacy_content_tim"), canditNode.get("spacy_content_tim"));
	    this.cTimLengthSP = weightedAverageLength(masterNode.get("spacy_content_tim"), canditNode.get("spacy_content_tim"));
	    
	    //Title Organization
	    this.tOrgRatioNE = weightedOverlapRatio(masterNode.get("ne_title_organization"), canditNode.get("ne_title_organization"));
	    this.tOrgRatioSP = weightedOverlapRatio(masterNode.get("spacy_title_org"), canditNode.get("spacy_title_org"));
	    this.tOrgLengthNE = weightedAverageLength(masterNode.get("ne_title_organization"), canditNode.get("ne_title_organization"));
	    this.tOrgLengthSP = weightedAverageLength(masterNode.get("spacy_title_org"), canditNode.get("spacy_title_org"));
	    
	    //Title Location
	    this.tLocRatioNE = weightedOverlapRatio(masterNode.get("ne_title_location"), canditNode.get("ne_title_location"));
	    this.tLocRatioSP = weightedOverlapRatio(masterNode.get("spacy_title_loc"), canditNode.get("spacy_title_loc"));
	    this.tLocLengthNE = weightedAverageLength(masterNode.get("ne_title_location"), canditNode.get("ne_title_location"));
	    this.tLocLengthSP = weightedAverageLength(masterNode.get("spacy_title_loc"), canditNode.get("spacy_title_loc"));
	    
	    //Title Person
	    this.tPerRatioNE = weightedOverlapRatio(masterNode.get("ne_title_person"), canditNode.get("ne_title_person"));
	    this.tPerRatioSP = weightedOverlapRatio(masterNode.get("spacy_title_per"), canditNode.get("spacy_title_per"));
	    this.tPerLengthNE = weightedAverageLength(masterNode.get("ne_title_person"), canditNode.get("ne_title_person"));
	    this.tPerLengthSP = weightedAverageLength(masterNode.get("spacy_title_per"), canditNode.get("spacy_title_per"));
	    
	    //Title NUM
	    this.tNUMRatioSP = weightedOverlapRatio(masterNode.get("spacy_title_num"), canditNode.get("spacy_title_num"));
	    this.tNUMLengthSP = weightedAverageLength(masterNode.get("spacy_title_num"), canditNode.get("spacy_title_num"));
	    
	    //Title Time
	    this.tTimRatioSP = weightedOverlapRatio(masterNode.get("spacy_title_tim"), canditNode.get("spacy_title_tim"));
	    this.tTimLengthSP = weightedAverageLength(masterNode.get("spacy_title_tim"), canditNode.get("spacy_title_tim"));
	    
	    //Category
	    JsonNode mCategory = masterNode.has("text_category") ? masterNode.get("text_category") : masterNode.get("text_category_v2");
	    JsonNode cCategory = canditNode.has("text_category") ? canditNode.get("text_category") : canditNode.get("text_category_v2");
	    this.catRatio = categoryOverlapRatio(mCategory, cCategory);
	    this.catLength = categoryAverageLength(mCategory, cCategory);
	    
	    //GEO Tag
	    JsonNode mGeo = masterNode.has("geotag") ? masterNode.get("geotag") : masterNode.get("geotag_v2");
	    JsonNode cGeo = canditNode.has("geotag") ? canditNode.get("geotag") : canditNode.get("geotag_v2");
	    this.geoRatio = geotagOverlapRatio(mGeo, cGeo, Boolean.FALSE);
	    this.strictGeoRatio = geotagOverlapRatio(mGeo, cGeo, Boolean.TRUE);
	    this.geoLength = geotagAverageLength(mGeo, cGeo);
	    
	    //Category Index : Huffman feature encoding for wide & deep model
	    // wrong       : -1
	    // normal      : 0
	    // sports      : 1
	    // economy     : 2
	    // celebrities : 4
	    // weather     : 8
	    this.categoryIndex = 0;
	    if (mCategory == null || mCategory.size() == 0) {
	    	this.categoryIndex = -1;
	    } else {
		    JsonNode firstCatNode = mCategory.get("first_cat");
		    Iterator<String> itrFirst = firstCatNode.fieldNames();
		    while (itrFirst.hasNext()) {
			    String key = itrFirst.next();
			    if (StringUtils.equals("Sports", key)){
				    this.categoryIndex += 1;
			    }
		    }
		    JsonNode secondCatNode = mCategory.get("second_cat");
		    Iterator<String> itrSecond = secondCatNode.fieldNames();
		    while (itrSecond.hasNext()) {
			    String key = itrSecond.next();
			    if (StringUtils.equals("BusinessEconomy_Markets", key)){
				    this.categoryIndex += 2;
			    } else if (StringUtils.equals("ArtsEntertainment_Celebrities", key)) {
				    this.categoryIndex += 4;
			    } else if (StringUtils.equals("ClimateEnvironment_Weather", key)) {
				    this.categoryIndex += 8;
			    }
		    }
	    }
    }
    
    
	public String getLabel() {
		return label;
	}
	
	public Double getTitleDist() {
		return titleDist;
	}
	
	public Double getTitleRatio() {
		return titleRatio;
	}
	
	public Double getTitleLength() {
		return titleLength;
	}
	
	public Double getSameSRC() {
		return sameSRC;
	}
	
	public Double getcWordSpan() {
		return cWordSpan;
	}
	
	public Double getParagraphSpan() {
		return paragraphSpan;
	}
	
	public Double getEpochSpan() {
		return epochSpan;
	}
	
	public Double getInsertSpan() {
		return insertSpan;
	}
	
	public Double getSimhashDist() {
		return simhashDist;
	}
	
	public Double getImghashDist() {
		return imghashDist;
	}
	
	public Double getFacehashDist() {
		return facehashDist;
	}
	
	public Double getcKWSRatio() {
		return cKWSRatio;
	}
	
	public Double getcKWSLength() {
		return cKWSLength;
	}
	
	public Double gettKWSRatio() {
		return tKWSRatio;
	}
	
	public Double gettKWSLength() {
		return tKWSLength;
	}
	
	public Double gethKWSRatio() {
		return hKWSRatio;
	}
	
	public Double gethKWSLength() {
		return hKWSLength;
	}
	
	public Double getChannelRatio() {
		return channelRatio;
	}
	
	public Double getChannelLength() {
		return channelLength;
	}
	
	public Double getcOrgRatioNE() {
		return cOrgRatioNE;
	}
	
	public Double getcOrgRatioSP() {
		return cOrgRatioSP;
	}
	
	public Double getcOrgLengthNE() {
		return cOrgLengthNE;
	}
	
	public Double getcOrgLengthSP() {
		return cOrgLengthSP;
	}
	
	public Double getcLocRatioNE() {
		return cLocRatioNE;
	}
	
	public Double getcLocRatioSP() {
		return cLocRatioSP;
	}
	
	public Double getcLocLengthNE() {
		return cLocLengthNE;
	}
	
	public Double getcLocLengthSP() {
		return cLocLengthSP;
	}
	
	public Double getcPerRatioNE() {
		return cPerRatioNE;
	}
	
	public Double getcPerRatioSP() {
		return cPerRatioSP;
	}
	
	public Double getcPerLengthNE() {
		return cPerLengthNE;
	}
	
	public Double getcPerLengthSP() {
		return cPerLengthSP;
	}
	
	public Double getcNUMRatioSP() {
		return cNUMRatioSP;
	}
	
	public Double getcNUMLengthSP() {
		return cNUMLengthSP;
	}
	
	public Double getcTimRatioSP() {
		return cTimRatioSP;
	}
	
	public Double getcTimLengthSP() {
		return cTimLengthSP;
	}
	
	public Double gettOrgRatioNE() {
		return tOrgRatioNE;
	}
	
	public Double gettOrgRatioSP() {
		return tOrgRatioSP;
	}
	
	public Double gettOrgLengthNE() {
		return tOrgLengthNE;
	}
	
	public Double gettOrgLengthSP() {
		return tOrgLengthSP;
	}
	
	public Double gettLocRatioNE() {
		return tLocRatioNE;
	}
	
	public Double gettLocRatioSP() {
		return tLocRatioSP;
	}
	
	public Double gettLocLengthNE() {
		return tLocLengthNE;
	}
	
	public Double gettLocLengthSP() {
		return tLocLengthSP;
	}
	
	public Double gettPerRatioNE() {
		return tPerRatioNE;
	}
	
	public Double gettPerRatioSP() {
		return tPerRatioSP;
	}
	
	public Double gettPerLengthNE() {
		return tPerLengthNE;
	}
	
	public Double gettPerLengthSP() {
		return tPerLengthSP;
	}
	
	public Double gettNUMRatioSP() {
		return tNUMRatioSP;
	}
	
	public Double gettNUMLengthSP() {
		return tNUMLengthSP;
	}
	
	public Double gettTimRatioSP() {
		return tTimRatioSP;
	}
	
	public Double gettTimLengthSP() {
		return tTimLengthSP;
	}
	
	public Double getCatRatio() {
		return catRatio;
	}
	
	public Double getCatLength() {
		return catLength;
	}
	
	public Double getGeoRatio() {
		return geoRatio;
	}
	public Double getStrictGeoRatio(){ return strictGeoRatio;}
	
	public Double getGeoLength() {
		return geoLength;
	}
	
	public Integer getCategoryIndex() {
		return categoryIndex;
	}
	
	public String getmTitle() {
		return mTitle;
	}
	
	public String getcTitle() {
		return cTitle;
	}
	
	public Instance toInstanceV0() {
		List<Double> doubleList = new ArrayList<>();
		doubleList.add(titleDist);
		doubleList.add(sameSRC);
		doubleList.add(cWordSpan);
		doubleList.add(epochSpan);
		doubleList.add(paragraphSpan);
		doubleList.add(simhashDist);
		doubleList.add(cKWSRatio);
		doubleList.add(channelRatio);
		doubleList.add(cOrgRatioNE);
		doubleList.add(cLocRatioNE);
		doubleList.add(cPerRatioNE);
		doubleList.add(tOrgRatioNE);
		doubleList.add(tLocRatioNE);
		doubleList.add(tPerRatioNE);
		doubleList.add(catRatio);
		doubleList.add(geoRatio);
		doubleList.add((StringUtils.isEmpty(label) || StringUtils.equals(label, "DIFF")) ? 0d : StringUtils.equals(label, "EVENT") ? 1d : 2d);
		double[] doubleArray = doubleList.stream().mapToDouble(d -> d == null ? Utils.missingValue() : d.doubleValue()).toArray();
		return new DenseInstance(1, doubleArray);
	}
	
	public Instance toInstanceV1() {
		List<Double> doubleList = new ArrayList<>();
		doubleList.add(titleDist);
		doubleList.add(titleRatio);
		doubleList.add(titleLength);
		doubleList.add(sameSRC);
		doubleList.add(cWordSpan);
		doubleList.add(paragraphSpan);
		doubleList.add(epochSpan);
		doubleList.add(insertSpan);
		doubleList.add(simhashDist);
		doubleList.add(imghashDist);
		doubleList.add(facehashDist);
		doubleList.add(cKWSRatio);
		doubleList.add(cKWSLength);
		doubleList.add(tKWSRatio);
		doubleList.add(tKWSLength);
		doubleList.add(hKWSRatio);
		doubleList.add(hKWSLength);
		doubleList.add(channelRatio);
		doubleList.add(channelLength);
		doubleList.add(cOrgRatioNE);
		doubleList.add(cOrgRatioSP);
		doubleList.add(cOrgLengthNE);
		doubleList.add(cOrgLengthSP);
		doubleList.add(cLocRatioNE);
		doubleList.add(cLocRatioSP);
		doubleList.add(cLocLengthNE);
		doubleList.add(cLocLengthSP);
		doubleList.add(cPerRatioNE);
		doubleList.add(cPerRatioSP);
		doubleList.add(cPerLengthNE);
		doubleList.add(cPerLengthSP);
		doubleList.add(cNUMRatioSP);
		doubleList.add(cNUMLengthSP);
		doubleList.add(cTimRatioSP);
		doubleList.add(cTimLengthSP);
		doubleList.add(tOrgRatioNE);
		doubleList.add(tOrgRatioSP);
		doubleList.add(tOrgLengthNE);
		doubleList.add(tOrgLengthSP);
		doubleList.add(tLocRatioNE);
		doubleList.add(tLocRatioSP);
		doubleList.add(tLocLengthNE);
		doubleList.add(tLocLengthSP);
		doubleList.add(tPerRatioNE);
		doubleList.add(tPerRatioSP);
		doubleList.add(tPerLengthNE);
		doubleList.add(tPerLengthSP);
		doubleList.add(tNUMRatioSP);
		doubleList.add(tNUMLengthSP);
		doubleList.add(tTimRatioSP);
		doubleList.add(tTimLengthSP);
		doubleList.add(catRatio);
		doubleList.add(catLength);
		doubleList.add(geoRatio);
		doubleList.add(geoLength);
		doubleList.add((StringUtils.isEmpty(label) || StringUtils.equals(label, "DIFF")) ? 0d : StringUtils.equals(label, "EVENT") ? 1d : 2d);
		double[] doubleArray = doubleList.stream().mapToDouble(d -> d == null ? Utils.missingValue() : d.doubleValue()).toArray();
		return new DenseInstance(1, doubleArray);
	}
	
	public Instance toInstanceV2() {
		List<Double> doubleList = new ArrayList<>();
		doubleList.add(titleRatio);
//		doubleList.add(titleLength);
		doubleList.add(sameSRC);
		doubleList.add(epochSpan);
		doubleList.add(insertSpan);
		doubleList.add(simhashDist);
		doubleList.add(cKWSRatio);
//		doubleList.add(cKWSLength);
		doubleList.add(tKWSRatio);
//		doubleList.add(tKWSLength);
		doubleList.add(hKWSRatio);
//		doubleList.add(hKWSLength);
		doubleList.add(channelRatio);
//		doubleList.add(channelLength);
		doubleList.add(cOrgRatioNE);
		doubleList.add(cOrgRatioSP);
//		doubleList.add(cOrgLengthNE);
//		doubleList.add(cOrgLengthSP);
		doubleList.add(cLocRatioNE);
		doubleList.add(cLocRatioSP);
//		doubleList.add(cLocLengthNE);
//		doubleList.add(cLocLengthSP);
		doubleList.add(cPerRatioNE);
		doubleList.add(cPerRatioSP);
//		doubleList.add(cPerLengthNE);
//		doubleList.add(cPerLengthSP);
		doubleList.add(catRatio);
//		doubleList.add(catLength);
		doubleList.add(geoRatio);
//		doubleList.add(geoLength);
		doubleList.add((StringUtils.isEmpty(label) || StringUtils.equals(label, "DIFF")) ? 0d : StringUtils.equals(label, "EVENT") ? 1d : 2d);
		double[] doubleArray = doubleList.stream().mapToDouble(d -> d == null ? Utils.missingValue() : d.doubleValue()).toArray();
		return new DenseInstance(1, doubleArray);
	}
	
	
	public String toStringV1() {
		StringBuilder sb = new StringBuilder();
		sb.append(titleDist == null ? "?" : titleDist.toString());
		sb.append(",");
		sb.append(titleRatio == null ? "?" : titleRatio.toString());
		sb.append(",");
		sb.append(titleLength == null ? "?" : titleLength.toString());
		sb.append(",");
		sb.append(sameSRC == null ? "?" : sameSRC.toString());
		sb.append(",");
		sb.append(cWordSpan == null ? "?" : cWordSpan.toString());
		sb.append(",");
		sb.append(paragraphSpan == null ? "?" : paragraphSpan.toString());
		sb.append(",");
		sb.append(epochSpan == null ? "?" : epochSpan.toString());
		sb.append(",");
		sb.append(insertSpan == null ? "?" : insertSpan.toString());
		sb.append(",");
		sb.append(simhashDist == null ? "?" : simhashDist.toString());
		sb.append(",");
		sb.append(imghashDist == null ? "?" : imghashDist.toString());
		sb.append(",");
		sb.append(facehashDist == null ? "?" : facehashDist.toString());
		sb.append(",");
		sb.append(cKWSRatio == null ? "?" : cKWSRatio.toString());
		sb.append(",");
		sb.append(cKWSLength == null ? "?" : cKWSLength.toString());
		sb.append(",");
		sb.append(tKWSRatio == null ? "?" : tKWSRatio.toString());
		sb.append(",");
		sb.append(tKWSLength == null ? "?" : tKWSLength.toString());
		sb.append(",");
		sb.append(hKWSRatio == null ? "?" : hKWSRatio.toString());
		sb.append(",");
		sb.append(hKWSLength == null ? "?" : hKWSLength.toString());
		sb.append(",");
		sb.append(channelRatio == null ? "?" : channelRatio.toString());
		sb.append(",");
		sb.append(channelLength == null ? "?" : channelLength.toString());
		sb.append(",");
		sb.append(cOrgRatioNE == null ? "?" : cOrgRatioNE.toString());
		sb.append(",");
		sb.append(cOrgRatioSP == null ? "?" : cOrgRatioSP.toString());
		sb.append(",");
		sb.append(cOrgLengthNE == null ? "?" : cOrgLengthNE.toString());
		sb.append(",");
		sb.append(cOrgLengthSP == null ? "?" : cOrgLengthSP.toString());
		sb.append(",");
		sb.append(cLocRatioNE == null ? "?" : cLocRatioNE.toString());
		sb.append(",");
		sb.append(cLocRatioSP == null ? "?" : cLocRatioSP.toString());
		sb.append(",");
		sb.append(cLocLengthNE == null ? "?" : cLocLengthNE.toString());
		sb.append(",");
		sb.append(cLocLengthSP == null ? "?" : cLocLengthSP.toString());
		sb.append(",");
		sb.append(cPerRatioNE == null ? "?" : cPerRatioNE.toString());
		sb.append(",");
		sb.append(cPerRatioSP == null ? "?" : cPerRatioSP.toString());
		sb.append(",");
		sb.append(cPerLengthNE == null ? "?" : cPerLengthNE.toString());
		sb.append(",");
		sb.append(cPerLengthSP == null ? "?" : cPerLengthSP.toString());
		sb.append(",");
		sb.append(cNUMRatioSP == null ? "?" : cNUMRatioSP.toString());
		sb.append(",");
		sb.append(cNUMLengthSP == null ? "?" : cNUMLengthSP.toString());
		sb.append(",");
		sb.append(cTimRatioSP == null ? "?" : cTimRatioSP.toString());
		sb.append(",");
		sb.append(cTimLengthSP == null ? "?" : cTimLengthSP.toString());
		sb.append(",");
		sb.append(tOrgRatioNE == null ? "?" : tOrgRatioNE.toString());
		sb.append(",");
		sb.append(tOrgRatioSP == null ? "?" : tOrgRatioSP.toString());
		sb.append(",");
		sb.append(tOrgLengthNE == null ? "?" : tOrgLengthNE.toString());
		sb.append(",");
		sb.append(tOrgLengthSP == null ? "?" : tOrgLengthSP.toString());
		sb.append(",");
		sb.append(tLocRatioNE == null ? "?" : tLocRatioNE.toString());
		sb.append(",");
		sb.append(tLocRatioSP == null ? "?" : tLocRatioSP.toString());
		sb.append(",");
		sb.append(tLocLengthNE == null ? "?" : tLocLengthNE.toString());
		sb.append(",");
		sb.append(tLocLengthSP == null ? "?" : tLocLengthSP.toString());
		sb.append(",");
		sb.append(tPerRatioNE == null ? "?" : tPerRatioNE.toString());
		sb.append(",");
		sb.append(tPerRatioSP == null ? "?" : tPerRatioSP.toString());
		sb.append(",");
		sb.append(tPerLengthNE == null ? "?" : tPerLengthNE.toString());
		sb.append(",");
		sb.append(tPerLengthSP == null ? "?" : tPerLengthSP.toString());
		sb.append(",");
		sb.append(tNUMRatioSP == null ? "?" : tNUMRatioSP.toString());
		sb.append(",");
		sb.append(tNUMLengthSP == null ? "?" : tNUMLengthSP.toString());
		sb.append(",");
		sb.append(tTimRatioSP == null ? "?" : tTimRatioSP.toString());
		sb.append(",");
		sb.append(tTimLengthSP == null ? "?" : tTimLengthSP.toString());
		sb.append(",");
		sb.append(catRatio == null ? "?" : catRatio.toString());
		sb.append(",");
		sb.append(catLength == null ? "?" : catLength.toString());
		sb.append(",");
		sb.append(geoRatio == null ? "?" : geoRatio.toString());
		sb.append(",");
		sb.append(geoLength == null ? "?" : geoLength.toString());
		sb.append(",");
		sb.append(StringUtils.isEmpty(label) ? "?" : label);
    	return sb.toString();
	}
	
	
	public String toStringV2() {
		StringBuilder sb = new StringBuilder();
		sb.append(titleRatio == null ? "?" : titleRatio.toString());
		sb.append(",");
//		sb.append(titleLength == null ? "?" : titleLength.toString());
//		sb.append(",");
		sb.append(sameSRC == null ? "?" : sameSRC.toString());
		sb.append(",");
		sb.append(epochSpan == null ? "?" : epochSpan.toString());
		sb.append(",");
		sb.append(insertSpan == null ? "?" : insertSpan.toString());
		sb.append(",");
		sb.append(simhashDist == null ? "?" : simhashDist.toString());
		sb.append(",");
		sb.append(cKWSRatio == null ? "?" : cKWSRatio.toString());
		sb.append(",");
//		sb.append(cKWSLength == null ? "?" : cKWSLength.toString());
//		sb.append(",");
		sb.append(tKWSRatio == null ? "?" : tKWSRatio.toString());
		sb.append(",");
//		sb.append(tKWSLength == null ? "?" : tKWSLength.toString());
//		sb.append(",");
		sb.append(hKWSRatio == null ? "?" : hKWSRatio.toString());
		sb.append(",");
//		sb.append(hKWSLength == null ? "?" : hKWSLength.toString());
//		sb.append(",");
		sb.append(channelRatio == null ? "?" : channelRatio.toString());
		sb.append(",");
//		sb.append(channelLength == null ? "?" : channelLength.toString());
//		sb.append(",");
		sb.append(cOrgRatioNE == null ? "?" : cOrgRatioNE.toString());
		sb.append(",");
		sb.append(cOrgRatioSP == null ? "?" : cOrgRatioSP.toString());
		sb.append(",");
//		sb.append(cOrgLengthNE == null ? "?" : cOrgLengthNE.toString());
//		sb.append(",");
//		sb.append(cOrgLengthSP == null ? "?" : cOrgLengthSP.toString());
//		sb.append(",");
		sb.append(cLocRatioNE == null ? "?" : cLocRatioNE.toString());
		sb.append(",");
		sb.append(cLocRatioSP == null ? "?" : cLocRatioSP.toString());
		sb.append(",");
//		sb.append(cLocLengthNE == null ? "?" : cLocLengthNE.toString());
//		sb.append(",");
//		sb.append(cLocLengthSP == null ? "?" : cLocLengthSP.toString());
//		sb.append(",");
		sb.append(cPerRatioNE == null ? "?" : cPerRatioNE.toString());
		sb.append(",");
		sb.append(cPerRatioSP == null ? "?" : cPerRatioSP.toString());
		sb.append(",");
//		sb.append(cPerLengthNE == null ? "?" : cPerLengthNE.toString());
//		sb.append(",");
//		sb.append(cPerLengthSP == null ? "?" : cPerLengthSP.toString());
//		sb.append(",");
		sb.append(catRatio == null ? "?" : catRatio.toString());
		sb.append(",");
//		sb.append(catLength == null ? "?" : catLength.toString());
//		sb.append(",");
		sb.append(geoRatio == null ? "?" : geoRatio.toString());
		sb.append(",");
//		sb.append(geoLength == null ? "?" : geoLength.toString());
//		sb.append(",");
		sb.append(StringUtils.isEmpty(label) ? "?" : label);
		return sb.toString();
	}
	
	
	public String toCSV() {
		StringBuilder sb = new StringBuilder();
		//Title Wide Feature
		sb.append(titleDist == null ?  "-1" : sparse2continuous(titleDist, new double[]{5, 15, 30}));
		sb.append(",");
		sb.append(titleRatio == null ?  "-1" : sparse2continuous(titleRatio, new double[]{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}));
		sb.append(",");
		//Title Cross Feature
		sb.append(titleRatio == null ?  "0" : sparse2continuous(titleRatio, new double[]{0.6}));
		sb.append(",");
		sb.append(titleLength == null ?  "0" : sparse2continuous(titleLength, new double[]{5}));
		sb.append(",");
		//Wide Feature
		sb.append(epochSpan == null ?  "-1" : sparse2continuous(epochSpan, new double[]{86400}));
		sb.append(",");
		sb.append(insertSpan == null ?  "-1" : sparse2continuous(insertSpan, new double[]{86400}));
		sb.append(",");
		sb.append(simhashDist == null ?  "-1" : sparse2continuous(simhashDist, new double[]{20, 40, 60, 80}));
		sb.append(",");
		//Content Words Wide Feature
		sb.append(cKWSRatio == null ?  "-1" : sparse2continuous(cKWSRatio, new double[]{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}));
		sb.append(",");
		//Content Words Cross Feature
		sb.append(cKWSRatio == null ?  "0" : sparse2continuous(cKWSRatio, new double[]{0.4}));
		sb.append(",");
		sb.append(cKWSLength == null ?  "0" : sparse2continuous(cKWSLength, new double[]{6}));
		sb.append(",");
		//Highlight Words Wide Feature
		sb.append(hKWSRatio == null ?  "-1" : sparse2continuous(hKWSRatio, new double[]{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}));
		sb.append(",");
		//Highlight Words Cross Feature
		sb.append(hKWSRatio == null ?  "0" : sparse2continuous(hKWSRatio, new double[]{0.4}));
		sb.append(",");
		sb.append(hKWSLength == null ?  "0" : sparse2continuous(hKWSLength, new double[]{3}));
		sb.append(",");
		//Channels Wide Feature
		sb.append(channelRatio == null ?  "-1" : sparse2continuous(channelRatio, new double[]{0.3, 0.6}));
		sb.append(",");
		//Channels Cross Feature
		sb.append(channelRatio == null ?  "0" : sparse2continuous(channelRatio, new double[]{0.5}));
		sb.append(",");
		sb.append(channelLength == null ?  "0" : sparse2continuous(channelLength, new double[]{3}));
		sb.append(",");
		//Content Organization Wide Feature
		sb.append(cOrgRatioNE == null ?  "-1" : sparse2continuous(cOrgRatioNE, new double[]{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}));
		sb.append(",");
		sb.append(cOrgRatioSP == null ?  "-1" : sparse2continuous(cOrgRatioSP, new double[]{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}));
		sb.append(",");
		//Content Organization Cross Feature
		sb.append(cOrgRatioNE == null ?  "0" : sparse2continuous(cOrgRatioNE, new double[]{0.5}));
		sb.append(",");
		sb.append(cOrgLengthNE == null ?  "0" : sparse2continuous(cOrgLengthNE, new double[]{6}));
		sb.append(",");
		sb.append(cOrgRatioSP == null ?  "0" : sparse2continuous(cOrgRatioSP, new double[]{0.5}));
		sb.append(",");
		sb.append(cOrgLengthSP == null ?  "0" : sparse2continuous(cOrgLengthSP, new double[]{6}));
		sb.append(",");
		//Content Location Wide Feature
		sb.append(cLocRatioNE == null ?  "-1" : sparse2continuous(cLocRatioNE, new double[]{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}));
		sb.append(",");
		sb.append(cLocRatioSP == null ?  "-1" : sparse2continuous(cLocRatioSP, new double[]{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}));
		sb.append(",");
		//Content Location Cross Feature
		sb.append(cLocRatioNE == null ?  "0" : sparse2continuous(cLocRatioNE, new double[]{0.5}));
		sb.append(",");
		sb.append(cLocLengthNE == null ?  "0" : sparse2continuous(cLocLengthNE, new double[]{6}));
		sb.append(",");
		sb.append(cLocRatioSP == null ?  "0" : sparse2continuous(cLocRatioSP, new double[]{0.5}));
		sb.append(",");
		sb.append(cLocLengthSP == null ?  "0" : sparse2continuous(cLocLengthSP, new double[]{6}));
		sb.append(",");
		//Content Person Wide Feature
		sb.append(cPerRatioNE == null ?  "-1" : sparse2continuous(cPerRatioNE, new double[]{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}));
		sb.append(",");
		sb.append(cPerRatioSP == null ?  "-1" : sparse2continuous(cPerRatioSP, new double[]{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}));
		sb.append(",");
		//Content Person Cross Feature
		sb.append(cPerRatioNE == null ?  "0" : sparse2continuous(cPerRatioNE, new double[]{0.5}));
		sb.append(",");
		sb.append(cPerLengthNE == null ?  "0" : sparse2continuous(cPerLengthNE, new double[]{4}));
		sb.append(",");
		sb.append(cPerRatioSP == null ?  "0" : sparse2continuous(cPerRatioSP, new double[]{0.5}));
		sb.append(",");
		sb.append(cPerLengthSP == null ?  "0" : sparse2continuous(cPerLengthSP, new double[]{4}));
		sb.append(",");
		//Content NUM Wide Feature
		sb.append(cNUMRatioSP == null ?  "-1" : sparse2continuous(cNUMRatioSP, new double[]{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}));
		sb.append(",");
		//Content NUM Cross Feature
		sb.append(cNUMRatioSP == null ?  "0" : sparse2continuous(cNUMRatioSP, new double[]{0.5}));
		sb.append(",");
		sb.append(cNUMLengthSP == null ?  "0" : sparse2continuous(cNUMLengthSP, new double[]{4}));
		sb.append(",");
		//Content TIM Wide Feature
		sb.append(cTimRatioSP == null ?  "-1" : sparse2continuous(cTimRatioSP, new double[]{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}));
		sb.append(",");
		//Content TIM Cross Feature
		sb.append(cTimRatioSP == null ?  "0" : sparse2continuous(cTimRatioSP, new double[]{0.5}));
		sb.append(",");
		sb.append(cTimLengthSP == null ?  "0" : sparse2continuous(cTimLengthSP, new double[]{4}));
		sb.append(",");
		//Category Wide Feature
		sb.append(catRatio == null ?  "-1" : sparse2continuous(catRatio, new double[]{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}));
		sb.append(",");
		//Category Cross Feature : 用于降权!!!!
		//Sports + Celebrities : Organization + Location + Person + KWS
		if ((categoryIndex == 1 || categoryIndex == 4 || categoryIndex == 5) &&
			(
				(cOrgRatioSP != null && cOrgRatioSP < 0.4 && cOrgLengthSP > 5) ||
				(cOrgRatioNE != null && cOrgRatioNE < 0.4 && cOrgLengthNE > 5) ||
				(cLocRatioSP != null && cLocRatioSP < 0.4 && cLocLengthSP > 5) ||
				(cLocRatioNE != null && cLocRatioNE < 0.4 && cLocLengthNE > 5) ||
				(cPerRatioSP != null && cPerRatioSP < 0.4 && cPerLengthSP > 5) ||
				(cPerRatioNE != null && cPerRatioNE < 0.4 && cPerLengthNE > 5) ||
				(cKWSRatio != null && cKWSRatio < 0.4 && cKWSLength > 5)
			)
		) {
			sb.append("1,1,");
		} else if (categoryIndex == 2 && //Economy Markets : ORG + NUM + TIM + KWS
			(
				(cNUMRatioSP != null && cNUMRatioSP < 0.4 && cNUMLengthSP > 5) ||
				(cTimRatioSP != null && cTimRatioSP < 0.4 && cTimLengthSP > 5) ||
				(cKWSRatio != null && cKWSRatio < 0.4 && cKWSLength > 5)
			)
		) {
			sb.append("1,1,");
		} else if (categoryIndex == 8 && //Weather : Location
			(
				(cLocRatioSP != null && cLocRatioSP < 0.4 && cLocLengthSP > 5) ||
				(cLocRatioNE != null && cLocRatioNE < 0.4 && cLocLengthNE > 5) ||
				(geoRatio != null && geoRatio < 0.4)
			)
		) {
			sb.append("1,1,");
		} else {
			sb.append("0,0,");
		}
		//Category Continuous Feature
		sb.append(String.valueOf(categoryIndex));
		return sb.toString();
	}
	
	
	public int sparse2continuous(double srcCode, double[] thrArray) {
		int result = thrArray.length;
		for (int i = 0; i < thrArray.length; i++) {
			if (srcCode < thrArray[i]) {
				return i;
			}
		}
		return result;
	}
	
	
	public static void main(String[] args) throws Exception {
		ObjectMapper mapper = new ObjectMapper();
		String mStr = "{\"_id\": \"0X6y8ysO\", \"c_word\": 955, \"channels\": [\"Endangered^^Species^^Act\", \"Wildlife^^Species\", \"Protected^^Species\", \"Animal^^Species\", \"Wildlife^^Service\"], \"channels_v2\": [\"endangered^^species\", \"woodpeckers\", \"woodpecker\", \"birds\", \"arkansas\"], \"epoch\": {\"$numberLong\": \"1601057829\"}, \"faces_phash\": [], \"geotag\": [], \"geotag_v2\": [], \"highlightkeyword_list\": [[\"David^^Bernhardt\", 0.002340801198350316], [\"Donald^^Trump\", 0.001787553661137706], [\"Fort^^Benning\", 0.0014867698661778419], [\"Noah^^Greenwald\", 0.0013646526756377637], [\"Joe^^Biden\", 0.0010302173560934457], [\"Ben^^Prater\", 0.0010030390601234737], [\"Sonny^^Purdue\", 8.410014017568528E-4], [\"Jeff^^Walters\", 7.769921688940232E-4], [\"Aurelia^^Skipwith\", 5.878020406668937E-4], [\"Ellen^^Knickmeyer\", 9.829738328068632E-5], [\"Georgia\", 9.026110668524878E-7], [\"Texas\", 8.785599005270214E-7], [\"Kentucky\", 8.765769300669047E-7], [\"Arkansas\", 8.748975339580471E-7], [\"Missouri\", 8.545596845427886E-7], [\"Florida\", 8.144286442485093E-7], [\"Tennessee\", 7.844627359718063E-7], [\"Agriculture\", 7.561516582406097E-7]], \"images_phash\": [\"0xac5827b3586cceb1\"], \"insert_time\": \"2020-09-26 02:03:58\", \"kw_title\": [\"woodpecker^^houses\", \"woodpecker^^holes\", \"protections\", \"Feds\"], \"kws\": [\"woodpecker^^houses\", \"woodpecker^^holes\", \"Species^^Act^^rules\", \"protections\", \"Wildlife^^Service\", \"red-cockaded^^woodpeckers\", \"species^^director\", \"Feds\", \"extinction\", \"landowners\", \"live^^trees\", \"safe^^harbor^^agreements\", \"coastal^^states\", \"minimal^^federal^^oversight\", \"nestlings\", \"automatic^^continued^^protection\", \"Fire^^suppression\", \"insecticides\", \"Scientists\", \"Biological^^Diversity\"], \"ne_content_location\": {\"Oklahoma City\": 1, \"southern Virginia\": 1, \"Fort Benning\": 1, \"Tennessee\": 1, \"Oklahoma\": 1, \"Kentucky\": 1, \"Florida\": 1, \"Arkansas\": 1, \"New Jersey\": 1, \"South Carolina\": 1, \"Texas\": 2, \"Missouri\": 1, \"New Orleans\": 1, \"Georgia\": 2}, \"ne_content_organization\": {\"Virginia Tech\": 1, \"U.S. Fish and Wildlife Service\": 1, \"Interior\": 1, \"Agriculture\": 1, \"National Resource Conservation Service\": 1, \"Center for Biological Diversity\": 1, \"Federal Register\": 1, \"Wildlife Service\": 1, \"Trump\": 1}, \"ne_content_person\": {\"Jeff Walters\": 1, \"Donald Trump\": 1, \"Joe Biden\": 1, \"David Bernhardt\": 1, \"Trump\": 2, \"Greenwald\": 1, \"McConnaughey\": 1, \"Noah Greenwald\": 1, \"Bernhardt\": 2, \"Ellen Knickmeyer\": 1, \"Biden\": 1, \"Ben Prater\": 1, \"Walters\": 1, \"Aurelia Skipwith\": 1, \"Sonny Purdue\": 1}, \"ne_title_location\": {}, \"ne_title_organization\": {}, \"ne_title_person\": {}, \"paragraph_count\": 24.0, \"simhash\": \"849f2b8fd8fb07814ec4436c8e712dd2\", \"spacy_content_loc\": [\"Fort Benning\", \"Georgia\", \"Georgia\", \"New Jersey\", \"Florida\", \"Texas\", \"Missouri\", \"Kentucky\", \"Tennessee\", \"Virginia\", \"Texas\", \"Arkansas\", \"Oklahoma\", \"South Carolina\"], \"spacy_content_num\": [\"11\", \"one\", \"13\", \"one\", \"5.1\", \"$ 408 million\", \"one\", \"90 million acres\", \"about 3 million\", \"36.4 million hectares\", \"1.2 million\", \"only 1,470\", \"nearly 7,800\", \"two\", \"87 percent\", \"second\", \"One\", \"more than 1.3 million acres -LRB- 526,000 hectares\"], \"spacy_content_org\": [\"the U.S. Fish and Wildlife Service\", \"Interior\", \"Agriculture\", \"Defenders of Wildlife\", \"the Center for Biological Diversity\", \"Trump\", \"the Fish and Wildlife Service\", \"Virginia Tech\", \"Trump\", \"Bernhardt\", \"the Federal Register\", \"Cabinet\", \"Democrat\", \"Trump\", \"Biden\", \"North American\"], \"spacy_content_per\": [\"Aurelia Skipwith\", \"David Bernhardt\", \"Sonny Purdue\", \"Ben Prater\", \"Noah Greenwald\", \"Bernhardt\", \"Greenwald\", \"Jeff Walters\", \"Donald Trump\", \"Joe Biden\"], \"spacy_content_tim\": [\"1970\", \"today\", \"Friday\", \"today\", \"Friday\", \"Friday\", \"April\", \"2016\", \"1998\", \"2016\", \"years\", \"the late 1970s\", \"1989\"], \"spacy_title_loc\": [], \"spacy_title_num\": [], \"spacy_title_org\": [], \"spacy_title_per\": [], \"spacy_title_tim\": [\"1970\"], \"src\": \"Meridian Star\", \"stitle\": \"Feds : Relax protections for woodpecker endangered since 1970\", \"text_category\": {\"first_cat\": {\"PoliticsGovernment\": 0.5332192989164418, \"Lifestyle\": 0.5620721341978404}, \"second_cat\": {\"PoliticsGovernment_Federal\": 0.8, \"Lifestyle_PetsAnimals\": 0.5620721341978404}, \"third_cat\": {\"PoliticsGovernment_Federal_POTUS\": 0.8, \"Lifestyle_PetsAnimals_Animals\": 0.5707528027362845}}, \"text_category_v2\": {\"first_cat\": {\"PoliticsGovernment\": 0.5332192989164418, \"Lifestyle\": 0.5620721341978404}, \"second_cat\": {\"PoliticsGovernment_Federal\": 0.8, \"Lifestyle_PetsAnimals\": 0.5620721341978404}, \"third_cat\": {\"PoliticsGovernment_Federal_POTUS\": 0.8, \"Lifestyle_PetsAnimals_Animals\": 0.5707528027362845}}, \"url\": \"https://www.meridianstar.com/region/feds-relax-protections-for-woodpecker-endangered-since-1970/article_84cc769a-3a7c-579a-be63-5a751bfbe9ee.html\"}";
		String cStr = "{\"_id\": \"0X6dJYLJ\", \"c_word\": 953, \"channels\": [\"Endangered^^Species^^Act\", \"Wildlife^^Species\", \"Protected^^Species\", \"Animal^^Species\", \"Wildlife^^Service\"], \"channels_v2\": [\"endangered^^species\", \"woodpeckers\", \"woodpecker\", \"u.s.^^fish^^and^^wildlife^^service\", \"fish^^and^^wildlife^^service\"], \"epoch\": {\"$numberLong\": \"1601048591\"}, \"faces_phash\": [], \"geotag\": [], \"geotag_v2\": [], \"highlightkeyword_list\": [[\"David^^Bernhardt\", 0.002340801198350316], [\"Donald^^Trump\", 0.001787553661137706], [\"Fort^^Benning\", 0.0014867698661778419], [\"Noah^^Greenwald\", 0.0013646526756377637], [\"Joe^^Biden\", 0.0010302173560934457], [\"Ben^^Prater\", 0.0010030390601234737], [\"Sonny^^Purdue\", 8.410014017568528E-4], [\"Jeff^^Walters\", 7.769921688940232E-4], [\"Aurelia^^Skipwith\", 5.878020406668937E-4], [\"Ellen^^Knickmeyer\", 9.829738328068632E-5], [\"Georgia\", 9.026110668524878E-7], [\"Texas\", 8.785599005270214E-7], [\"Kentucky\", 8.765769300669047E-7], [\"Arkansas\", 8.748975339580471E-7], [\"Missouri\", 8.545596845427886E-7], [\"Florida\", 8.144286442485093E-7], [\"Tennessee\", 7.844627359718063E-7], [\"Agriculture\", 7.561516582406097E-7]], \"images_phash\": [\"0xac5927b3486cceb1\"], \"insert_time\": \"2020-09-26 09:57:31\", \"kw_title\": [\"woodpecker^^houses\", \"woodpecker^^holes\", \"protections\", \"Feds\"], \"kws\": [\"woodpecker^^houses\", \"woodpecker^^holes\", \"Species^^Act^^rules\", \"protections\", \"Wildlife^^Service\", \"red-cockaded^^woodpeckers\", \"species^^director\", \"Feds\", \"extinction\", \"landowners\", \"live^^trees\", \"safe^^harbor^^agreements\", \"coastal^^states\", \"minimal^^federal^^oversight\", \"nestlings\", \"automatic^^continued^^protection\", \"Fire^^suppression\", \"insecticides\", \"Scientists\", \"Biological^^Diversity\"], \"ne_content_location\": {\"Oklahoma City\": 1, \"southern Virginia\": 1, \"Fort Benning\": 1, \"Tennessee\": 1, \"Oklahoma\": 1, \"Kentucky\": 1, \"Florida\": 1, \"Arkansas\": 1, \"New Jersey\": 1, \"South Carolina\": 1, \"Texas\": 2, \"Missouri\": 1, \"New Orleans\": 1, \"Georgia\": 2}, \"ne_content_organization\": {\"Virginia Tech\": 1, \"U.S. Fish and Wildlife Service\": 1, \"Interior\": 1, \"Agriculture\": 1, \"National Resource Conservation Service\": 1, \"Center for Biological Diversity\": 1, \"Federal Register\": 1, \"Wildlife Service\": 1}, \"ne_content_person\": {\"Jeff Walters\": 1, \"Donald Trump\": 1, \"Joe Biden\": 1, \"David Bernhardt\": 1, \"Trump\": 3, \"Greenwald\": 1, \"McConnaughey\": 1, \"Noah Greenwald\": 1, \"Bernhardt\": 2, \"Ellen Knickmeyer\": 1, \"Biden\": 1, \"Ben Prater\": 1, \"Walters\": 1, \"Aurelia Skipwith\": 1, \"Sonny Purdue\": 1}, \"ne_title_location\": {}, \"ne_title_organization\": {}, \"ne_title_person\": {}, \"paragraph_count\": 24.0, \"simhash\": \"849f238fd8fb07814ec4436c8e712dd2\", \"spacy_content_loc\": [\"Fort Benning\", \"Georgia\", \"Georgia\", \"New Jersey\", \"Florida\", \"Texas\", \"Missouri\", \"Kentucky\", \"Tennessee\", \"Virginia\", \"Texas\", \"Arkansas\", \"Oklahoma\", \"South Carolina\"], \"spacy_content_num\": [\"11\", \"one\", \"13\", \"one\", \"5.1\", \"$ 408 million\", \"one\", \"90 million acres\", \"about 3 million\", \"36.4 million hectares\", \"1.2 million\", \"only 1,470\", \"nearly 7,800\", \"two\", \"87 percent\", \"second\", \"One\", \"more than 1.3 million acres -LRB- 526,000 hectares\"], \"spacy_content_org\": [\"the U.S. Fish and Wildlife Service\", \"Interior\", \"Agriculture\", \"Defenders of Wildlife\", \"the Center for Biological Diversity\", \"Trump\", \"the Fish and Wildlife Service\", \"Virginia Tech\", \"Trump\", \"Bernhardt\", \"the Federal Register\", \"Cabinet\", \"Democrat\", \"Trump\", \"Biden\", \"North American\"], \"spacy_content_per\": [\"Aurelia Skipwith\", \"David Bernhardt\", \"Sonny Purdue\", \"Ben Prater\", \"Noah Greenwald\", \"Bernhardt\", \"Greenwald\", \"Jeff Walters\", \"Donald Trump\", \"Joe Biden\"], \"spacy_content_tim\": [\"1970\", \"today\", \"Friday\", \"today\", \"Friday\", \"Friday\", \"April\", \"2016\", \"1998\", \"2016\", \"years\", \"the late 1970s\", \"1989\"], \"spacy_title_loc\": [], \"spacy_title_num\": [], \"spacy_title_org\": [], \"spacy_title_per\": [], \"spacy_title_tim\": [\"1970\"], \"src\": \"Beaumont Enterprise\", \"stitle\": \"Feds : Relax protections for woodpecker endangered since 1970\", \"text_category\": {\"first_cat\": {\"PoliticsGovernment\": 0.5332192989164418, \"Lifestyle\": 0.5620721341978404}, \"second_cat\": {\"PoliticsGovernment_Federal\": 0.8, \"Lifestyle_PetsAnimals\": 0.5620721341978404}, \"third_cat\": {\"PoliticsGovernment_Federal_POTUS\": 0.8, \"Lifestyle_PetsAnimals_Animals\": 0.5707528027362845}}, \"text_category_v2\": {\"first_cat\": {\"PoliticsGovernment\": 0.5332192989164418, \"Lifestyle\": 0.5620721341978404}, \"second_cat\": {\"PoliticsGovernment_Federal\": 0.8, \"Lifestyle_PetsAnimals\": 0.5620721341978404}, \"third_cat\": {\"PoliticsGovernment_Federal_POTUS\": 0.8, \"Lifestyle_PetsAnimals_Animals\": 0.5707528027362845}}, \"url\": \"https://www.beaumontenterprise.com/news/article/Feds-Relax-protections-for-woodpecker-endangered-15597040.php\"}";
		JsonNode mNode = mapper.readTree(mStr);
		JsonNode cNode = mapper.readTree(cStr);
		EventFeature feature = new EventFeature(mNode, cNode, "");
		System.out.println(mapper.writeValueAsString(feature));
		System.out.println(feature.toInstanceV1());
		//
	}
}
