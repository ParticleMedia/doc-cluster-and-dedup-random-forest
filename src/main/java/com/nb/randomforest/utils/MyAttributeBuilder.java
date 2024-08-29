package com.nb.randomforest.utils;

import java.util.ArrayList;

import weka.core.Attribute;

/**
 * @author yuxi
 * @date 2020/11/12
 */
public class MyAttributeBuilder {
	
	public static ArrayList<Attribute> buildMyAttributesV0() {
		ArrayList<Attribute> attributes = new ArrayList<>();
		attributes.add(new Attribute("TitleDist"));
		attributes.add(new Attribute("SameSRC"));
		attributes.add(new Attribute("CWordSpan"));
		attributes.add(new Attribute("EpochSpan"));
		attributes.add(new Attribute("ParagraphSpan"));
		attributes.add(new Attribute("SimhashDist"));
		attributes.add(new Attribute("CKWSRatio"));
		attributes.add(new Attribute("ChannelRatio"));
		attributes.add(new Attribute("COrgRatioNE"));
		attributes.add(new Attribute("CLocRatioNE"));
		attributes.add(new Attribute("CPrsRatioNE"));
		attributes.add(new Attribute("TOrgRatioNE"));
		attributes.add(new Attribute("TLocRatioNE"));
		attributes.add(new Attribute("TPerRatioNE"));
		attributes.add(new Attribute("CatRatio"));
		attributes.add(new Attribute("GeoRatio"));
		// - nominal
		ArrayList<String> attVals = new ArrayList<>();
		attVals.add("DIFF");
		attVals.add("EVENT");
		attributes.add(new Attribute("Label", attVals));
		return attributes;
	}
	
	public static ArrayList<Attribute> buildMyAttributesV1() {
		ArrayList<Attribute> attributes = new ArrayList<>();
		attributes.add(new Attribute("TitleDist"));
		attributes.add(new Attribute("TitleRatio"));
		attributes.add(new Attribute("TitleLength"));
		attributes.add(new Attribute("SameSRC"));
		attributes.add(new Attribute("CWordSpan"));
		attributes.add(new Attribute("ParagraphSpan"));
		attributes.add(new Attribute("EpochSpan"));
		attributes.add(new Attribute("InsertSpan"));
		attributes.add(new Attribute("SimhashDist"));
		//新增特征 - 2021-03-03 img,face hash dist
		attributes.add(new Attribute("ImgHashDist"));
		attributes.add(new Attribute("FaceHashDist"));
		//
		attributes.add(new Attribute("CKWSRatio"));
		attributes.add(new Attribute("CKWSLength"));
		attributes.add(new Attribute("TKWSRatio"));
		attributes.add(new Attribute("TKWSLength"));
		attributes.add(new Attribute("HKWSRatio"));
		attributes.add(new Attribute("HKWSLength"));
		attributes.add(new Attribute("ChannelRatio"));
		attributes.add(new Attribute("ChannelLength"));
		attributes.add(new Attribute("COrgRatioNE"));
		attributes.add(new Attribute("COrgRatioSP"));
		attributes.add(new Attribute("COrgLengthNE"));
		attributes.add(new Attribute("COrgLengthSP"));
		attributes.add(new Attribute("CLocRatioNE"));
		attributes.add(new Attribute("CLocRatioSP"));
		attributes.add(new Attribute("CLocLengthNE"));
		attributes.add(new Attribute("CLocLengthSP"));
		attributes.add(new Attribute("CPerRatioNE"));
		attributes.add(new Attribute("CPerRatioSP"));
		attributes.add(new Attribute("CPerLengthNE"));
		attributes.add(new Attribute("CPerLengthSP"));
		attributes.add(new Attribute("CNUMRatioSP"));
		attributes.add(new Attribute("CNUMLengthSP"));
		attributes.add(new Attribute("CTimRatioSP"));
		attributes.add(new Attribute("CTimLengthSP"));
		attributes.add(new Attribute("TOrgRatioNE"));
		attributes.add(new Attribute("TOrgRatioSP"));
		attributes.add(new Attribute("TOrgLengthNE"));
		attributes.add(new Attribute("TOrgLengthSP"));
		attributes.add(new Attribute("TLocRatioNE"));
		attributes.add(new Attribute("TLocRatioSP"));
		attributes.add(new Attribute("TLocLengthNE"));
		attributes.add(new Attribute("TLocLengthSP"));
		attributes.add(new Attribute("TPerRatioNE"));
		attributes.add(new Attribute("TPerRatioSP"));
		attributes.add(new Attribute("TPerLengthNE"));
		attributes.add(new Attribute("TPerLengthSP"));
		attributes.add(new Attribute("TNUMRatioSP"));
		attributes.add(new Attribute("TNUMLengthSP"));
		attributes.add(new Attribute("TTimRatioSP"));
		attributes.add(new Attribute("TTimLengthSP"));
		attributes.add(new Attribute("CatRatio"));
		attributes.add(new Attribute("CatLength"));
		attributes.add(new Attribute("GeoRatio"));
		attributes.add(new Attribute("GeoLength"));
		// - nominal
		ArrayList<String> attVals = new ArrayList<>();
		attVals.add("DIFF");
		attVals.add("EVENT");
		attributes.add(new Attribute("Label", attVals));
		return attributes;
	}
	
	
	public static ArrayList<Attribute> buildMyAttributesV2() {
		ArrayList<Attribute> attributes = new ArrayList<>();
		attributes.add(new Attribute("TitleRatio"));
//		attributes.add(new Attribute("TitleLength"));
		attributes.add(new Attribute("SameSRC"));
		attributes.add(new Attribute("EpochSpan"));
		attributes.add(new Attribute("InsertSpan"));
		attributes.add(new Attribute("SimhashDist"));
		attributes.add(new Attribute("CKWSRatio"));
//		attributes.add(new Attribute("CKWSLength"));
		attributes.add(new Attribute("TKWSRatio"));
//		attributes.add(new Attribute("TKWSLength"));
		attributes.add(new Attribute("HKWSRatio"));
//		attributes.add(new Attribute("HKWSLength"));
		attributes.add(new Attribute("ChannelRatio"));
//		attributes.add(new Attribute("ChannelLength"));
		attributes.add(new Attribute("COrgRatioNE"));
		attributes.add(new Attribute("COrgRatioSP"));
//		attributes.add(new Attribute("COrgLengthNE"));
//		attributes.add(new Attribute("COrgLengthSP"));
		attributes.add(new Attribute("CLocRatioNE"));
		attributes.add(new Attribute("CLocRatioSP"));
//		attributes.add(new Attribute("CLocLengthNE"));
//		attributes.add(new Attribute("CLocLengthSP"));
		attributes.add(new Attribute("CPrsRatioNE"));
		attributes.add(new Attribute("CPrsRatioSP"));
//		attributes.add(new Attribute("CPrsLengthNE"));
//		attributes.add(new Attribute("CPrsLengthSP"));
		attributes.add(new Attribute("CatRatio"));
//		attributes.add(new Attribute("CatLength"));
		attributes.add(new Attribute("GeoRatio"));
//		attributes.add(new Attribute("GeoLength"));
		// - nominal
		ArrayList<String> attVals = new ArrayList<>();
		attVals.add("DIFF");
		attVals.add("EVENT");
		attributes.add(new Attribute("Label", attVals));
		return attributes;
	}
	
	public static ArrayList<Attribute> buildMyAttributesV3() {
		ArrayList<Attribute> attributes = new ArrayList<>();
		attributes.add(new Attribute("TitleDist"));
		attributes.add(new Attribute("TitleRatio"));
		attributes.add(new Attribute("TitleLength"));
		attributes.add(new Attribute("SameSRC"));
		attributes.add(new Attribute("CWordSpan"));
		attributes.add(new Attribute("ParagraphSpan"));
		attributes.add(new Attribute("EpochSpan"));
		attributes.add(new Attribute("InsertSpan"));
		attributes.add(new Attribute("SimhashDist"));
		//新增特征 - 2021-03-03 img,face hash dist
		attributes.add(new Attribute("ImgHashDist"));
		attributes.add(new Attribute("FaceHashDist"));
		//
		attributes.add(new Attribute("CKWSRatio"));
		attributes.add(new Attribute("CKWSLength"));
		attributes.add(new Attribute("TKWSRatio"));
		attributes.add(new Attribute("TKWSLength"));
		attributes.add(new Attribute("HKWSRatio"));
		attributes.add(new Attribute("HKWSLength"));
		attributes.add(new Attribute("ChannelRatio"));
		attributes.add(new Attribute("ChannelLength"));
		attributes.add(new Attribute("COrgRatioNE"));
		attributes.add(new Attribute("COrgRatioSP"));
		attributes.add(new Attribute("COrgLengthNE"));
		attributes.add(new Attribute("COrgLengthSP"));
		attributes.add(new Attribute("CLocRatioNE"));
		attributes.add(new Attribute("CLocRatioSP"));
		attributes.add(new Attribute("CLocLengthNE"));
		attributes.add(new Attribute("CLocLengthSP"));
		attributes.add(new Attribute("CPerRatioNE"));
		attributes.add(new Attribute("CPerRatioSP"));
		attributes.add(new Attribute("CPerLengthNE"));
		attributes.add(new Attribute("CPerLengthSP"));
		attributes.add(new Attribute("CNUMRatioSP"));
		attributes.add(new Attribute("CNUMLengthSP"));
		attributes.add(new Attribute("CTimRatioSP"));
		attributes.add(new Attribute("CTimLengthSP"));
		attributes.add(new Attribute("TOrgRatioNE"));
		attributes.add(new Attribute("TOrgRatioSP"));
		attributes.add(new Attribute("TOrgLengthNE"));
		attributes.add(new Attribute("TOrgLengthSP"));
		attributes.add(new Attribute("TLocRatioNE"));
		attributes.add(new Attribute("TLocRatioSP"));
		attributes.add(new Attribute("TLocLengthNE"));
		attributes.add(new Attribute("TLocLengthSP"));
		attributes.add(new Attribute("TPerRatioNE"));
		attributes.add(new Attribute("TPerRatioSP"));
		attributes.add(new Attribute("TPerLengthNE"));
		attributes.add(new Attribute("TPerLengthSP"));
		attributes.add(new Attribute("TNUMRatioSP"));
		attributes.add(new Attribute("TNUMLengthSP"));
		attributes.add(new Attribute("TTimRatioSP"));
		attributes.add(new Attribute("TTimLengthSP"));
		attributes.add(new Attribute("CatRatio"));
		attributes.add(new Attribute("CatLength"));
		attributes.add(new Attribute("GeoRatio"));
		attributes.add(new Attribute("GeoLength"));
		// - nominal
		ArrayList<String> attVals = new ArrayList<>();
		attVals.add("DIFF");
		attVals.add("EVENT");
		attVals.add("DUP");
		attributes.add(new Attribute("Label", attVals));
		return attributes;
	}
	
	public static void main(String[] args) {
	
	}
}
