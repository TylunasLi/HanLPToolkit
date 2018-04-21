package org.pwstudio.nlp.seg.sp;

import java.util.ArrayList;
import java.util.List;

import org.pwstudio.nlp.model.perceptron.StructuredPerceptronModel;
import org.pwstudio.nlp.model.tagger.FeatureTemplate;

public class PerceptronSegmentModel extends StructuredPerceptronModel
{

    private PerceptronSegmentModel(String[] id2tag, List<FeatureTemplate> templateList)
    {
        super(id2tag, templateList);
    }

    /**
     * 参数通过静态方法传入
     * @return
     */
    public static PerceptronSegmentModel create()
    {
        String [] tags = {"S","B","M","E"};
        ArrayList<FeatureTemplate> templateList = new ArrayList<FeatureTemplate>(7);
        templateList.add(FeatureTemplate.create("%x[-1,0]:U1"));
        templateList.add(FeatureTemplate.create("%x[0,0]:U2"));
        templateList.add(FeatureTemplate.create("%x[1,0]:U3"));
        templateList.add(FeatureTemplate.create("%x[-2,0]/%x[-1,0]:U4"));
        templateList.add(FeatureTemplate.create("%x[-1,0]/%x[0,0]:U5"));
        templateList.add(FeatureTemplate.create("%x[0,0]/%x[1,0]:U6"));
        templateList.add(FeatureTemplate.create("%x[1,0]/%x[2,0]:U7"));
        PerceptronSegmentModel model = new PerceptronSegmentModel(tags, templateList);
        return model;
    }

    
}
