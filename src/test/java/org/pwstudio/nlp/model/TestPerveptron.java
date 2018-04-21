package org.pwstudio.nlp.model;

import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.util.ArrayList;

import org.junit.Test;
import org.pwstudio.nlp.model.perceptron.StructuredPerceptronModel;
import org.pwstudio.nlp.model.tagger.FeatureTemplate;
import org.pwstudio.nlp.model.tagger.Table;

import com.hankcs.hanlp.corpus.io.ByteArray;


public class TestPerveptron
{

    public StructuredPerceptronModel createSegmentModel()
    {
        String [] tags = {"S","B","I","E"};
        ArrayList<FeatureTemplate> templateList = new ArrayList<FeatureTemplate>(7);
        templateList.add(FeatureTemplate.create("%x[-1,0]:U1"));
        templateList.add(FeatureTemplate.create("%x[0,0]:U2"));
        templateList.add(FeatureTemplate.create("%x[1,0]:U3"));
        templateList.add(FeatureTemplate.create("%x[-2,0]/%x[-1,0]:U4"));
        templateList.add(FeatureTemplate.create("%x[-1,0]/%x[0,0]:U5"));
        templateList.add(FeatureTemplate.create("%x[0,0]/%x[1,0]:U6"));
        templateList.add(FeatureTemplate.create("%x[1,0]/%x[2,0]:U7"));
        StructuredPerceptronModel model = new StructuredPerceptronModel(tags, templateList);
        return model;
    }
    
    /**
     * @param model 创建好的模型对象
     * @param iteration 迭代次数
     */
    public void trainPerceptronModel(StructuredPerceptronModel model, String path, DataSetIterator iterator, 
            int iteration, boolean average) throws Exception
    {
        if (new File(path).exists())
        {
            ByteArray array = ByteArray.createByteArray(path);
            model.load(array);
        }
        Table instance = new Table();
        for (int i=0; i<iteration; i++) 
        {
            // TODO 打乱训练顺序
            while (iterator.hasNext()) 
            {
                instance.v = iterator.next();
                model.train(instance);
            }
            iterator.reset();
        }
        if (average)
            model.average();
        DataOutputStream out = new DataOutputStream(new FileOutputStream(path));
        model.save(out);
        out.close();
    }
    
    @Test
    public void test() {
        try
        {
            File path = new File("data/models/segment");   
            if (!path.exists())
                path.mkdirs();
            StructuredPerceptronModel model = createSegmentModel();
            DataSetIterator iterator = new FileDataSetIterator("train_corpus.utf8");
            trainPerceptronModel(model,"data/models/segment/SAPSegmentModel.bin", iterator, 10, true);
        }
        catch (Exception e)
        {
            e.printStackTrace();
        }
    }
}
