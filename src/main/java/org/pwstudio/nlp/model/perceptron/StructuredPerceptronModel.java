package org.pwstudio.nlp.model.perceptron;

import static com.hankcs.hanlp.utility.Predefine.logger;

import java.io.DataOutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

import org.pwstudio.nlp.model.tagger.FeatureTemplate;
import org.pwstudio.nlp.model.tagger.Table;

import com.hankcs.hanlp.collection.trie.ITrie;
import com.hankcs.hanlp.collection.trie.bintrie.BinTrie;
import com.hankcs.hanlp.corpus.io.ByteArray;
import com.hankcs.hanlp.corpus.io.ICacheAble;
import com.hankcs.hanlp.utility.TextUtility;

/**
 * 结构化感知机模型
 * 
 * @author TylunasLi
 */
public class StructuredPerceptronModel implements ICacheAble
{
    /**
     * 标签和id的相互转换
     */
    Map<String, Integer> tag2id;
    /**
     * id转标签
     */
    protected String[] id2tag;
    /**
     * 特征模板
     */
    List<FeatureTemplate> featureTemplateList;
    /**
     * 特征函数及权重集合
     */
    ITrie<FeatureFunctionWeights> featureFunctionTrie;
    /**
     * 训练特征函数集合
     */
    BinTrie<FeatureFunctionWeights> trainFeatureFunctions;
    /**
     * tag的二元转移矩阵，适用于BiGram Feature
     */
    protected int[][] matrix;

    private int[][] totalMatrix;

    private int[][] lastStepMatrix;

    /**
     * 当前执行的总步数
     */
    protected int step;

    public StructuredPerceptronModel()
    {
        featureFunctionTrie = new BinTrie<FeatureFunctionWeights>();
    }

    /**
     * 训练模型用的加载方式
     * @param id2tag
     * @param templateList 
     */
    public StructuredPerceptronModel(String[] id2tag, List<FeatureTemplate> templateList)
    {
        this.id2tag = id2tag;
        tag2id = new HashMap<String, Integer>();
        for (int i=0; i<id2tag.length; i++)
        {
            tag2id.put(id2tag[i], i);
        }
        featureTemplateList = templateList;
        trainFeatureFunctions = new BinTrie<FeatureFunctionWeights>();
        featureFunctionTrie = trainFeatureFunctions;
        matrix = new int[id2tag.length][id2tag.length];
        totalMatrix = new int[id2tag.length][id2tag.length];
        lastStepMatrix = new int[id2tag.length][id2tag.length];
    }

    /**
     * 以指定的trie树结构储存内部特征函数
     * @param featureFunctionTrie
     */
    public StructuredPerceptronModel(ITrie<FeatureFunctionWeights> featureFunctionTrie)
    {
        this.featureFunctionTrie = featureFunctionTrie;
    }

    protected void onLoadTxtFinished()
    {
        // do nothing
    }
    
    /**
     * 添加单个实例进行训练
     * @param instance
     */
    public void train(Table instance)
    {
//        Table instanceForPredict = new Table();
//        instanceForPredict.v = instance.v.clone();
//        for (int i=0; i<instanceForPredict.v.length; i++)
//        {
//            instanceForPredict.v[i] = instance.v[i].clone();
//        }
        Table instanceForPredict = instance.clone();
        tag(instanceForPredict);
        boolean correct = true;
        for (int i=0; i<instanceForPredict.v.length; i++)
        {
            if (!instanceForPredict.getLast(i).equals(instance.getLast(i))) {
                correct = false;
                break;
            }
        }
        if (correct)
            return;
        int lastTag = -1, lastPredicted = -1;
        for (int i=0; i<instance.size(); i++)
        {
            int tag = tag2id.get(instance.getLast(i)).intValue();
            int predictedTag = tag2id.get(instanceForPredict.getLast(i)).intValue();
            if (lastTag != -1)
            {
                int numStepBeforeUpdate = step - lastStepMatrix[lastTag][tag];
                totalMatrix[lastTag][tag] += matrix[lastPredicted][predictedTag]*numStepBeforeUpdate;
                lastStepMatrix[lastTag][tag] = step;
                matrix[lastTag][tag] += 1;
                numStepBeforeUpdate = step - lastStepMatrix[lastPredicted][predictedTag];
                totalMatrix[lastPredicted][predictedTag] += matrix[lastPredicted][predictedTag]*numStepBeforeUpdate;
                lastStepMatrix[lastPredicted][predictedTag] = step;
                matrix[lastPredicted][predictedTag] -= 1;
            }
            if (tag != predictedTag)
            {
                for (FeatureFunctionWeights function : queryFeatureFUncionList(instanceForPredict, i, true))
                {
                    function.update(step, tag, 1);
                    function.update(step, predictedTag, -1);
                }
            }
            lastTag = tag;
            lastPredicted = predictedTag;
        }
        step++;
    }
    
    /**
     * 求每个函数的平均值
     */
    public void average()
    {
        TreeMap<String,FeatureFunctionWeights> map = new TreeMap<String, FeatureFunctionWeights>();
        Set<Map.Entry<String, FeatureFunctionWeights>> entrySet = trainFeatureFunctions.entrySet();
        for (Map.Entry<String, FeatureFunctionWeights> entry : entrySet)
        {
            entry.getValue().average(step);
            map.put(entry.getKey(), entry.getValue());
        }
        featureFunctionTrie.build(map);
        if (matrix != null)
        {
            for (int i=0; i<matrix.length; i++)
            {
                for (int j=0; j<matrix.length; j++)
                {
                    int numStepBeforeUpdate = step - lastStepMatrix[i][j];
                    totalMatrix[i][j] += matrix[i][j]*numStepBeforeUpdate;
                    matrix[i][j] /= step;
                }
            }
        }
    }

    /**
     * 维特比后向算法标注
     *
     * @param table
     */
    public void tag(Table table)
    {
        int size = table.size();
        if (size == 0) return;
        int tagSize = id2tag.length;
        double[][] net = new double[size][tagSize];
        for (int i = 0; i < size; ++i)
        {
            LinkedList<FeatureFunctionWeights> functionList = queryFeatureFUncionList(table, i, false);
            for (int tag = 0; tag < tagSize; ++tag)
            {
                net[i][tag] = computeScore(functionList, tag);
            }
        }

        if (size == 1)
        {
            double maxScore = -1e10;
            int bestTag = 0;
            for (int tag = 0; tag < net[0].length; ++tag)
            {
                if (net[0][tag] > maxScore)
                {
                    maxScore = net[0][tag];
                    bestTag = tag;
                }
            }
            table.setLast(0, id2tag[bestTag]);
            return;
        }

        int[][] from = new int[size][tagSize];
        for (int i = 1; i < size; ++i)
        {
            for (int now = 0; now < tagSize; ++now)
            {
                double maxScore = -1e10;
                for (int pre = 0; pre < tagSize; ++pre)
                {
                    double score = net[i - 1][pre] + matrix[pre][now] + net[i][now];
                    if (score > maxScore)
                    {
                        maxScore = score;
                        from[i][now] = pre;
                    }
                }
                net[i][now] = maxScore;
            }
        }
        // 反向回溯最佳路径
        double maxScore = -1e10;
        int maxTag = 0;
        for (int tag = 0; tag < net[size - 1].length; ++tag)
        {
            if (net[size - 1][tag] > maxScore)
            {
                maxScore = net[size - 1][tag];
                maxTag = tag;
            }
        }

        table.setLast(size - 1, id2tag[maxTag]);
        maxTag = from[size - 1][maxTag];
        for (int i = size - 2; i > 0; --i)
        {
            table.setLast(i, id2tag[maxTag]);
            maxTag = from[i][maxTag];
        }
        table.setLast(0, id2tag[maxTag]);
    }

    /**
     * 根据特征函数计算输出
     * @param table
     * @param current
     * @return
     */
    protected LinkedList<FeatureFunctionWeights> queryFeatureFUncionList(Table table, 
            int current, boolean train)
    {
        LinkedList<FeatureFunctionWeights> functionList = new LinkedList<FeatureFunctionWeights>();
        for (FeatureTemplate featureTemplate : featureTemplateList)
        {
            char[] o = featureTemplate.generateParameter(table, current);
            FeatureFunctionWeights featureFunction = featureFunctionTrie.get(o);
            if (featureFunction == null) 
            {
                if (train)
                {
                    featureFunction = new FeatureFunctionWeights(o,id2tag.length);
                    if (trainFeatureFunctions != null)
                        trainFeatureFunctions.put(o, featureFunction);
                }
                else
                {
                    continue;
                }
            }
            functionList.add(featureFunction);
        }

        return functionList;
    }

    /**
     * 给一系列特征函数结合tag打分
     *
     * @param scoreList
     * @param tag
     * @return
     */
    protected static int computeScore(LinkedList<FeatureFunctionWeights> functionList, int tag)
    {
        int score = 0;
        for (FeatureFunctionWeights function : functionList)
        {
            score += function.w[tag];
        }
        return score;
    }

    @Override
    public void save(DataOutputStream out) throws Exception
    {
        out.writeInt(id2tag.length);
        for (String tag : id2tag)
        {
            out.writeUTF(tag);
        }
        FeatureFunctionWeights[] valueArray = featureFunctionTrie.getValueArray(new FeatureFunctionWeights[0]);
        out.writeInt(valueArray.length);
        for (FeatureFunctionWeights featureFunction : valueArray)
        {
            featureFunction.save(out);
        }
        featureFunctionTrie.save(out);
        out.writeInt(featureTemplateList.size());
        for (FeatureTemplate featureTemplate : featureTemplateList)
        {
            featureTemplate.save(out);
        }
        if (matrix != null)
        {
            out.writeInt(matrix.length);
            for (int[] line : matrix)
            {
                for (int v : line)
                {
                    out.writeInt(v);
                }
            }
        }
        else
        {
            out.writeInt(0);
        }
    }

    @Override
    public boolean load(ByteArray byteArray)
    {
        if (byteArray == null) return false;
        try
        {
            int size = byteArray.nextInt();
            id2tag = new String[size];
            tag2id = new HashMap<String, Integer>(size);
            for (int i = 0; i < id2tag.length; i++)
            {
                id2tag[i] = byteArray.nextUTF();
                tag2id.put(id2tag[i], i);
            }
            FeatureFunctionWeights[] valueArray = new FeatureFunctionWeights[byteArray.nextInt()];
            for (int i = 0; i < valueArray.length; i++)
            {
                valueArray[i] = new FeatureFunctionWeights();
                valueArray[i].load(byteArray);
            }
            featureFunctionTrie.load(byteArray, valueArray);
            size = byteArray.nextInt();
            featureTemplateList = new ArrayList<FeatureTemplate>(size);
            for (int i = 0; i < size; ++i)
            {
                FeatureTemplate featureTemplate = new FeatureTemplate();
                featureTemplate.load(byteArray);
                featureTemplateList.add(featureTemplate);
            }
            size = byteArray.nextInt();
            if (size == 0) return true;
            matrix = new int[size][size];
            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    matrix[i][j] = byteArray.nextInt();
                }
            }
        }
        catch (Exception e)
        {
            logger.warning("缓存载入失败，可能是由于版本变迁带来的不兼容。具体异常是：\n" + TextUtility.exceptionToString(e));
            return false;
        }

        return true;
    }

    /**
     * 加载二进制形式的模型<br>
     * @param path
     * @return
     */
    public static StructuredPerceptronModel loadBin(String path)
    {
        ByteArray byteArray = ByteArray.createByteArray(path);
        if (byteArray == null) return null;
        StructuredPerceptronModel model = new StructuredPerceptronModel();
        if (model.load(byteArray)) return model;
        return null;
    }

    /**
     * 获取某个tag的ID
     * @param tag
     * @return
     */
    public Integer getTagId(String tag)
    {
        return tag2id.get(tag);
    }
}
