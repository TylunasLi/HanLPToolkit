/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/12/9 20:57</create-date>
 *
 * <copyright file="FeatureFunction.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package org.pwstudio.nlp.model.perceptron;

import com.hankcs.hanlp.corpus.io.ByteArray;
import com.hankcs.hanlp.corpus.io.ICacheAble;

import java.io.DataOutputStream;

/**
 * 特征函数，其实是tag.size个特征函数的集合
 * @author TylunasLi
 */
public class FeatureFunctionWeights implements ICacheAble
{
    /**
     * 特征名称
     */
    char[] name;
    /**
     * 权值，按照index对应于tag的id
     */
    int[] w;
    /**
     * 训练时的累计权值
     * 按照index对应于tag的id
     */
    int[] total;
    /**
     * 上次更新时的训练实例步数
     * 按照index对应于tag的id
     */
    int[] lastStep;

    /**
     * 创建一个用于训练的特征函数权重表
     * @param o
     * @param tagSize
     */
    public FeatureFunctionWeights(char[] name, int tagSize)
    {
        this.name = name;
        w = new int[tagSize];
        total = new int[tagSize];
        lastStep = new int[tagSize];
    }

    /**
     * 丛数据源读取时使用的构造函数
     */
    public FeatureFunctionWeights()
    {
    }

    @Override
    public void save(DataOutputStream out) throws Exception
    {
        out.writeInt(name.length);
        for (char c : name)
        {
            out.writeChar(c);
        }
        out.writeInt(w.length);
        for (int v : w)
        {
            out.writeInt(v);
        }
    }

    @Override
    public boolean load(ByteArray byteArray)
    {
        int size = byteArray.nextInt();
        name = new char[size];
        for (int i = 0; i < size; ++i)
        {
            name[i] = byteArray.nextChar();
        }
        size = byteArray.nextInt();
        w = new int[size];
        for (int i = 0; i < size; ++i)
        {
            w[i] = byteArray.nextInt();
        }
        return true;
    }
    
    /**
     * 更新权重方法
     * @param stepCount 当前步数
     * @param tagId 更新哪个类别的权重
     * @param delta 更新量
     */
    public void update(int stepCount, int tagId, int delta)
    {
        if (total == null || lastStep == null)
        {
            total = w.clone();
            lastStep = new int[w.length];
        }
        int numStepBeforeUpdate = stepCount - lastStep[tagId];
        total[tagId] += w[tagId]*numStepBeforeUpdate;
        lastStep[tagId] = stepCount;
        w[tagId] += delta;
    }

    /**
     * 权重求平均
     * @param totalStep 总步数
     */
    public void average(int totalStep)
    {
        if (total == null || lastStep == null)
        {
            return;
        }
        for (int tagId = 0; tagId < w.length; tagId++)
        {
            int numStepBeforeUpdate = totalStep - lastStep[tagId];
            total[tagId] += w[tagId]*numStepBeforeUpdate;
            w[tagId] = total[tagId]/totalStep;
        }
    }

    public char[] getName()
    {
        return name;
    }

    public int weight(int tagId)
    {
        if (tagId > w.length)
            tagId=w.length-1;
        return w[tagId];
    }

}