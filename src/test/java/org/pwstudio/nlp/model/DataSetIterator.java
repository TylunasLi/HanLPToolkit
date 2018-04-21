package org.pwstudio.nlp.model;

import java.util.Iterator;

/**
 * 用于训练感知机模型的实例迭代器
 * @author TylunasLi
 *
 */
public abstract class DataSetIterator implements Iterator<String[][]> {

    @Override
    public void remove() {
        throw new IllegalStateException ("This Iterator does not support remove().");
    }

    public abstract void reset();

    public abstract void close();

}
