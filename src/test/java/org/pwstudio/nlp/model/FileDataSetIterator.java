package org.pwstudio.nlp.model;

import static com.hankcs.hanlp.utility.Predefine.logger;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.LinkedList;
import java.util.List;

import com.hankcs.hanlp.utility.TextUtility;

public class FileDataSetIterator extends DataSetIterator {
    
    private BufferedReader reader;
    private String filename;
    private String line;
    private List<String[]> next;

    public FileDataSetIterator(String filename) {
        this.filename = filename;
        next = new LinkedList<String[]>();
    }
    
    @Override
    public boolean hasNext() {
        try
        {
            if (reader == null)
                reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename)));
            while ((line = reader.readLine()) != null)
            {
                if (line.isEmpty())
                    break;
                String[] features = line.split("\\s");
                next.add(features);
            }
            return !next.isEmpty();
        } catch (IOException e) {
            // TODO 自动生成的 catch 块
            e.printStackTrace();
            return false;
        }
    }

    @Override
    public String[][] next() {
        if (next.isEmpty())
            return null;
        String[][] table = new String[next.size()][];
        table = next.toArray(table);
        next.clear();
        return table;
    }

    @Override
    public void reset() {
        try {
            if (reader != null)
                reader.close();
            reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename)));
        } catch (IOException e) {
            // TODO 自动生成的 catch 块
            e.printStackTrace();
        }
    }

    public void close()
    {
        if (reader == null) return;
        try
        {
            reader.close();
            reader = null;
        }
        catch (IOException e)
        {
            logger.warning("关闭文件失败" + TextUtility.exceptionToString(e));
        }
        return;
    }
}
