package com.prome.parser;

import java.util.List;
import java.util.ArrayList;

public class DataPipeline {
    private List<String> records = new ArrayList<>();

    public void addRecord(String record) {
        this.records.add(record);
    }

    public int getRecordCount() {
        return this.records.size();
    }
}