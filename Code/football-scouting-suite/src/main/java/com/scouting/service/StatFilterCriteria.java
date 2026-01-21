package com.scouting.service;

public class StatFilterCriteria {
    private String statName;
    private Double minValue;
    private Double maxValue;

    public StatFilterCriteria(String statName, Double minValue, Double maxValue) {
        this.statName = statName;
        this.minValue = minValue;
        this.maxValue = maxValue;
    }

    // Getters
    public String getStatName() { return statName; }
    public Double getMinValue() { return minValue; }
    public Double getMaxValue() { return maxValue; }
}