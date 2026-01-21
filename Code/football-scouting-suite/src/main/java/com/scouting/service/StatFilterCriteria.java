package com.scouting.service;

/**
 * Una classe di supporto semplice che incapsula una singola richiesta di filtro statistico.
 * Contiene tre informazioni vitali: "Su quale campo voglio filtrare?" (es. Goals), "Qual è il minimo?" e "Qual è il massimo?".
 * La lista di questi oggetti permette all'utente di applicare combinazioni infinite di filtri.
 */

public class StatFilterCriteria {
    private String statName;
    private Double minValue;
    private Double maxValue;

    public StatFilterCriteria(String statName, Double minValue, Double maxValue) {
        this.statName = statName;
        this.minValue = minValue;
        this.maxValue = maxValue;
    }

    public String getStatName() { return statName; }
    public Double getMinValue() { return minValue; }
    public Double getMaxValue() { return maxValue; }
}