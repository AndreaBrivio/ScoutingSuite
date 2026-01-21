package com.scouting.service;

import java.util.List;

/**
 * Questo Record Java rappresenta un "Parameter Object".
 * Invece di passare 7 o 8 parametri separati al metodo di ricerca (cosa che renderebbe il codice illeggibile e prono a errori,
 * aumentando l'accoppiamento), impacchettiamo tutti i criteri di filtro in questo oggetto immutabile.
 * Agisce come un contratto tra la UI ed il Service: la UI riempie questo oggetto, e il Service lo legge.
 */

public record PlayerFilterRequest(
    Integer minAge,
    Integer maxAge,
    String name,
    String squad,
    String competition,
    String nation,
    String position,
    List<StatFilterCriteria> statFilters) {}