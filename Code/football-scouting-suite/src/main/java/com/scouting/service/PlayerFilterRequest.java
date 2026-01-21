package com.scouting.service;

import java.util.List;

// Usiamo un Record (Java 14+) che è immutabile e conciso.
// Questo funge da "Parameter Object" (Refactoring Pattern)
public record PlayerFilterRequest(
    Integer minAge,
    Integer maxAge,
    String name,
    String squad,
    String competition,
    String nation,
    String position,
    List<StatFilterCriteria> statFilters
) {}