package com.scouting.service.specification.strategy;

import jakarta.persistence.criteria.CriteriaBuilder;
import jakarta.persistence.criteria.Path;
import jakarta.persistence.criteria.Predicate;

/**
 * Questa interfaccia definisce il "Contratto" per Strategy Pattern.
 * Nel nostro sistema, dobbiamo filtrare dati di tipi diversi (Interi, Decimali) senza riempire il codice di controlli "if".
 * Ogni classe che implementa questa interfaccia promette di saper gestire un tipo specifico di dato e di saperlo trasformare
 * in un predicato JPA. È la chiave per rendere il sistema estensibile: per supportare nuovi tipi,
 * basta creare una nuova classe che implementi questa interfaccia, senza toccare quelle esistenti.
 */

public interface StatQueryStrategy {
    
    boolean supports(Class<?> fieldType);

    Predicate buildPredicate(CriteriaBuilder cb, Path<?> path, Double min, Double max);
}