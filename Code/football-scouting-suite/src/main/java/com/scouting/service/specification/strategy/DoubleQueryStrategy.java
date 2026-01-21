package com.scouting.service.specification.strategy;

import jakarta.persistence.criteria.CriteriaBuilder;
import jakarta.persistence.criteria.Expression;
import jakarta.persistence.criteria.Path;
import jakarta.persistence.criteria.Predicate;
import org.springframework.stereotype.Component;

/**
 * Questa è l'implementazione concreta della strategia per i numeri decimali (Double).
 * Gestisce statistiche come "xG" o "Goals per 90". La sua responsabilità è dire al database come confrontare
 * numeri con la virgola, gestendo anche i casi limite (es. cercare valori tra min e max, o solo maggiori di min).
 * Essendo un Component Spring, viene rilevata automaticamente e iniettata nella Factory.
 */

@Component
public class DoubleQueryStrategy implements StatQueryStrategy {

    @Override
    public boolean supports(Class<?> fieldType) {
        return fieldType.equals(Double.class) || fieldType.equals(double.class);
    }

    @Override
    public Predicate buildPredicate(CriteriaBuilder cb, Path<?> path, Double min, Double max) {
        // FIX: .as() restituisce Expression, non Path
        Expression<Double> doubleExpr = path.as(Double.class);

        if (min != null && max != null) {
            return cb.between(doubleExpr, min, max);
        } else if (min != null) {
            return cb.greaterThanOrEqualTo(doubleExpr, min);
        } else if (max != null) {
            return cb.lessThanOrEqualTo(doubleExpr, max);
        }
        return null;
    }
}