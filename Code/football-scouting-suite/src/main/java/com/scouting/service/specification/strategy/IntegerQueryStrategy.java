package com.scouting.service.specification.strategy;

import jakarta.persistence.criteria.CriteriaBuilder;
import jakarta.persistence.criteria.Expression;
import jakarta.persistence.criteria.Path;
import jakarta.persistence.criteria.Predicate;
import org.springframework.stereotype.Component;

/**
 * Implementazione concreta della strategia per i numeri interi (Integer).
 * Si occupa di statistiche "discrete" come Gol segnati, Età o Presenze. Sebbene la logica sembri simile a quella dei Double,
 * a livello di database i tipi sono diversi e richiedono trattamenti distinti.
 */

@Component
public class IntegerQueryStrategy implements StatQueryStrategy {

    @Override
    public boolean supports(Class<?> fieldType) {
        return fieldType.equals(Integer.class) || fieldType.equals(int.class);
    }

    @Override
    public Predicate buildPredicate(CriteriaBuilder cb, Path<?> path, Double min, Double max) {
        Expression<Integer> intExpr = path.as(Integer.class);
        
        if (min != null && max != null) {
            return cb.between(intExpr, min.intValue(), max.intValue());
        } else if (min != null) {
            return cb.greaterThanOrEqualTo(intExpr, min.intValue());
        } else if (max != null) {
            return cb.lessThanOrEqualTo(intExpr, max.intValue());
        }
        return null;
    }
}