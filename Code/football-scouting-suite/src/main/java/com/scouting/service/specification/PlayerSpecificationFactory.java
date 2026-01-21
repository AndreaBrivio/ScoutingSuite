package com.scouting.service.specification;

import com.scouting.data.model.Player;
import com.scouting.service.PlayerFilterRequest;
import com.scouting.service.StatFilterCriteria;
import com.scouting.service.specification.strategy.StatQueryStrategy; // Import
import jakarta.persistence.criteria.*;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.data.jpa.domain.Specification;
import org.springframework.stereotype.Component;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;

/**
 * Questa classe risolve un problema classico: come filtrare una tabella con decine di colonne diverse
 * senza scrivere una catena infinita di "if-else". Per farlo, implementa l'<strong>Open/Closed Principle</strong>:
 * il sistema è aperto a nuovi filtri ma chiuso alla modifica del codice esistente.
 * <p>
 * Utilizziamo <strong>Strategy Pattern</strong>: invece di codificare la logica di ogni tipo di dato qui dentro,
 * la Factory possiede una lista di strategie (es. una per i numeri interi, una per i decimali).
 * A runtime, quando arriva una richiesta, la Factory seleziona automaticamente la strategia corretta per quel campo
 * e delega a lei la costruzione del predicato SQL. Questo rende il codice estremamente modulare ed estensibile.
 */

@Component
public class PlayerSpecificationFactory {

    private static final Logger logger = LogManager.getLogger(PlayerSpecificationFactory.class);
    
    private final List<StatQueryStrategy> strategies;

    public PlayerSpecificationFactory(List<StatQueryStrategy> strategies) {
        this.strategies = strategies;
    }

    public Specification<Player> createSpecification(PlayerFilterRequest req) {
        return (root, query, cb) -> {
            List<Predicate> predicates = new ArrayList<>();

            addRangeFilter(predicates, cb, root.get("age"), req.minAge(), req.maxAge());
            addLikeFilter(predicates, cb, root.get("name"), req.name());
            addLikeFilter(predicates, cb, root.get("squad"), req.squad());
            addEqualFilter(predicates, cb, root.get("competition"), req.competition());
            addEqualFilter(predicates, cb, root.get("nation"), req.nation());
            addEqualFilter(predicates, cb, root.get("position"), req.position());

            if (req.statFilters() != null) {
                for (StatFilterCriteria filter : req.statFilters()) {
                    addStatFilter(predicates, cb, root, filter);
                }
            }

            return cb.and(predicates.toArray(new Predicate[0]));
        };
    }

    private void addRangeFilter(List<Predicate> predicates, CriteriaBuilder cb, Path<Integer> path, Integer min, Integer max) {
        if (min != null) predicates.add(cb.greaterThanOrEqualTo(path, min));
        if (max != null) predicates.add(cb.lessThanOrEqualTo(path, max));
    }
    private void addLikeFilter(List<Predicate> predicates, CriteriaBuilder cb, Expression<String> path, String value) {
        if (value != null && !value.isEmpty()) predicates.add(cb.like(cb.lower(path), "%" + value.toLowerCase() + "%"));
    }
    private void addEqualFilter(List<Predicate> predicates, CriteriaBuilder cb, Path<?> path, Object value) {
        if (value != null) predicates.add(cb.equal(path, value));
    }

    private void addStatFilter(List<Predicate> predicates, CriteriaBuilder cb, Root<Player> root, StatFilterCriteria filter) {
        String statName = filter.getStatName();
        Double min = filter.getMinValue();
        Double max = filter.getMaxValue();

        if (statName == null || statName.isEmpty() || min == null && max == null) return;

        try {
            Field statField = Player.class.getDeclaredField(statName);
            Class<?> fieldType = statField.getType();

            // PATTERN DELEGATION: Cerca la strategia giusta
            for (StatQueryStrategy strategy : strategies) {
                if (strategy.supports(fieldType)) {
                    Predicate p = strategy.buildPredicate(cb, root.get(statName), min, max);
                    if (p != null) predicates.add(p);
                    return; // Strategia trovata ed eseguita, esci.
                }
            }
            
            // Se nessuna strategia supporta il tipo (es. String), non fare nulla o logga
            
        } catch (NoSuchFieldException | SecurityException e) {
            logger.warn("Campo statistica non trovato: {}", statName);
        } catch (Exception e) {
            logger.error("Errore strategia filtro: {}", e.getMessage());
        }
    }
}