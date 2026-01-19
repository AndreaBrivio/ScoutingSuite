package com.scouting.service;

import com.scouting.data.model.Player;
import com.scouting.data.repository.PlayerRepository;
import org.springframework.data.jpa.domain.Specification;
import org.springframework.stereotype.Service;
import jakarta.persistence.criteria.Predicate;
import java.util.ArrayList;
import java.util.List;

@Service
public class ScoutingService {
    
    private final PlayerRepository playerRepository;
    
    public ScoutingService(PlayerRepository playerRepository) {
        this.playerRepository = playerRepository;
    }
    
    public List<Player> getAllPlayers() { return playerRepository.findAll(); }
    
    public List<Player> findPlayersByCriteria(
        Integer minAge, Integer maxAge,
        String playerName, String squad, String comp, String nation, String position,
        String statName, Double minValue, Double maxValue
    ) {
        // Lambda cleaner
        return playerRepository.findAll((Specification<Player>) (root, query, cb) -> {
            var predicates = new ArrayList<Predicate>();

            // Helper per check null/empty stringhe
            if (minAge != null) predicates.add(cb.greaterThanOrEqualTo(root.get("age"), minAge));
            if (maxAge != null) predicates.add(cb.lessThanOrEqualTo(root.get("age"), maxAge));
            
            if (isValid(playerName)) predicates.add(cb.like(cb.lower(root.get("name")), "%" + playerName.toLowerCase() + "%"));
            if (isValid(squad)) predicates.add(cb.like(cb.lower(root.get("squad")), "%" + squad.toLowerCase() + "%"));
            
            if (comp != null) predicates.add(cb.equal(root.get("competition"), comp));
            if (nation != null) predicates.add(cb.equal(root.get("nation"), nation));
            if (position != null) predicates.add(cb.equal(root.get("position"), position));

            // Filtro dinamico (Reflection)
            if (isValid(statName) && (minValue != null || maxValue != null)) {
                try {
                    var fieldType = Player.class.getDeclaredField(statName).getType();
                    var isInteger = fieldType.equals(Integer.class);

                    if (Number.class.isAssignableFrom(fieldType)) {
                        if (minValue != null && maxValue != null) {
                            predicates.add(isInteger 
                                ? cb.between(root.get(statName), minValue.intValue(), maxValue.intValue())
                                : cb.between(root.get(statName), minValue, maxValue));
                        } else if (minValue != null) {
                            predicates.add(isInteger
                                ? cb.greaterThanOrEqualTo(root.get(statName), minValue.intValue())
                                : cb.greaterThanOrEqualTo(root.get(statName), minValue));
                        } else if (maxValue != null) {
                            predicates.add(isInteger
                                ? cb.lessThanOrEqualTo(root.get(statName), maxValue.intValue())
                                : cb.lessThanOrEqualTo(root.get(statName), maxValue));
                        }
                    }
                } catch (NoSuchFieldException e) {
                    // Log silenzioso o gestione custom
                    System.err.println("Invalid field: " + statName);
                }
            }

            return cb.and(predicates.toArray(Predicate[]::new));
        });
    }

    private boolean isValid(String s) {
        return s != null && !s.isEmpty();
    }
}