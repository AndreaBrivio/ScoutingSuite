package com.scouting.service;

import com.scouting.data.model.Player;
import com.scouting.data.repository.PlayerRepository;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.transaction.annotation.Transactional;

import java.util.ArrayList;
import java.util.List;

@SpringBootTest
@Transactional // Esegue il rollback dopo ogni test per non sporcare il DB
class ScoutingServiceIntegrationTest {

    @Autowired
    private ScoutingService scoutingService;

    @Autowired
    private PlayerRepository playerRepository;

    @BeforeEach
    void setup() {
        playerRepository.deleteAll();
        
        Player p1 = new Player();
        p1.setName("Low Scorer");
        p1.setGoals(2);
        p1.setAssists(1);
        p1.setAge(20);
        
        Player p2 = new Player();
        p2.setName("High Scorer");
        p2.setGoals(20);
        p2.setAssists(10);
        p2.setAge(25);

        playerRepository.saveAll(List.of(p1, p2));
    }

    @Test
    void testFindPlayersByDynamicCriteria() {
        List<StatFilterCriteria> filters = new ArrayList<>();
        
        filters.add(new StatFilterCriteria("goals", 10.0, 30.0));

        List<Player> results = scoutingService.findPlayersByCriteria(
            null, null, null, null, null, null, null, 
            filters
        );

        Assertions.assertEquals(1, results.size());
        Assertions.assertEquals("High Scorer", results.get(0).getName());
    }

    @Test
    void testFindPlayersByMultipleDynamicCriteria() {
        List<StatFilterCriteria> filters = new ArrayList<>();

        filters.add(new StatFilterCriteria("goals", 1.0, null));
        filters.add(new StatFilterCriteria("assists", null, 5.0));

        List<Player> results = scoutingService.findPlayersByCriteria(
            null, null, null, null, null, null, null, 
            filters
        );

        Assertions.assertEquals(1, results.size());
        Assertions.assertEquals("Low Scorer", results.get(0).getName());
    }
    
    @Test
    void testReflectionErrorHandling() {
        List<StatFilterCriteria> filters = new ArrayList<>();

        filters.add(new StatFilterCriteria("campoInesistente", 10.0, 20.0));

        List<Player> results = scoutingService.findPlayersByCriteria(
             null, null, null, null, null, null, null, filters
        );
        
        Assertions.assertEquals(2, results.size());
    }
}