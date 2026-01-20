package com.scouting.data.repository;

import com.scouting.data.model.Player;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.orm.jpa.DataJpaTest;

import java.util.Optional;

@DataJpaTest // Configura un database H2 in-memory reale (No Mock, No Stub)
class PlayerRepositoryTest {

    @Autowired
    private PlayerRepository playerRepository;

    // DRIVER: Verifica il salvataggio e recupero reale dal DB
    @Test
    void testSaveAndFindPlayer() {
        Player p = new Player();
        p.setName("Roberto Baggio");
        p.setNation("Italy");
        p.setGoals(10);
        
        Player savedPlayer = playerRepository.save(p);
        
        Assertions.assertNotNull(savedPlayer.getId(), "L'ID dovrebbe essere generato automaticamente dal DB");
        
        Optional<Player> retrieved = playerRepository.findById(savedPlayer.getId());
        Assertions.assertTrue(retrieved.isPresent());
        Assertions.assertEquals("Roberto Baggio", retrieved.get().getName());
        Assertions.assertEquals(10, retrieved.get().getGoals());
    }
}